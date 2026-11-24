"""QmtLocalProvider / QmtDataDir 测试。

QMT 缓存目录中的 ``.DAT`` 是 64 字节定长二进制记录。测试用 :mod:`struct` 按
真实布局（小端、无对齐填充）独立打包合成文件，避免与被测的 ``KLINE_DTYPE``
定义互相「自证」：

    "<10I Q 4I"
    ├─ reserved0, reserved1
    ├─ timestamp（Unix 秒）
    ├─ open / high / low / close（×1000）
    ├─ reserved2, volume（手）, reserved3
    ├─ amount（元，u64）
    └─ open_interest, reserved4, reserved5, pre_close（×1000）

覆盖点：目录发现、附属文件过滤、二进制解析、preClose 除权因子与噪声阈值、
分钟/周/月聚合、Provider 标准 schema 与异常映射。
"""

from __future__ import annotations

import struct
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml4t.data.core.exceptions import DataNotAvailableError, SymbolNotFoundError
from ml4t.data.providers.qmt_datadir import (
    QmtDataDir,
    QmtFormatError,
    adjustment_factors,
    split_symbol,
)
from ml4t.data.providers.qmt_local_provider import QmtLocalProvider

# A 股无夏令时，固定 +08:00，避免测试依赖系统 tzdata
CST = timezone(timedelta(hours=8))
PRICE_SCALE = 1000


# ---------------------------------------------------------------------------
# 二进制 fixture 构造
# ---------------------------------------------------------------------------


def epoch(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    """返回指定 Asia/Shanghai 时刻的 Unix 秒。"""
    return int(datetime(year, month, day, hour, minute, tzinfo=CST).timestamp())


def _units(price: float) -> int:
    """价格 → 整数毫元（文件内的 ×1000 表示）。"""
    return int(round(price * PRICE_SCALE))


def pack_bar(
    ts: int,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: int,
    amount: int,
    pre_close: float,
) -> bytes:
    """按真实 64 字节布局打包一条 K 线记录。"""
    payload = struct.pack(
        "<10IQ4I",
        0,
        0,
        ts,
        _units(open),
        _units(high),
        _units(low),
        _units(close),
        0,
        volume,
        0,
        amount,
        0,
        0,
        0,
        _units(pre_close),
    )
    assert len(payload) == 64
    return payload


def bar(
    ts: int,
    close: float,
    pre_close: float | None = None,
    volume: int = 1000,
    amount: int = 1_000_000,
) -> dict:
    """构造一条 OHLC 合法的对称 bar。"""
    return {
        "ts": ts,
        "open": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": volume,
        "amount": amount,
        "pre_close": close if pre_close is None else pre_close,
    }


def sample_daily_bars() -> list[dict]:
    """5 个交易日，2024-01-05 发生一次除权（9.0 / 10.0 = 0.9）。"""
    days = [(2024, 1, 2), (2024, 1, 3), (2024, 1, 4), (2024, 1, 5), (2024, 1, 8)]
    closes = [10.0, 10.0, 10.0, 9.1, 9.2]
    pre_closes = [10.0, 10.0, 10.0, 9.0, 9.1]
    return [
        bar(epoch(*day), close, pre_close)
        for day, close, pre_close in zip(days, closes, pre_closes, strict=True)
    ]


def sample_minute_bars(day: tuple[int, int, int] = (2024, 1, 2)) -> list[dict]:
    """A 股一个完整交易日的 1 分钟线（09:30 + 09:31-11:30 + 13:01-15:00）。"""
    minutes = list(range(9 * 60 + 30, 11 * 60 + 31)) + list(range(13 * 60 + 1, 15 * 60 + 1))
    bars: list[dict] = []
    previous = 10.0
    for index, minute_of_day in enumerate(minutes):
        price = 10.0 + index * 0.001
        bars.append(
            bar(
                epoch(*day, hour=minute_of_day // 60, minute=minute_of_day % 60),
                price,
                pre_close=previous,
                volume=1,
                amount=1000,
            )
        )
        previous = price
    return bars


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def qmt_root(tmp_path: Path) -> Path:
    """构造含 SH/SZ/BJ × 日线/分钟线 目录的空 datadir。"""
    root = tmp_path / "datadir"
    for market in ("SH", "SZ", "BJ"):
        for period_dir in ("86400", "60"):
            (root / market / period_dir).mkdir(parents=True)
    return root


def write_bars(
    root: Path,
    market: str,
    period_dir: str,
    code: str,
    bars: list[dict],
    suffix: str = ".DAT",
) -> Path:
    """把 bar 列表写成二进制 K 线文件。"""
    path = root / market / period_dir / f"{code}{suffix}"
    path.write_bytes(b"".join(pack_bar(**item) for item in bars))
    return path


def write_sectors(root: Path) -> None:
    """构造 Sector 板块缓存：两个分类 + xml + 空板块 + 跳分类重名板块。"""
    base = root / "Sector"
    index_dir = base / "指数成分股板块"
    industry_dir = base / "申万一级行业板块"
    mine_dir = base / "我的"
    concept_dir = base / "T概念"
    for directory in (index_dir, industry_dir, mine_dir, concept_dir):
        directory.mkdir(parents=True)

    (index_dir / "sectorConfig.xml").write_text("<CustomSector/>", encoding="utf-8")
    (index_dir / "沪深300").write_text("600000.SH,600009.SH,600010.SH", encoding="utf-8")
    (index_dir / "000809").write_text("600108.SH,600127.SH", encoding="utf-8")
    (industry_dir / "SW1医药生物").write_text("600055.SH,600056.SH", encoding="utf-8")
    (mine_dir / "我的自选").write_text("513310.SH,159941.SZ", encoding="utf-8")
    (mine_dir / "空板块").write_bytes(b"")
    # 同名板块出现在两个分类中，不带分类前缀时应报歧义
    (concept_dir / "医药生物").write_text("600055.SH", encoding="utf-8")
    (industry_dir / "医药生物").write_text("600056.SH", encoding="utf-8")
    # 与指数板块同名，用于验证歧义时优先「指数成分股板块」
    (concept_dir / "沪深300").write_text("000001.SZ", encoding="utf-8")


@pytest.fixture
def datadir(qmt_root: Path) -> QmtDataDir:
    write_bars(qmt_root, "SZ", "86400", "000001", sample_daily_bars())
    write_bars(qmt_root, "SZ", "60", "000001", sample_minute_bars())
    return QmtDataDir(qmt_root)


# ---------------------------------------------------------------------------
# 代码解析与目录发现
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("600000.SH", ("SH", "600000")),
        ("sh600000", ("SH", "600000")),
        ("SH.600000", ("SH", "600000")),
        (" 000001.sz ", ("SZ", "000001")),
        ("BJ430047", ("BJ", "430047")),
    ],
)
def test_split_symbol_variants(raw: str, expected: tuple[str, str]) -> None:
    """三种市场后缀写法都能拆成 (市场, 代码)。"""
    assert split_symbol(raw) == expected


@pytest.mark.parametrize("raw", ["", "600000", "600000.XX", ".SH"])
def test_split_symbol_rejects_invalid(raw: str) -> None:
    """缺市场标识或市场未知时给出明确错误。"""
    with pytest.raises(ValueError):
        split_symbol(raw)


def test_datadir_rejects_dir_without_markets(tmp_path: Path) -> None:
    """不含 SH/SZ/BJ 子目录的目录不是合法 datadir。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no market subdirectory"):
        QmtDataDir(empty)


def test_discover_explicit_root(qmt_root: Path) -> None:
    """显式传入 root 时直接使用，不做扫描。"""
    assert QmtDataDir.discover(qmt_root).root == qmt_root


def test_discover_uses_env_var(qmt_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """未显式传入时优先读取环境变量。"""
    monkeypatch.setenv("ML4T_QMT_DATA_DIR", str(qmt_root))
    assert QmtDataDir.discover().root == qmt_root


def test_discover_scans_client_layout(tmp_path: Path) -> None:
    """自动扫描能识别 ``*QMT*/userdata_mini/datadir`` 布局。"""
    client = tmp_path / "国金证券QMT交易端"
    (client / "userdata_mini" / "datadir" / "SZ" / "86400").mkdir(parents=True)
    found = QmtDataDir.discover(search_roots=[tmp_path])
    assert found.root == client / "userdata_mini" / "datadir"


def test_discover_raises_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """找不到 datadir 时报错并提示环境变量用法。"""
    monkeypatch.delenv("ML4T_QMT_DATA_DIR", raising=False)
    monkeypatch.delenv("QMT_DATA_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="ML4T_QMT_DATA_DIR"):
        QmtDataDir.discover(search_roots=[tmp_path / "absent"])


def test_kline_path_supports_both_suffixes(qmt_root: Path) -> None:
    """``.DAT`` 与 ``.dat`` 两种后缀都能定位（Windows 客户端不区分大小写）。"""
    datadir = QmtDataDir(qmt_root)
    upper = write_bars(qmt_root, "SZ", "86400", "000001", sample_daily_bars())
    assert datadir.kline_path("SZ", "000001") == upper

    lower = qmt_root / "SZ" / "86400" / "000002.dat"
    lower.write_bytes(b"".join(pack_bar(**item) for item in sample_daily_bars()))
    assert datadir.kline_path("SZ", "000002") == lower


def test_available_symbols_filters_auxiliary_files(qmt_root: Path) -> None:
    """过滤 ``<code>_<4位>.DAT`` 附属文件与不足一条记录的空文件。"""
    write_bars(qmt_root, "SZ", "86400", "000001", sample_daily_bars())
    write_bars(qmt_root, "SZ", "86400", "000003", sample_daily_bars(), suffix=".dat")
    (qmt_root / "SZ" / "86400" / "000002.DAT").write_bytes(b"")
    (qmt_root / "SZ" / "86400" / "000001_9000.DAT").write_bytes(b"\x00" * 64)

    datadir = QmtDataDir(qmt_root)
    assert datadir.available_symbols("1d") == ["000001.SZ", "000003.SZ"]
    assert datadir.available_symbols("1d", market="SH") == []


def test_available_symbols_traded_only_drops_placeholders(qmt_root: Path) -> None:
    """traded_only 剔除零成交的代码占位文件。"""
    write_bars(qmt_root, "SZ", "86400", "000001", sample_daily_bars())
    # 客户端为未下载行情的品种写入的占位记录：价格恒 100、无成交
    placeholder = [bar(epoch(2024, 1, 2), 100.0, volume=0, amount=0)]
    write_bars(qmt_root, "SZ", "86400", "190000", placeholder)

    datadir = QmtDataDir(qmt_root)

    assert datadir.available_symbols("1d") == ["000001.SZ", "190000.SZ"]
    assert datadir.available_symbols("1d", traded_only=True) == ["000001.SZ"]


def test_available_symbols_rejects_aggregated_period(qmt_root: Path) -> None:
    """聚合周期不存在于磁盘，应抛出 ValueError。"""
    datadir = QmtDataDir(qmt_root)

    with pytest.raises(ValueError, match="not stored on disk"):
        datadir.available_symbols("5m")
    with pytest.raises(ValueError, match="not stored on disk"):
        datadir.available_symbols("5m", traded_only=True)


def test_coverage_reports_symbols_rows_and_dates(qmt_root: Path) -> None:
    """coverage 汇总标的数、记录条数与起止日期。"""
    write_bars(qmt_root, "SZ", "86400", "000001", sample_daily_bars())
    write_bars(qmt_root, "SZ", "86400", "000002", sample_daily_bars()[:2])

    (stats,) = QmtDataDir(qmt_root).coverage("1d")

    assert stats.market == "SZ"
    assert stats.period == "1d"
    assert stats.symbols == 2
    assert stats.traded == 2
    assert stats.rows_min == 2
    assert stats.rows_max == 5
    assert stats.first_date == date(2024, 1, 2)
    assert stats.last_date == date(2024, 1, 8)
    assert stats.unreadable == 0


def test_coverage_counts_placeholders_and_stale_symbols(qmt_root: Path) -> None:
    """占位文件不计入 traded；结束日落后的标的按 stale_days 统计。"""
    write_bars(qmt_root, "SZ", "86400", "000001", sample_daily_bars())
    write_bars(qmt_root, "SZ", "86400", "000002", sample_daily_bars()[:1])
    write_bars(
        qmt_root,
        "SZ",
        "86400",
        "190000",
        [bar(epoch(2024, 1, 2), 100.0, volume=0, amount=0)],
    )

    (relaxed,) = QmtDataDir(qmt_root).coverage("1d", stale_days=30)
    assert relaxed.symbols == 3
    assert relaxed.traded == 2
    assert relaxed.stale == 0

    (strict,) = QmtDataDir(qmt_root).coverage("1d", stale_days=3)
    assert strict.stale == 2


def test_coverage_skips_markets_without_data(qmt_root: Path) -> None:
    """没有有效文件的市场不出现在结果中。"""
    write_bars(qmt_root, "SH", "86400", "600000", sample_daily_bars())

    stats = QmtDataDir(qmt_root).coverage("1d")

    assert [item.market for item in stats] == ["SH"]


def test_coverage_rejects_aggregated_period(qmt_root: Path) -> None:
    """coverage 只接受原生周期。"""
    with pytest.raises(ValueError, match="not stored on disk"):
        QmtDataDir(qmt_root).coverage("1w")


# ---------------------------------------------------------------------------
# 板块 / 行业成分股
# ---------------------------------------------------------------------------


@pytest.fixture
def sector_datadir(qmt_root: Path) -> QmtDataDir:
    write_sectors(qmt_root)
    return QmtDataDir(qmt_root)


def test_sector_kinds_and_names(sector_datadir: QmtDataDir) -> None:
    """分类与板块枚举，跳过 xml 和 0 字节空板块。"""
    assert sector_datadir.sector_kinds() == [
        "T概念",
        "我的",
        "指数成分股板块",
        "申万一级行业板块",
    ]

    names = sector_datadir.sector_names()
    assert "指数成分股板块/沪深300" in names
    assert "我的/我的自选" in names
    assert "我的/空板块" not in names
    assert not any(name.endswith(".xml") for name in names)
    assert sector_datadir.sector_names(kind="我的") == ["我的/我的自选"]


def test_sector_symbols_by_name_and_kind(sector_datadir: QmtDataDir) -> None:
    """按板块名或分类前缀读取成分股；同名时优先指数分类。"""
    assert sector_datadir.sector_symbols("沪深300") == ["600000.SH", "600009.SH", "600010.SH"]
    assert sector_datadir.sector_symbols("T概念/沪深300") == ["000001.SZ"]
    assert sector_datadir.sector_symbols("000809") == ["600108.SH", "600127.SH"]
    assert sector_datadir.sector_symbols("我的/我的自选") == ["513310.SH", "159941.SZ"]
    assert sector_datadir.sector_symbols("SW1医药生物", kind="申万一级行业板块") == [
        "600055.SH",
        "600056.SH",
    ]


def test_sector_symbols_dedupes_and_uppercases(qmt_root: Path) -> None:
    """成分股去重并统一为大写 code.MARKET。"""
    write_sectors(qmt_root)
    target = qmt_root / "Sector" / "我的" / "带空格"
    target.write_text(" 600000.sh,600000.SH ,600001.SH, ", encoding="utf-8")

    assert QmtDataDir(qmt_root).sector_symbols("带空格") == ["600000.SH", "600001.SH"]


def test_sector_symbols_ambiguous_name(sector_datadir: QmtDataDir) -> None:
    """跨分类重名时要求加分类前缀。"""
    with pytest.raises(ValueError, match="ambiguous"):
        sector_datadir.sector_symbols("医药生物")

    assert sector_datadir.sector_symbols("T概念/医药生物") == ["600055.SH"]
    assert sector_datadir.sector_symbols("医药生物", kind="T概念") == ["600055.SH"]


def test_sector_symbols_missing(sector_datadir: QmtDataDir) -> None:
    """不存在的板块报 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError, match="not found"):
        sector_datadir.sector_symbols("不存在的板块")
    with pytest.raises(FileNotFoundError, match="not found"):
        sector_datadir.sector_symbols("我的/不存在")


def test_sector_without_cache_directory(qmt_root: Path) -> None:
    """datadir 无 Sector 目录时返回空枚举并报错。"""
    datadir = QmtDataDir(qmt_root)

    assert datadir.sector_root is None
    assert datadir.sector_kinds() == []
    assert datadir.sector_names() == []
    with pytest.raises(FileNotFoundError, match="no Sector directory"):
        datadir.sector_symbols("沪深300")


# ---------------------------------------------------------------------------
# 市场分类板块（沪市基金 / 深市A股 / 创业板 …，无对应文件）
# ---------------------------------------------------------------------------


MARKET_SAMPLES: tuple[tuple[str, str], ...] = (
    ("SH", "600000"),
    ("SH", "688001"),
    ("SH", "510300"),
    ("SH", "511880"),
    ("SH", "000001"),
    ("SH", "113001"),
    ("SZ", "000001"),
    ("SZ", "300001"),
    ("SZ", "159919"),
    ("SZ", "399001"),
    ("SZ", "123001"),
    ("BJ", "830001"),
)


@pytest.fixture
def market_datadir(qmt_root: Path) -> QmtDataDir:
    """写入覆盖各市场分类前缀的日线缓存。"""
    for market, code in MARKET_SAMPLES:
        write_bars(qmt_root, market, "86400", code, sample_daily_bars())
    return QmtDataDir(qmt_root)


def test_market_sector_names_listed(market_datadir: QmtDataDir) -> None:
    """市场分类板块名单包含常用分类且升序。"""
    names = market_datadir.market_sector_names()

    assert "沪市基金" in names
    assert "深市基金" in names
    assert "沪深A股" in names
    assert names == sorted(names)


def test_market_sector_filters_market_and_prefix(market_datadir: QmtDataDir) -> None:
    """市场分类按「市场 + 代码前缀」从缓存中筛选。"""
    assert market_datadir.sector_symbols("沪市基金") == ["510300.SH", "511880.SH"]
    assert market_datadir.sector_symbols("深市基金") == ["159919.SZ"]
    assert market_datadir.sector_symbols("沪深基金") == [
        "510300.SH",
        "511880.SH",
        "159919.SZ",
    ]
    assert market_datadir.sector_symbols("沪市A股") == ["600000.SH", "688001.SH"]
    assert market_datadir.sector_symbols("沪市主板") == ["600000.SH"]
    assert market_datadir.sector_symbols("深市A股") == ["000001.SZ", "300001.SZ"]
    assert market_datadir.sector_symbols("创业板") == ["300001.SZ"]
    assert market_datadir.sector_symbols("科创板") == ["688001.SH"]
    assert market_datadir.sector_symbols("沪市指数") == ["000001.SH"]
    assert market_datadir.sector_symbols("深市指数") == ["399001.SZ"]
    assert market_datadir.sector_symbols("沪市转债") == ["113001.SH"]
    assert market_datadir.sector_symbols("深市转债") == ["123001.SZ"]
    assert market_datadir.sector_symbols("北交所") == ["830001.BJ"]


def test_market_sector_without_sector_dir(qmt_root: Path) -> None:
    """无 Sector 目录时市场分类仍可用（不依赖板块文件）。"""
    write_bars(qmt_root, "SH", "86400", "510300", sample_daily_bars())
    datadir = QmtDataDir(qmt_root)

    assert datadir.sector_root is None
    assert datadir.sector_symbols("沪市基金") == ["510300.SH"]


def test_sector_file_wins_over_market_rules(sector_datadir: QmtDataDir, qmt_root: Path) -> None:
    """同名文件板块优先于代码前缀规则。"""
    (qmt_root / "Sector" / "我的" / "沪市基金").write_text("999999.SH", encoding="utf-8")
    write_bars(qmt_root, "SH", "86400", "510300", sample_daily_bars())

    assert sector_datadir.sector_symbols("沪市基金") == ["999999.SH"]


# ---------------------------------------------------------------------------
# 二进制解析
# ---------------------------------------------------------------------------


def test_read_daily_bars_schema(datadir: QmtDataDir) -> None:
    """日线输出列、类型与时间戳时区符合约定。"""
    frame = datadir.read_bars("000001.SZ")

    assert frame.columns == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "suspend_flag",
    ]
    assert frame.schema["timestamp"] == pl.Datetime(time_unit="ms", time_zone="Asia/Shanghai")
    assert frame.height == 5
    assert frame["timestamp"][0] == datetime(2024, 1, 2, tzinfo=CST)
    assert frame["close"].to_list() == [10.0, 10.0, 10.0, 9.1, 9.2]
    assert frame["volume"].to_list() == [1000.0] * 5
    assert frame["amount"].to_list() == [1_000_000.0] * 5


def test_read_bars_missing_file(datadir: QmtDataDir) -> None:
    """本地无缓存文件时抛 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError, match="kline file not found"):
        datadir.read_bars("600000.SH")


def test_read_bars_empty_file(qmt_root: Path) -> None:
    """不足一条记录的残文件视为格式错误。"""
    (qmt_root / "SZ" / "86400" / "000001.DAT").write_bytes(b"\x00" * 10)
    datadir = QmtDataDir(qmt_root)
    with pytest.raises(QmtFormatError, match="empty QMT kline file"):
        datadir.read_bars("000001.SZ")


def test_read_bars_ignores_trailing_partial_record(qmt_root: Path) -> None:
    """尾部不足 64 字节的残留字节被忽略，不影响已有记录。"""
    path = write_bars(qmt_root, "SZ", "86400", "000001", sample_daily_bars())
    path.write_bytes(path.read_bytes() + b"\x7f" * 7)
    assert QmtDataDir(qmt_root).read_bars("000001.SZ").height == 5


def test_read_bars_deduplicates_keeping_last(qmt_root: Path) -> None:
    """同一时间戳重复出现时保留最后一条（客户端重写场景）。"""
    ts = epoch(2024, 1, 2)
    write_bars(
        qmt_root,
        "SZ",
        "86400",
        "000001",
        [bar(ts, 10.0), bar(ts, 11.0)],
    )
    frame = QmtDataDir(qmt_root).read_bars("000001.SZ")
    assert frame.height == 1
    assert frame["close"].to_list() == [11.0]


def test_read_bars_date_range_is_inclusive(datadir: QmtDataDir) -> None:
    """start/end 都是闭区间。"""
    frame = datadir.read_bars("000001.SZ", start="2024-01-03", end="2024-01-05")
    assert frame["timestamp"].dt.date().to_list() == [
        datetime(2024, 1, 3).date(),
        datetime(2024, 1, 4).date(),
        datetime(2024, 1, 5).date(),
    ]


def test_read_bars_compact_date_format(datadir: QmtDataDir) -> None:
    """支持 ``YYYYMMDD`` 写法。"""
    frame = datadir.read_bars("000001.SZ", start="20240105", end="20240108")
    assert frame.height == 2


def test_suspend_flag_marks_zero_volume_rows(qmt_root: Path) -> None:
    """volume 与 amount 同时为 0 的行标记为停牌。"""
    write_bars(
        qmt_root,
        "SZ",
        "86400",
        "000001",
        [
            bar(epoch(2024, 1, 2), 10.0),
            bar(epoch(2024, 1, 3), 10.0, volume=0, amount=0),
        ],
    )
    frame = QmtDataDir(qmt_root).read_bars("000001.SZ")
    assert frame["suspend_flag"].to_list() == [0, 1]


# ---------------------------------------------------------------------------
# 复权
# ---------------------------------------------------------------------------


def test_adjustment_factors_treat_one_tick_gap_as_noise() -> None:
    """恰好一个最小价位（0.01 元）的差值必须视为噪声，而非除权事件。

    回归点：float64 下 ``10.01 - 10.0 = 0.010000000000000497 > 0.01``，用浮点
    直接比较会把每 10 派 0.1 元的舍入噪声当成除权，注入约 0.1% 的假收益。
    """
    close = np.array([10.0, 10.0, 10.0])
    pre_close = np.array([10.0, 10.01, 9.0])

    factors = adjustment_factors(close, pre_close)

    assert factors[0] == 1.0
    assert factors[1] == 1.0
    assert factors[2] == pytest.approx(0.9)


def test_adjustment_factors_accepts_gap_above_one_tick() -> None:
    """超过一个最小价位（0.011 元）即视为除权事件。"""
    close = np.array([10.0, 10.0])
    pre_close = np.array([10.0, 10.011])
    factors = adjustment_factors(close, pre_close)
    assert factors[1] == pytest.approx(10.011 / 10.0)


def test_adjustment_none_keeps_raw_prices(datadir: QmtDataDir) -> None:
    """不复权时价格与文件原始值一致。"""
    frame = datadir.read_bars("000001.SZ", dividend_type="none")
    assert frame["close"].to_list() == [10.0, 10.0, 10.0, 9.1, 9.2]


def test_adjustment_front_keeps_latest_price(datadir: QmtDataDir) -> None:
    """前复权：最新一根价格不变，历史价格按未来因子缩放。"""
    frame = datadir.read_bars("000001.SZ", dividend_type="front")
    closes = frame["close"].to_list()

    assert closes[-1] == pytest.approx(9.2)
    assert closes[:3] == pytest.approx([9.0, 9.0, 9.0])
    assert closes[3] == pytest.approx(9.1)


def test_adjustment_back_keeps_first_price(datadir: QmtDataDir) -> None:
    """后复权：首端价格不变，后续价格按已发生因子放大。"""
    frame = datadir.read_bars("000001.SZ", dividend_type="back")
    closes = frame["close"].to_list()

    assert closes[0] == pytest.approx(10.0)
    assert closes[2] == pytest.approx(10.0)
    assert closes[3] == pytest.approx(9.1 / 0.9)
    assert closes[4] == pytest.approx(9.2 / 0.9)


def test_adjustment_scales_ohlc_consistently(datadir: QmtDataDir) -> None:
    """复权等比缩放 OHLC，且不改变 volume / amount。"""
    raw = datadir.read_bars("000001.SZ", dividend_type="none")
    front = datadir.read_bars("000001.SZ", dividend_type="front")

    assert front["volume"].to_list() == raw["volume"].to_list()
    assert front["amount"].to_list() == raw["amount"].to_list()
    for index in range(raw.height):
        for column in ("open", "high", "low", "close"):
            expected = raw[column][index] * (front["close"][index] / raw["close"][index])
            assert front[column][index] == pytest.approx(expected)


def test_adjustment_rejects_unknown_dividend_type(datadir: QmtDataDir) -> None:
    """复权类型非法时报错。"""
    with pytest.raises(ValueError, match="unsupported dividend_type"):
        datadir.read_bars("000001.SZ", dividend_type="split")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 周期派生
# ---------------------------------------------------------------------------


def test_resample_intraday_bar_counts(datadir: QmtDataDir) -> None:
    """一个完整交易日的 241 根分钟线聚合为 48/16/8/4 根。"""
    assert datadir.read_bars("000001.SZ", period="1m").height == 241
    assert datadir.read_bars("000001.SZ", period="5m").height == 48
    assert datadir.read_bars("000001.SZ", period="15m").height == 16
    assert datadir.read_bars("000001.SZ", period="30m").height == 8
    assert datadir.read_bars("000001.SZ", period="1h").height == 4


def test_resample_intraday_labels_are_bar_end(datadir: QmtDataDir) -> None:
    """集合竞价归一到 09:31，N 分钟线按 bar 结束时间标注。"""
    five = datadir.read_bars("000001.SZ", period="5m")
    labels = [moment.strftime("%H:%M") for moment in five["timestamp"].to_list()]

    assert labels[0] == "09:35"
    assert labels[23] == "11:30"
    assert labels[24] == "13:05"
    assert labels[-1] == "15:00"


def test_resample_intraday_aggregates_first_bar(datadir: QmtDataDir) -> None:
    """首根 5 分钟 bar 覆盖 09:30-09:35 共 6 根分钟线。"""
    one_minute = datadir.read_bars("000001.SZ", period="1m")
    five = datadir.read_bars("000001.SZ", period="5m")
    first = five.row(0, named=True)

    assert first["open"] == pytest.approx(one_minute["open"][0])
    assert first["close"] == pytest.approx(one_minute["close"][5])
    assert first["high"] == pytest.approx(one_minute["high"][:6].max())
    assert first["low"] == pytest.approx(one_minute["low"][:6].min())
    assert first["volume"] == pytest.approx(6.0)


def test_resample_conserves_volume_and_amount(datadir: QmtDataDir) -> None:
    """聚合不丢量：各周期 volume / amount 合计与 1m 一致。"""
    one_minute = datadir.read_bars("000001.SZ", period="1m")
    total_volume = one_minute["volume"].sum()
    total_amount = one_minute["amount"].sum()

    for period in ("5m", "15m", "30m", "1h"):
        resampled = datadir.read_bars("000001.SZ", period=period)
        assert resampled["volume"].sum() == pytest.approx(total_volume)
        assert resampled["amount"].sum() == pytest.approx(total_amount)


def test_resample_weekly_labels_last_trading_day(qmt_root: Path) -> None:
    """周线按周一为起点分组，标签取区间内最后一个交易日。"""
    days = [
        (2024, 1, 29),
        (2024, 1, 30),
        (2024, 1, 31),
        (2024, 2, 1),
        (2024, 2, 2),
        (2024, 2, 5),
        (2024, 2, 6),
    ]
    write_bars(
        qmt_root,
        "SZ",
        "86400",
        "000001",
        [bar(epoch(*day), 10.0 + index) for index, day in enumerate(days)],
    )
    frame = QmtDataDir(qmt_root).read_bars("000001.SZ", period="1w")

    assert frame.height == 2
    assert frame["timestamp"].dt.date().to_list() == [
        datetime(2024, 2, 2).date(),
        datetime(2024, 2, 6).date(),
    ]
    assert frame["open"].to_list() == [10.0, 15.0]
    assert frame["close"].to_list() == [14.0, 16.0]
    assert frame["volume"].to_list() == [5000.0, 2000.0]


def test_resample_monthly_splits_at_month_boundary(qmt_root: Path) -> None:
    """月线按自然月分组，标签取当月最后一个交易日。"""
    days = [(2024, 1, 30), (2024, 1, 31), (2024, 2, 1)]
    write_bars(
        qmt_root,
        "SZ",
        "86400",
        "000001",
        [bar(epoch(*day), 10.0) for day in days],
    )
    frame = QmtDataDir(qmt_root).read_bars("000001.SZ", period="1mon")

    assert frame["timestamp"].dt.date().to_list() == [
        datetime(2024, 1, 31).date(),
        datetime(2024, 2, 1).date(),
    ]
    assert frame["volume"].to_list() == [2000.0, 1000.0]


def test_unsupported_period_raises(datadir: QmtDataDir) -> None:
    """未支持的周期给出可读错误。"""
    with pytest.raises(ValueError, match="unsupported period"):
        datadir.read_bars("000001.SZ", period="2h")


# ---------------------------------------------------------------------------
# Provider 接口一致性
# ---------------------------------------------------------------------------


def test_provider_name_and_capabilities(qmt_root: Path) -> None:
    """provider 名称为 qmt_local，声明支持分钟级且无需 API Key。"""
    provider = QmtLocalProvider(data_dir=qmt_root)
    capabilities = provider.capabilities()

    assert provider.name == "qmt_local"
    assert provider.data_dir == qmt_root
    assert capabilities.supports_intraday is True
    assert capabilities.requires_api_key is False
    assert capabilities.rate_limit == QmtLocalProvider.DEFAULT_RATE_LIMIT


def test_provider_rejects_unknown_dividend_type(qmt_root: Path) -> None:
    """构造时就校验复权类型，避免延迟到取数阶段报错。"""
    with pytest.raises(ValueError, match="unsupported dividend_type"):
        QmtLocalProvider(data_dir=qmt_root, dividend_type="split")


def test_provider_fetch_ohlcv_schema(qmt_root: Path) -> None:
    """fetch_ohlcv 输出 canonical schema + 额外列，顺序与 QmtProvider 一致。"""
    write_bars(qmt_root, "SZ", "86400", "000001", sample_daily_bars())
    provider = QmtLocalProvider(data_dir=qmt_root, dividend_type="none")
    frame = provider.fetch_ohlcv("000001.SZ", "2024-01-02", "2024-01-08", "daily")

    assert frame.columns == [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "suspend_flag",
    ]
    assert frame["symbol"].unique().to_list() == ["000001.SZ"]
    assert frame["close"].dtype == pl.Float64
    assert frame.height == 5


def test_provider_normalizes_symbol(qmt_root: Path) -> None:
    """``SH600004`` 这类写法在输出里统一成 ``600004.SH``。"""
    write_bars(qmt_root, "SH", "86400", "600004", sample_daily_bars())
    provider = QmtLocalProvider(data_dir=qmt_root, dividend_type="none")
    frame = provider.fetch_ohlcv("sh600004", "2024-01-02", "2024-01-08", "daily")

    assert frame["symbol"].unique().to_list() == ["600004.SH"]


def test_provider_default_is_front_adjusted(qmt_root: Path) -> None:
    """默认口径与 QmtProvider 一致，为前复权。"""
    write_bars(qmt_root, "SZ", "86400", "000001", sample_daily_bars())
    provider = QmtLocalProvider(data_dir=qmt_root)
    frame = provider.fetch_ohlcv("000001.SZ", "2024-01-02", "2024-01-08", "daily")

    assert frame["close"].to_list()[-1] == pytest.approx(9.2)
    assert frame["close"].to_list()[0] == pytest.approx(9.0)


@pytest.mark.parametrize(
    ("frequency", "expected_rows"),
    [
        ("daily", 5),
        ("1d", 5),
        ("1day", 5),
        ("weekly", 2),
        ("monthly", 1),
    ],
)
def test_provider_accepts_frequency_aliases(
    qmt_root: Path, frequency: str, expected_rows: int
) -> None:
    """日/周/月周期的别名与 QmtProvider.FREQUENCY_MAP 对齐。"""
    write_bars(qmt_root, "SZ", "86400", "000001", sample_daily_bars())
    provider = QmtLocalProvider(data_dir=qmt_root, dividend_type="none")
    frame = provider.fetch_ohlcv("000001.SZ", "2024-01-02", "2024-01-08", frequency)
    assert frame.height == expected_rows


@pytest.mark.parametrize(
    ("frequency", "expected_rows"),
    [("minute", 241), ("1m", 241), ("5m", 48), ("15m", 16), ("30m", 8), ("1h", 4), ("60m", 4)],
)
def test_provider_supports_intraday_aliases(
    qmt_root: Path, frequency: str, expected_rows: int
) -> None:
    """分钟级别名（含 1h / 60m）都映射到本地 1 分钟缓存。"""
    write_bars(qmt_root, "SZ", "60", "000001", sample_minute_bars())
    provider = QmtLocalProvider(data_dir=qmt_root, dividend_type="none")
    frame = provider.fetch_ohlcv("000001.SZ", "2024-01-02", "2024-01-02", frequency)
    assert frame.height == expected_rows


def test_provider_empty_range_keeps_schema(qmt_root: Path) -> None:
    """区间内无数据时返回带完整 schema 的空 DataFrame。"""
    write_bars(qmt_root, "SZ", "86400", "000001", sample_daily_bars())
    provider = QmtLocalProvider(data_dir=qmt_root)
    frame = provider.fetch_ohlcv("000001.SZ", "1990-01-01", "1990-12-31", "daily")

    assert frame.is_empty()
    assert frame.columns == [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "suspend_flag",
    ]


def test_provider_symbol_not_found(qmt_root: Path) -> None:
    """本地无缓存文件时映射为 SymbolNotFoundError 并附排查提示。"""
    provider = QmtLocalProvider(data_dir=qmt_root)
    with pytest.raises(SymbolNotFoundError) as excinfo:
        provider.fetch_ohlcv("600000.SH", "2024-01-02", "2024-01-08", "daily")

    assert excinfo.value.symbol == "600000.SH"
    assert "QMT" in excinfo.value.details.get("hint", "")


def test_provider_corrupt_file_maps_to_data_not_available(qmt_root: Path) -> None:
    """文件格式异常映射为 DataNotAvailableError（stage=parse）。"""
    (qmt_root / "SZ" / "86400" / "000001.DAT").write_bytes(b"\x00" * 8)
    provider = QmtLocalProvider(data_dir=qmt_root)
    with pytest.raises(DataNotAvailableError) as excinfo:
        provider.fetch_ohlcv("000001.SZ", "2024-01-02", "2024-01-08", "daily")

    assert excinfo.value.details.get("stage") == "parse"


def test_provider_rejects_unknown_frequency(qmt_root: Path) -> None:
    """未知频率直接报错，不静默降级为日线。"""
    write_bars(qmt_root, "SZ", "86400", "000001", sample_daily_bars())
    provider = QmtLocalProvider(data_dir=qmt_root)
    with pytest.raises(ValueError, match="unsupported frequency"):
        provider.fetch_ohlcv("000001.SZ", "2024-01-02", "2024-01-08", "2h")


def test_provider_batch_tolerates_partial_failure(qmt_root: Path) -> None:
    """批量读取中单标的失败不中断，成功的标的照常返回。"""
    write_bars(qmt_root, "SZ", "86400", "000001", sample_daily_bars())
    write_bars(qmt_root, "SH", "86400", "600004", sample_daily_bars())
    provider = QmtLocalProvider(data_dir=qmt_root, dividend_type="none")

    frame = provider.fetch_batch_ohlcv(
        ["000001.SZ", "600004.SH", "999999.SZ"], "2024-01-02", "2024-01-08"
    )

    assert sorted(frame["symbol"].unique().to_list()) == ["000001.SZ", "600004.SH"]
    assert frame.height == 10
    # 长格式按 [symbol, timestamp] 排序：整体 timestamp 不单调，需分组检查
    assert frame["symbol"].is_sorted()
    for symbol in frame["symbol"].unique(maintain_order=True):
        assert frame.filter(pl.col("symbol") == symbol)["timestamp"].is_sorted()


def test_provider_batch_all_failed_returns_empty(qmt_root: Path) -> None:
    """全部标的失败时返回空 DataFrame 而不是抛异常。"""
    provider = QmtLocalProvider(data_dir=qmt_root)
    frame = provider.fetch_batch_ohlcv(["999999.SZ"], "2024-01-02", "2024-01-08")
    assert frame.is_empty()
    assert "amount" in frame.columns


def test_provider_batch_concat_with_single_fetch(qmt_root: Path) -> None:
    """批量与单标的取数结果 schema 一致，可直接 concat。"""
    write_bars(qmt_root, "SZ", "86400", "000001", sample_daily_bars())
    provider = QmtLocalProvider(data_dir=qmt_root, dividend_type="none")

    single = provider.fetch_ohlcv("000001.SZ", "2024-01-02", "2024-01-03", "daily")
    batch = provider.fetch_batch_ohlcv(["000001.SZ"], "2024-01-02", "2024-01-03")

    assert pl.concat([single, batch]).height == 4


def test_provider_available_symbols_and_coverage(qmt_root: Path) -> None:
    """traded_only 与 coverage 从 provider 透传到解析层。"""
    write_bars(qmt_root, "SZ", "86400", "000001", sample_daily_bars())
    write_bars(
        qmt_root,
        "SZ",
        "86400",
        "190000",
        [bar(epoch(2024, 1, 2), 100.0, volume=0, amount=0)],
    )
    provider = QmtLocalProvider(data_dir=qmt_root)

    assert provider.available_symbols() == ["000001.SZ", "190000.SZ"]
    assert provider.available_symbols(traded_only=True) == ["000001.SZ"]

    (stats,) = provider.coverage("1d")
    assert stats.symbols == 2
    assert stats.traded == 1
    assert stats.last_date == date(2024, 1, 8)


def test_provider_sector_queries(qmt_root: Path) -> None:
    """板块/行业查询与 QmtProvider 同名方法签名一致。"""
    write_sectors(qmt_root)
    provider = QmtLocalProvider(data_dir=qmt_root)

    expected = ["600000.SH", "600009.SH", "600010.SH"]
    assert provider.get_stock_list_in_sector("沪深300") == expected
    assert provider.get_sector("000300.SH") == expected
    assert provider.get_sector("SH000300") == expected
    assert provider.get_sector("沪深300") == expected
    assert provider.get_sector("000809") == ["600108.SH", "600127.SH"]
    assert provider.get_industry("SW1医药生物") == ["600055.SH", "600056.SH"]
    assert provider.sector_symbols("我的/我的自选") == ["513310.SH", "159941.SZ"]
    assert "指数成分股板块/沪深300" in provider.sector_names()
    assert "指数成分股板块" in provider.sector_kinds()


def test_provider_get_sector_unknown_code(qmt_root: Path) -> None:
    """未收录的指数代码直接按文件名查，找不到时报错。"""
    write_sectors(qmt_root)
    provider = QmtLocalProvider(data_dir=qmt_root)

    with pytest.raises(FileNotFoundError, match="not found"):
        provider.get_sector("999999.SH")


def test_provider_weight_in_index_not_supported(qmt_root: Path) -> None:
    """指数权重存于 LevelDB，明确报 NotImplementedError。"""
    provider = QmtLocalProvider(data_dir=qmt_root)

    with pytest.raises(NotImplementedError, match="LevelDB"):
        provider.get_weight_in_index("000300.SH", "600000.SH")


def test_provider_market_sectors(qmt_root: Path) -> None:
    """市场分类板块可经 get_stock_list_in_sector / get_sector 访问。"""
    write_bars(qmt_root, "SH", "86400", "510300", sample_daily_bars())
    write_bars(qmt_root, "SH", "86400", "600000", sample_daily_bars())
    write_bars(qmt_root, "SZ", "86400", "159919", sample_daily_bars())
    provider = QmtLocalProvider(data_dir=qmt_root)

    assert provider.get_stock_list_in_sector("沪市基金") == ["510300.SH"]
    assert provider.get_stock_list_in_sector("深市基金") == ["159919.SZ"]
    assert provider.get_sector("沪市基金") == ["510300.SH"]
    assert provider.get_stock_list_in_sector("沪市A股") == ["600000.SH"]
    assert "沪市基金" in provider.market_sector_names()
