"""QMT（MiniQMT）本地缓存 datadir 二进制解析层。

本模块只做「目录发现 + 二进制解析 + 复权/周期派生」，不依赖 QMT 客户端、
不依赖 xtquant，供 :class:`~ml4t.data.providers.qmt_local_provider.QmtLocalProvider`
读取客户端已下载到本地的历史行情缓存。

数据流：
    <datadir>/<SH|SZ|BJ>/<周期秒>/<代码>.DAT
        → 按 64 字节定长记录 np.frombuffer（小端）
        → 时间戳 u32 秒 → Datetime[ms, Asia/Shanghai]，价格 /1000
        → 用文件内 preClose 推导除权因子 → 前/后复权
        → 周期派生（5m/15m/30m/1h 由 1m 聚合，周/月由日线聚合）
        → pl.DataFrame[timestamp, open, high, low, close, volume, amount, suspend_flag]

二进制布局（日线 / 分钟线，均为 64 字节定长记录）：

    ==========  ======  ====================================
    偏移        类型    字段
    ==========  ======  ====================================
    8           u32     时间戳（Unix 秒）
    12          u32     开盘价 ×1000
    16          u32     最高价 ×1000
    20          u32     最低价 ×1000
    24          u32     收盘价 ×1000
    32          u32     成交量（手）
    40          u64     成交额（元）
    48          u32     持仓量
    60          u32     除权参考价（昨收，×1000）
    ==========  ======  ====================================

数值语义（已用真实缓存交叉验证：minute 末根 volume × close × 100 与 amount 精确相等）：

- ``volume`` 单位为「手」，``amount`` 单位为「元」，两者与 xtdata 输出一致，均不做缩放
- 日线与分钟线共用同一布局；tick 为 144 字节记录，本模块不涉及

除权（复权）语义：

- 记录中的 ``preClose`` 是**交易所除权参考价**，即 ``(昨收 - 每股派息) / (1 + 送转比例)``
- 因此除权日因子 ``f = preClose_t / close_{t-1}``，非除权日为 1
- 前复权：``adj_t = price_t × Π_{s>t} f_s``（末端价格不变，与 xtdata 前复权口径一致）
- 后复权：``adj_t = price_t / Π_{s<=t} f_s``（首端价格不变）
- ``|preClose - close_{t-1}|`` 不超过一个最小价位（0.01 元）的差值视为舍入噪声，
  默认不作为除权事件处理，避免把 ±0.01 的噪声放大成假收益。比较在整数毫元
  上进行（浮点比较会把 10.01-10.0 这类恰好一个价位的差异误判成事件）
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import ClassVar, Final, Literal, NamedTuple

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger()

TZ: Final[str] = "Asia/Shanghai"
# A 股无夏令时，时间戳 → 日期换算用固定 +08:00，避免依赖系统 tzdata
_CST: Final[timezone] = timezone(timedelta(hours=8), "CST")
RECORD_SIZE: Final[int] = 64
PRICE_SCALE: Final[float] = 1000.0

# 除权事件最小价差：不超过一个最小价位（0.01 元）的差值视为舍入噪声
DEFAULT_EX_RIGHTS_THRESHOLD: Final[float] = 0.01

# 64 字节定长记录布局；itemsize 必须恰好等于 64
KLINE_DTYPE: Final[np.dtype] = np.dtype(
    [
        ("reserved0", "<u4"),
        ("reserved1", "<u4"),
        ("timestamp", "<u4"),
        ("open", "<u4"),
        ("high", "<u4"),
        ("low", "<u4"),
        ("close", "<u4"),
        ("reserved2", "<u4"),
        ("volume", "<u4"),
        ("reserved3", "<u4"),
        ("amount", "<u8"),
        ("open_interest", "<u4"),
        ("reserved4", "<u4"),
        ("reserved5", "<u4"),
        ("pre_close", "<u4"),
    ]
)
if KLINE_DTYPE.itemsize != RECORD_SIZE:  # pragma: no cover - 结构体定义自检
    raise RuntimeError(f"KLINE_DTYPE itemsize={KLINE_DTYPE.itemsize} != {RECORD_SIZE}")

DividendType = Literal["front", "back", "none"]

# 指数成分股所在的板块分类；同名板块跨分类重名时优先使用它（见 sector_symbols）
INDEX_SECTOR_KIND: Final[str] = "指数成分股板块"

# --- 市场分类板块的代码规则 ---
# 客户端左侧的「沪市基金」「深市A股」等分类由「市场 + 代码前缀」动态生成，
# datadir 中没有对应文件，只能从本地已缓存的标的中筛选。
_SH_STOCKS: Final[tuple[str, ...]] = ("600", "601", "603", "605", "688", "689")
_SZ_STOCKS: Final[tuple[str, ...]] = ("000", "001", "002", "003", "300", "301", "302")
_SH_MAIN: Final[tuple[str, ...]] = ("600", "601", "603", "605")
_SZ_MAIN: Final[tuple[str, ...]] = ("000", "001", "002", "003")
_SH_FUNDS: Final[tuple[str, ...]] = ("50", "51", "52", "53", "55", "56", "58")
_SZ_FUNDS: Final[tuple[str, ...]] = ("15", "16", "18")
_SH_INDEX: Final[tuple[str, ...]] = ("000", "880", "999")
_SZ_INDEX: Final[tuple[str, ...]] = ("399", "980")
# 转债前缀需精确到 3 位：沪深 12x/13x 段还混有企业债、公司债，2 位前缀会误判
_SH_CONVERT: Final[tuple[str, ...]] = ("110", "111", "113", "118", "132")
_SZ_CONVERT: Final[tuple[str, ...]] = ("123", "125", "126", "127", "128", "129")

MarketSectorRules = tuple[tuple[str, tuple[str, ...]], ...]

# 板块名 → ((市场, 代码前缀), ...)；同一分类内多个市场按元组顺序拼接
MARKET_SECTORS: Final[dict[str, MarketSectorRules]] = {
    "沪深A股": (("SH", _SH_STOCKS), ("SZ", _SZ_STOCKS)),
    "沪市A股": (("SH", _SH_STOCKS),),
    "深市A股": (("SZ", _SZ_STOCKS),),
    "沪市主板": (("SH", _SH_MAIN),),
    "深市主板": (("SZ", _SZ_MAIN),),
    "创业板": (("SZ", ("300", "301", "302")),),
    "科创板": (("SH", ("688", "689")),),
    "北交所": (("BJ", ("",)),),
    "沪深基金": (("SH", _SH_FUNDS), ("SZ", _SZ_FUNDS)),
    "沪市基金": (("SH", _SH_FUNDS),),
    "深市基金": (("SZ", _SZ_FUNDS),),
    "沪深指数": (("SH", _SH_INDEX), ("SZ", _SZ_INDEX)),
    "沪市指数": (("SH", _SH_INDEX),),
    "深市指数": (("SZ", _SZ_INDEX),),
    "沪深转债": (("SH", _SH_CONVERT), ("SZ", _SZ_CONVERT)),
    "沪市转债": (("SH", _SH_CONVERT),),
    "深市转债": (("SZ", _SZ_CONVERT),),
}


class QmtFormatError(Exception):
    """QMT 缓存文件格式异常（记录长度不足、时间戳非法等）。"""


class _SymbolStat(NamedTuple):
    """单个缓存文件的首尾扫描结果。"""

    symbol: str
    rows: int
    first_ts: int
    last_ts: int
    traded: bool


@dataclass(frozen=True)
class CoverageStats:
    """单个「市场 × 周期」的本地缓存覆盖统计。

    Attributes:
        market: 市场目录名（SH/SZ/BJ）
        period: 原生周期（'1d' / '1m'）
        symbols: 有效文件数（至少一条完整记录）
        traded: 有成交的标的数（首尾记录至少一条 volume/amount 非 0）
        rows_min: 记录条数最小值
        rows_median: 记录条数中位数
        rows_max: 记录条数最大值
        first_date: 全部标的中最早的起始日
        last_date: 全部标中最新的结束日
        median_last_date: 结束日的中位数（反映整体新鲜度）
        stale: 结束日落后 ``last_date`` 超过 ``stale_days`` 的标的数
        unreadable: 无法读取（被占用/权限不足）的文件数
    """

    market: str
    period: str
    symbols: int
    traded: int
    rows_min: int
    rows_median: float
    rows_max: int
    first_date: date
    last_date: date
    median_last_date: date
    stale: int
    unreadable: int


def split_symbol(symbol: str) -> tuple[str, str]:
    """拆分证券代码为 (市场, 纯代码)。

    支持 ``'600000.SH'``、``'SH600000'``、``'SH.600000'`` 三种写法。

    Args:
        symbol: 证券代码

    Returns:
        ``(market, code)``，market 为 ``SH`` / ``SZ`` / ``BJ``，code 为纯数字代码

    Raises:
        ValueError: 代码为空、缺少市场标识或市场未知
    """
    raw = symbol.strip().upper()
    if not raw:
        raise ValueError("symbol cannot be empty")

    if "." in raw:
        left, right = raw.split(".", 1)
        if left.isdigit():
            code, market = left, right
        else:
            market, code = left, right
    elif len(raw) > 2 and raw[:2] in QmtDataDir.MARKETS:
        market, code = raw[:2], raw[2:]
    else:
        raise ValueError(f"symbol must include market suffix (e.g. '600000.SH'), got {symbol!r}")

    if market not in QmtDataDir.MARKETS:
        raise ValueError(f"unsupported market {market!r} (expected one of {QmtDataDir.MARKETS})")
    if not code:
        raise ValueError(f"empty security code in {symbol!r}")
    return market, code


class QmtDataDir:
    """QMT 本地缓存根目录句柄。

    Attributes:
        MARKETS: datadir 下的市场目录名
        ENV_VARS: 自动发现时读取的环境变量（按顺序尝试）
        PERIOD_DIRS: 直接在磁盘上存在的周期目录（秒）
        INTRADAY_MINUTES: 需要由 1 分钟线聚合的周期
        DAILY_DERIVED: 需要由日线聚合的周期
    """

    MARKETS: ClassVar[tuple[str, ...]] = ("SH", "SZ", "BJ")
    ENV_VARS: ClassVar[tuple[str, ...]] = ("ML4T_QMT_DATA_DIR", "QMT_DATA_DIR")
    PERIOD_DIRS: ClassVar[dict[str, str]] = {"1d": "86400", "1m": "60"}
    INTRADAY_MINUTES: ClassVar[dict[str, int]] = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}
    DAILY_DERIVED: ClassVar[tuple[str, ...]] = ("1w", "1mon")

    def __init__(self, root: str | Path) -> None:
        """初始化 datadir 句柄。

        Args:
            root: datadir 根目录（含 SH/SZ/BJ 子目录的那一层）

        Raises:
            FileNotFoundError: 目录不存在或不含任何市场子目录
        """
        path = Path(root)
        if not path.is_dir():
            raise FileNotFoundError(f"QMT datadir not found: {path}")
        markets = [m for m in self.MARKETS if (path / m).is_dir()]
        if not markets:
            raise FileNotFoundError(f"QMT datadir has no market subdirectory (SH/SZ/BJ): {path}")
        self._root = path

    def __repr__(self) -> str:
        return f"QmtDataDir({str(self._root)!r})"

    @property
    def root(self) -> Path:
        """返回 datadir 根目录。"""
        return self._root

    # ------------------------------------------------------------------
    # 目录发现
    # ------------------------------------------------------------------

    @classmethod
    def discover(
        cls,
        root: str | Path | None = None,
        search_roots: list[str | Path] | None = None,
    ) -> QmtDataDir:
        """定位 QMT datadir。

        优先级：显式 ``root`` → 环境变量（``ML4T_QMT_DATA_DIR`` / ``QMT_DATA_DIR``）
        → 在 ``search_roots`` 或各本地磁盘根下查找 ``*QMT*/datadir`` 与
        ``*QMT*/userdata_mini/datadir``（Windows 客户端两种布局都存在）。

        Args:
            root: 显式指定的 datadir
            search_roots: 自动扫描的起始目录（默认为各本地磁盘根）

        Returns:
            可用的 QmtDataDir

        Raises:
            FileNotFoundError: 未找到任何可用的 datadir
        """
        if root is not None:
            return cls(root)

        candidates: list[Path] = []
        for env_var in cls.ENV_VARS:
            value = os.environ.get(env_var)
            if value:
                candidates.append(Path(value))

        if not candidates:
            bases = [Path(p) for p in search_roots] if search_roots else _local_drive_roots()
            for base in bases:
                candidates.extend(_qmt_candidates_under(base))

        tried: list[Path] = []
        for candidate in candidates:
            tried.append(candidate)
            if not candidate.is_dir():
                continue
            try:
                return cls(candidate)
            except FileNotFoundError:
                continue

        raise FileNotFoundError(
            "QMT datadir not found. 请显式传入 data_dir，或设置环境变量 "
            f"{' / '.join(cls.ENV_VARS)}。已尝试: {[str(p) for p in tried[:5]]}"
        )

    # ------------------------------------------------------------------
    # 路径与标的枚举
    # ------------------------------------------------------------------

    def kline_path(self, market: str, code: str, period: str = "1d") -> Path:
        """返回周期 K 线文件路径。

        Args:
            market: 市场目录名（SH/SZ/BJ）
            code: 纯代码，如 '600000'
            period: 原生周期 '1d' / '1m'

        Returns:
            文件路径（存在性由调用方校验）

        Raises:
            ValueError: period 不是原生周期
        """
        period_dir = self._native_period_dir(period)
        directory = self._root / market / period_dir
        for suffix in (".DAT", ".dat"):
            candidate = directory / f"{code}{suffix}"
            if candidate.is_file():
                return candidate
        return directory / f"{code}.DAT"

    def available_symbols(
        self,
        period: str = "1d",
        market: str | None = None,
        traded_only: bool = False,
    ) -> list[str]:
        """列出本地已有数据的标的（``'600000.SH'`` 形式，升序）。

        会过滤 ``<code>_<4位>.DAT`` 这类板块/财务附属文件，以及不足一条记录的
        空文件。

        注意：QMT 客户端会为大量未下载行情的品种写入「代码占位」文件（收盘价
        恒为 100.000、成交量为 0），这类文件同样会被列出。``traded_only=True``
        时读取每个文件的首尾记录，剔除首尾都无成交的占位文件（代价是数万次
        小随机读，日线约需十几秒）。

        Args:
            period: 原生周期 '1d' / '1m'
            market: 限定市场（SH/SZ/BJ），None 表示全部
            traded_only: 只保留首尾至少一条有成交的标的

        Returns:
            标的代码列表

        Raises:
            ValueError: period 不是原生周期
        """
        if traded_only:
            stats, unreadable = self._scan_symbols(period, market)
            if unreadable:
                logger.warning("Unreadable QMT kline files skipped", count=unreadable)
            return sorted(stat.symbol for stat in stats if stat.traded)

        period_dir = self._native_period_dir(period)
        markets = [market.upper()] if market else list(self.MARKETS)
        symbols: list[str] = []
        for market_name in markets:
            directory = self._root / market_name / period_dir
            if not directory.is_dir():
                continue
            for entry in directory.iterdir():
                if not entry.is_file() or entry.stat().st_size < RECORD_SIZE:
                    continue
                stem = entry.stem
                if not stem.isdigit():
                    continue
                symbols.append(f"{stem}.{market_name}")
        return sorted(symbols)

    def coverage(
        self,
        period: str = "1d",
        market: str | None = None,
        stale_days: int = 30,
    ) -> list[CoverageStats]:
        """统计本地缓存覆盖情况（按市场）。

        读取每个文件的首尾记录，汇总标的数、记录条数分位数、起止日期与新鲜度，
        用于回答「本地还缺哪些数据」。无有效文件的目录不会出现在结果中。

        Args:
            period: 原生周期 '1d' / '1m'
            market: 限定市场（SH/SZ/BJ），None 表示全部
            stale_days: 结束日落后最新日超过该天数的标的计为陈旧

        Returns:
            每个有数据的市场一条统计，按市场名升序

        Raises:
            ValueError: period 不是原生周期
        """
        self._native_period_dir(period)
        markets = [market.upper()] if market else list(self.MARKETS)
        results: list[CoverageStats] = []
        for market_name in markets:
            stats, unreadable = self._scan_symbols(period, market_name)
            if not stats:
                continue
            rows = np.array([stat.rows for stat in stats])
            firsts = np.array([stat.first_ts for stat in stats])
            lasts = np.array([stat.last_ts for stat in stats])
            latest = int(lasts.max())
            cutoff = latest - stale_days * 86400
            results.append(
                CoverageStats(
                    market=market_name,
                    period=period,
                    symbols=len(stats),
                    traded=sum(1 for stat in stats if stat.traded),
                    rows_min=int(rows.min()),
                    rows_median=float(np.median(rows)),
                    rows_max=int(rows.max()),
                    first_date=_ts_to_date(int(firsts.min())),
                    last_date=_ts_to_date(latest),
                    median_last_date=_ts_to_date(int(np.median(lasts))),
                    stale=int((lasts < cutoff).sum()),
                    unreadable=unreadable,
                )
            )
        return results

    def _native_period_dir(self, period: str) -> str:
        """把原生周期映射为目录名。"""
        period_dir = self.PERIOD_DIRS.get(period)
        if period_dir is None:
            raise ValueError(
                f"period {period!r} is not stored on disk; native periods: "
                f"{sorted(self.PERIOD_DIRS)}"
            )
        return period_dir

    def _scan_symbols(
        self,
        period: str,
        market: str | None = None,
    ) -> tuple[list[_SymbolStat], int]:
        """读取每个缓存文件的首尾记录，返回明细与不可读文件数。"""
        period_dir = self._native_period_dir(period)
        markets = [market.upper()] if market else list(self.MARKETS)
        stats: list[_SymbolStat] = []
        unreadable = 0
        for market_name in markets:
            directory = self._root / market_name / period_dir
            if not directory.is_dir():
                continue
            for entry in directory.iterdir():
                if not entry.is_file():
                    continue
                stem = entry.stem
                if not stem.isdigit():
                    continue
                try:
                    size = entry.stat().st_size
                    if size < RECORD_SIZE:
                        continue
                    count = size // RECORD_SIZE
                    with entry.open("rb") as handle:
                        head = np.frombuffer(handle.read(RECORD_SIZE), dtype=KLINE_DTYPE)[0]
                        handle.seek((count - 1) * RECORD_SIZE)
                        tail = np.frombuffer(handle.read(RECORD_SIZE), dtype=KLINE_DTYPE)[0]
                except OSError:
                    unreadable += 1
                    continue
                stats.append(
                    _SymbolStat(
                        symbol=f"{stem}.{market_name}",
                        rows=count,
                        first_ts=int(head["timestamp"]),
                        last_ts=int(tail["timestamp"]),
                        traded=bool(
                            head["volume"] or head["amount"] or tail["volume"] or tail["amount"]
                        ),
                    )
                )
        return stats, unreadable

    # ------------------------------------------------------------------
    # 板块 / 行业成分股（Sector/ 缓存）
    # ------------------------------------------------------------------

    @property
    def sector_root(self) -> Path | None:
        """``<datadir>/Sector`` 目录（板块成分股缓存），不存在时返回 None。"""
        directory = self._root / "Sector"
        return directory if directory.is_dir() else None

    def sector_kinds(self) -> list[str]:
        """列出板块分类目录（如 ``'指数成分股板块'``、``'申万一级行业板块'``）。"""
        root = self.sector_root
        if root is None:
            return []
        return sorted(entry.name for entry in root.iterdir() if entry.is_dir())

    def market_sector_names(self) -> list[str]:
        """列出市场分类板块名（升序）。

        这类板块（``'沪市基金'``、``'深市A股'``、``'创业板'``…）在 QMT 客户端
        由「市场 + 代码前缀」动态生成，datadir 没有文件，只能从本地已缓存的
        标的中筛选。
        """
        return sorted(MARKET_SECTORS)

    def sector_names(self, kind: str | None = None) -> list[str]:
        """列出板块标识（``'分类/板块名'``，升序）。

        跳过 ``sectorConfig.xml`` 与 0 字节的空板块（如未下载的港股板块）。

        Args:
            kind: 限定分类目录，None 表示全部分类

        Returns:
            形如 ``'指数成分股板块/沪深300'`` 的标识列表
        """
        root = self.sector_root
        if root is None:
            return []
        kinds = [kind] if kind is not None else self.sector_kinds()
        names: list[str] = []
        for kind_name in kinds:
            directory = root / kind_name
            if not directory.is_dir():
                continue
            for entry in directory.iterdir():
                if not entry.is_file() or entry.suffix.lower() == ".xml":
                    continue
                try:
                    if entry.stat().st_size == 0:
                        continue
                except OSError:
                    continue
                names.append(f"{kind_name}/{entry.name}")
        return sorted(names)

    def sector_symbols(self, sector: str, kind: str | None = None) -> list[str]:
        """读取板块成分股（``'600000.SH'`` 形式，保持文件内顺序并去重）。

        板块文件是逗号分隔的纯文本，文件名即板块名（中文名为主，少数指数
        用代码，如 ``000809``）。文件未命中时会回退到 :data:`MARKET_SECTORS`
        定义的市场分类（如 ``'沪市基金'``），此时返回的是**本地已缓存**
        的标的，而非全市场清单。

        Args:
            sector: 板块名；可写成 ``'申万一级行业板块/SW1医药生物'`` 指定
                分类，也可只写 ``'SW1医药生物'``（全部分类中唯一时）
            kind: 分类目录；``sector`` 已含 ``/`` 时忽略

        Returns:
            成分股代码列表

        Raises:
            FileNotFoundError: 既无此板块文件，也不是已知市场分类
            ValueError: 板块名在多个分类中重名，需加分类前缀
        """
        root = self.sector_root
        missing: FileNotFoundError | None = None
        if root is not None:
            try:
                path = self._resolve_sector_path(root, sector, kind)
            except FileNotFoundError as exc:
                missing = exc
            else:
                return self._read_sector_file(path)

        virtual = self._market_sector_symbols(sector)
        if virtual is not None:
            return virtual

        if missing is not None:
            raise missing
        raise FileNotFoundError(
            f"Sector {sector!r} not found: no Sector directory under {self._root} "
            f"and not a known market sector ({', '.join(sorted(MARKET_SECTORS))})"
        )

    @staticmethod
    def _read_sector_file(path: Path) -> list[str]:
        """解析逗号分隔的板块成分股文件。"""
        raw = path.read_text(encoding="utf-8", errors="replace")
        symbols: list[str] = []
        seen: set[str] = set()
        for item in raw.split(","):
            symbol = item.strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)
        return symbols

    def _market_sector_symbols(self, sector: str) -> list[str] | None:
        """按代码前缀从本地缓存中筛选市场分类板块；不是市场分类时返回 None。"""
        rules = MARKET_SECTORS.get(sector)
        if rules is None:
            return None

        by_market: dict[str, list[str]] = {}
        for symbol in self.available_symbols(period="1d"):
            code, _, market_name = symbol.partition(".")
            by_market.setdefault(market_name, []).append(code)

        selected: list[str] = []
        for market, prefixes in rules:
            for code in by_market.get(market, []):
                if code.startswith(prefixes):
                    selected.append(f"{code}.{market}")
        return selected

    def _resolve_sector_path(self, root: Path, sector: str, kind: str | None) -> Path:
        """定位板块成分股文件。"""
        if "/" in sector:
            kind_name, _, name = sector.partition("/")
            candidate = root / kind_name / name
            if not candidate.is_file():
                raise FileNotFoundError(f"Sector {sector!r} not found under {root}")
            return candidate

        kinds = [kind] if kind is not None else self.sector_kinds()
        matches = [
            root / kind_name / sector
            for kind_name in kinds
            if (root / kind_name / sector).is_file()
        ]
        if not matches:
            raise FileNotFoundError(
                f"Sector {sector!r} not found; kinds: {', '.join(self.sector_kinds())}"
            )
        if len(matches) > 1:
            preferred = [path for path in matches if path.parent.name == INDEX_SECTOR_KIND]
            if len(preferred) == 1:
                return preferred[0]
            ambiguous = ", ".join(f"{path.parent.name}/{path.name}" for path in matches)
            raise ValueError(f"Sector {sector!r} is ambiguous, qualify it with a kind: {ambiguous}")
        return matches[0]

    # ------------------------------------------------------------------
    # 解析主入口
    # ------------------------------------------------------------------

    def read_bars(
        self,
        symbol: str,
        period: str = "1d",
        start: str | date | None = None,
        end: str | date | None = None,
        dividend_type: DividendType = "none",
        ex_rights_threshold: float = DEFAULT_EX_RIGHTS_THRESHOLD,
    ) -> pl.DataFrame:
        """读取并解析单个标的的 K 线。

        Args:
            symbol: 证券代码，如 '600000.SH'
            period: '1d' / '1m' / '5m' / '15m' / '30m' / '1h' / '1w' / '1mon'
            start: 起始日期（inclusive），'YYYY-MM-DD' 或 'YYYYMMDD'
            end: 截止日期（inclusive），'YYYY-MM-DD' 或 'YYYYMMDD'
            dividend_type: 复权类型，'front'=前复权, 'back'=后复权, 'none'=不复权
            ex_rights_threshold: 判定除权事件的最小价差（元），默认 0.01

        Returns:
            ``[timestamp, open, high, low, close, volume, amount, suspend_flag]``；
            区间内无数据时返回空 DataFrame（含 schema）

        Raises:
            FileNotFoundError: 本地无该标的/该周期文件
            ValueError: 周期或复权类型非法
            QmtFormatError: 文件损坏（记录长度不足）
        """
        base_period, resample = self._resolve_period(period)
        market, code = split_symbol(symbol)

        raw = self._read_kline(self.kline_path(market, code, base_period))
        frame = _apply_adjustment(raw, dividend_type, ex_rights_threshold)

        start_date = _parse_date(start)
        end_date = _parse_date(end)
        if start_date is not None or end_date is not None:
            frame = self._filter_range(frame, start_date, end_date)

        if resample is not None:
            frame = resample(frame)

        return frame

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _resolve_period(
        self, period: str
    ) -> tuple[str, Callable[[pl.DataFrame], pl.DataFrame] | None]:
        """周期 → (原生周期, 聚合函数) 映射。"""
        key = period.lower()
        if key in self.PERIOD_DIRS:
            return key, None
        if key in self.INTRADAY_MINUTES:
            minutes = self.INTRADAY_MINUTES[key]
            return "1m", lambda df: _resample_intraday(df, minutes)
        if key in self.DAILY_DERIVED:
            every = "1w" if key == "1w" else "1mon"
            return "1d", lambda df: _resample_daily(df, every)
        raise ValueError(
            f"unsupported period {period!r}; supported: "
            f"{sorted([*self.PERIOD_DIRS, *self.INTRADAY_MINUTES, *self.DAILY_DERIVED])}"
        )

    def _read_kline(self, path: Path) -> pl.DataFrame:
        """读取单个定长记录文件并转成未复权 DataFrame。"""
        if not path.is_file():
            raise FileNotFoundError(f"QMT kline file not found: {path}")
        try:
            payload = path.read_bytes()
        except OSError as exc:  # 客户端独占写、权限不足等
            raise OSError(f"failed to read QMT kline file {path}: {exc}") from exc

        count = len(payload) // RECORD_SIZE
        if count == 0:
            raise QmtFormatError(f"empty QMT kline file: {path}")
        records = np.frombuffer(payload[: count * RECORD_SIZE], dtype=KLINE_DTYPE)

        # 时间戳 u32 秒：已是真实 UTC 时刻，可直接按 Asia/Shanghai 呈现
        timestamps = pl.Series("timestamp", records["timestamp"].astype("int64") * 1000).cast(
            pl.Datetime("ms")
        )

        frame = pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": records["open"].astype("float64") / PRICE_SCALE,
                "high": records["high"].astype("float64") / PRICE_SCALE,
                "low": records["low"].astype("float64") / PRICE_SCALE,
                "close": records["close"].astype("float64") / PRICE_SCALE,
                "volume": records["volume"].astype("float64"),
                "amount": records["amount"].astype("float64"),
                "_pre_close": records["pre_close"].astype("float64") / PRICE_SCALE,
            }
        ).with_columns(pl.col("timestamp").dt.replace_time_zone("UTC").dt.convert_time_zone(TZ))

        if frame["timestamp"].null_count():
            raise QmtFormatError(f"invalid timestamp in QMT kline file: {path}")

        # 文件内记录本身升序唯一；重复记录保留最后一条以防客户端重写
        return (
            frame.unique(subset=["timestamp"], keep="last", maintain_order=True)
            .sort("timestamp")
            .with_columns(
                ((pl.col("volume") == 0) & (pl.col("amount") == 0))
                .cast(pl.Int8)
                .alias("suspend_flag")
            )
        )

    @staticmethod
    def _filter_range(
        frame: pl.DataFrame,
        start_date: date | None,
        end_date: date | None,
    ) -> pl.DataFrame:
        """按日期区间过滤（inclusive）。"""
        out = frame
        if start_date is not None:
            out = out.filter(pl.col("timestamp").dt.date() >= pl.lit(start_date))
        if end_date is not None:
            out = out.filter(pl.col("timestamp").dt.date() <= pl.lit(end_date))
        return out


# ----------------------------------------------------------------------
# 复权
# ----------------------------------------------------------------------


def adjustment_factors(
    close: np.ndarray,
    pre_close: np.ndarray,
    threshold: float = DEFAULT_EX_RIGHTS_THRESHOLD,
) -> np.ndarray:
    """由文件内除权参考价推导逐日除权因子。

    Args:
        close: 收盘价序列（升序）
        pre_close: 文件记录的除权参考价序列（与 close 等长）
        threshold: 判定事件的最小价差（元），默认 0.01 表示「价差超过 0.01 元」

    Returns:
        因子序列，非除权日为 1.0，除权日为 ``pre_close / prev_close``
    """
    factors = np.ones(len(close), dtype="float64")
    if len(close) < 2:
        return factors

    # 价格精度为 0.001 元：先换算成整数毫元再比较。float64 下
    # 10.01 - 10.0 = 0.010000000000000497 > 0.01，直接用浮点比较会把「恰好
    # 一个最小价位」的舍入噪声误判为除权事件
    min_units = int(round(threshold * 1000))
    prev_close = close[:-1]
    cur_pre_close = pre_close[1:]
    prev_units = np.rint(prev_close * 1000).astype("int64")
    cur_units = np.rint(cur_pre_close * 1000).astype("int64")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(prev_close > 0, cur_pre_close / prev_close, 1.0)

    is_event = (
        (np.abs(cur_units - prev_units) > min_units)
        & np.isfinite(ratio)
        & (ratio > 0)
        & (prev_close > 0)
    )
    factors[1:] = np.where(is_event, ratio, 1.0)
    return factors


def _apply_adjustment(
    frame: pl.DataFrame,
    dividend_type: DividendType,
    threshold: float,
) -> pl.DataFrame:
    """按除权因子生成前/后复权价格（只调整 OHLC，不动 volume/amount）。"""
    if dividend_type not in ("front", "back", "none"):
        raise ValueError(
            f"unsupported dividend_type {dividend_type!r} (expected 'front'/'back'/'none')"
        )

    factors = adjustment_factors(
        frame["close"].to_numpy(), frame["_pre_close"].to_numpy(), threshold
    )

    if dividend_type == "none":
        return frame.drop("_pre_close")

    if dividend_type == "front":
        # 前复权：末端（最新一根）价格不变，历史价格按未来全部因子累计缩放
        suffix = np.cumprod(factors[::-1])[::-1]
        scale = np.ones(len(factors))
        scale[:-1] = suffix[1:]
    else:
        # 后复权：首端价格不变，后续价格按已发生因子累计放大
        scale = 1.0 / np.cumprod(factors)

    scale_series = pl.Series("_scale", scale)
    price_columns = ("open", "high", "low", "close")
    return frame.with_columns(
        [(pl.col(column) * scale_series).alias(column) for column in price_columns]
    ).drop("_pre_close")


# ----------------------------------------------------------------------
# 周期派生
# ----------------------------------------------------------------------

# A 股连续竞价时段（分钟数，自 00:00 起算）；09:30 集合竞价 bar 归一到 09:31
_SESSION_STARTS: Final[tuple[int, ...]] = (9 * 60 + 31, 13 * 60 + 1)
_SESSION_LENGTH: Final[int] = 120


def _resample_intraday(frame: pl.DataFrame, minutes: int) -> pl.DataFrame:
    """把 1 分钟线聚合为 N 分钟线（按 bar 结束时间标注，与 xtdata 惯例一致）。

    A 股 1 分钟文件每日 241 根（09:30 集合竞价 + 09:31–11:30 + 13:01–15:00），
    聚合时把 09:30 归一到 09:31，因此 N 分钟线每日 120/N 根，首根覆盖
    ``[09:30, 09:35]``（N=5 时），与常见数据商口径一致。
    """
    if minutes <= 0 or _SESSION_LENGTH % minutes:
        raise ValueError(f"minutes must divide {_SESSION_LENGTH}, got {minutes}")
    if frame.is_empty():
        return frame

    seconds = frame["timestamp"].dt.epoch("ms").to_numpy() // 1000
    seconds_of_day = (seconds + 8 * 3600) % 86_400
    local_day_start = seconds - seconds_of_day
    minutes_of_day = seconds_of_day // 60

    in_morning = minutes_of_day <= 11 * 60 + 30
    session_start = np.where(in_morning, _SESSION_STARTS[0], _SESSION_STARTS[1])
    rel = np.clip(minutes_of_day - session_start, 0, _SESSION_LENGTH - 1)
    group = rel // minutes
    label_minutes_of_day = session_start + group * minutes + minutes - 1

    labels = (
        pl.Series("timestamp", (local_day_start + label_minutes_of_day * 60) * 1000)
        .cast(pl.Datetime("ms"))
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(TZ)
    )

    work = frame.drop("_pre_close", strict=False).with_columns(
        labels,
        pl.Series("_group", (group + np.where(in_morning, 0, 1000)).astype("int64")),
    )

    return (
        work.group_by(["_group"], maintain_order=True)
        .agg(
            pl.col("timestamp").last(),
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("amount").sum(),
            pl.col("suspend_flag").max(),
        )
        .drop("_group")
        .sort("timestamp")
    )


def _resample_daily(frame: pl.DataFrame, every: Literal["1w", "1mon"]) -> pl.DataFrame:
    """把日线聚合为周线/月线（按区间内最后一个交易日标注）。"""
    if frame.is_empty():
        return frame

    dates = frame["timestamp"].dt.date()
    if every == "1w":
        days = dates.dt.epoch("d").to_numpy()
        # 1970-01-05 为周一：以周一为周起点分组
        keys = (days - 4) // 7
    else:
        keys = dates.dt.year().to_numpy().astype("int64") * 100 + dates.dt.month().to_numpy()

    work = frame.drop("_pre_close", strict=False).with_columns(
        pl.Series("_group", keys.astype("int64"))
    )
    aggregated = (
        work.group_by(["_group"], maintain_order=True)
        .agg(
            pl.col("timestamp").last(),
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("amount").sum(),
            pl.col("suspend_flag").max(),
        )
        .drop("_group")
        .sort("timestamp")
    )
    return aggregated


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------


def _parse_date(value: str | date | None) -> date | None:
    """解析 'YYYY-MM-DD' / 'YYYYMMDD' / date。"""
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"invalid date {value!r} (expected 'YYYY-MM-DD' or 'YYYYMMDD')")


def _local_drive_roots() -> list[Path]:
    """列出本地磁盘根目录（用于自动发现，跳过不可用磁盘）。"""
    listdrives = getattr(os, "listdrives", None)
    if listdrives is None:  # pragma: no cover - 依赖平台版本
        return []
    roots: list[Path] = []
    for drive in listdrives():
        try:
            if os.path.exists(drive):
                roots.append(Path(f"{drive}\\"))
        except OSError:  # pragma: no cover - 断连的网络盘
            continue
    return roots


def _ts_to_date(timestamp: int) -> date:
    """Unix 秒 → Asia/Shanghai 日期（固定 +08:00）。"""
    return datetime.fromtimestamp(timestamp, _CST).date()


def _qmt_candidates_under(base: Path) -> list[Path]:
    """在给定目录下寻找 QMT 客户端的两种 datadir 布局。"""
    candidates: list[Path] = []
    try:
        entries = [entry for entry in base.iterdir() if entry.is_dir()]
    except OSError:  # pragma: no cover - 权限不足
        return candidates

    for entry in entries:
        if "qmt" not in entry.name.lower():
            continue
        candidates.append(entry / "datadir")
        candidates.append(entry / "userdata_mini" / "datadir")
    return candidates
