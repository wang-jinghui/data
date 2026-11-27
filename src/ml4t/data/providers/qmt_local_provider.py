"""QMT 本地缓存 datadir A 股 OHLCV Provider.

继承 BaseProvider，直接解析 QMT（含 MiniQMT）客户端缓存目录中的二进制 K 线，
**不要求 QMT 客户端在线、不要求安装 xtquant**，输出与 QmtProvider /
XtDataProvider 一致的标准长格式 Polars DataFrame。

数据流：
    <datadir>/<SH|SZ|BJ>/<86400|60>/<代码>.DAT
        → np.frombuffer（64 字节定长记录，小端）
        → Datetime[ms, Asia/Shanghai]，价格 /1000
        → 用文件内 preClose 推导除权因子 → 前/后复权
        → 周期派生（5m/15m/30m/1h 由 1m 聚合；1w/1mon 由日线聚合）
        → symbol 列 + 标准列序
        → pl.DataFrame[timestamp, symbol, open, high, low, close, volume, amount, suspend_flag]

前置条件：QMT 客户端此前已下载过目标标的历史数据（datadir 内存在对应 .DAT
文件）。本 provider **只读缓存、不触发下载**；若缺数据，请在 QMT 客户端里
补充下载后再读取。

与 QmtProvider 的差异：
    - 不依赖 QMT 客户端进程与 HTTP 转发服务，纯本地文件读取
    - 数据新鲜度取决于客户端最后一次下载/补全时间
    - 分钟线覆盖范围取决于客户端本地保留的分钟数据（通常仅最近数月）
    - 不提供 get_stock_name 等 ContextInfo 元数据查询，也不解析 ``Weight/``
      中的指数权重（LevelDB）；但板块/行业成分股可从 ``Sector/`` 纯文本缓存
      读取（见 :meth:`get_stock_list_in_sector` / :meth:`get_sector`）
    - 不解析 tick 数据（144 字节记录），仅支持 64 字节 K 线记录
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Final, cast

import polars as pl
import structlog

from ml4t.data.core.exceptions import DataNotAvailableError, SymbolNotFoundError
from ml4t.data.providers.base import BaseProvider
from ml4t.data.providers.protocols import ProviderCapabilities
from ml4t.data.providers.qmt_datadir import (
    DEFAULT_EX_RIGHTS_THRESHOLD,
    TZ,
    CoverageStats,
    DividendType,
    QmtDataDir,
    QmtFormatError,
    split_symbol,
)

logger = structlog.get_logger()

try:  # tqdm 为可选依赖：缺失时 fetch_batch_ohlcv 不显示进度条
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment,misc]

# 标准列（canonical schema），其余列作为额外列追加在后
_STD_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

# 常见宽基指数代码 → 本地板块名。QMT 的 Sector/指数成分股板块 按中文名存文件
INDEX_SECTOR_ALIASES: Final[dict[str, str]] = {
    "000001": "上证指数",
    "000009": "上证380",
    "000010": "上证180",
    "000016": "上证50",
    "000300": "沪深300",
    "000688": "科创50",
    "000852": "中证1000",
    "000905": "中证500",
    "000906": "中证800",
    "399001": "深证成指",
    "399006": "创业板指",
}


class QmtLocalProvider(BaseProvider):
    """QMT 本地缓存数据 Provider（离线读取 datadir）。

    直接解析 QMT 客户端缓存目录中的定长二进制 K 线记录，无需客户端在线。
    输出与 QmtProvider / XtDataProvider 一致的列结构：

    ``[timestamp, symbol, open, high, low, close, volume, amount, suspend_flag]``

    其中 ``amount``（成交额，元）与 ``suspend_flag``（停牌行标记）为额外列；
    ``volume`` 单位为「手」，与 xtdata 口径一致。

    Usage:
        >>> from ml4t.data import DataManager
        >>> from ml4t.data.providers.qmt_local_provider import QmtLocalProvider
        >>> dm = DataManager()
        >>> dm._provider_manager.register_provider("qmt_local", QmtLocalProvider)
        >>> df = dm.fetch("510300.SH", "2020-01-01", "2026-07-01", provider="qmt_local")

    也可直接指定缓存目录：

        >>> provider = QmtLocalProvider(data_dir=r"D:\\国金证券QMT交易端\\datadir")
        >>> provider.available_symbols(period="1d", market="SH")[:5]
    """

    # 频率名 → 周期（与 QmtProvider / XtDataProvider 对齐）
    FREQUENCY_MAP: ClassVar[dict[str, str]] = {
        "daily": "1d",
        "1day": "1d",
        "1d": "1d",
        "weekly": "1w",
        "1week": "1w",
        "1w": "1w",
        "monthly": "1mon",
        "1month": "1mon",
        "1mon": "1mon",
        "minute": "1m",
        "1minute": "1m",
        "1m": "1m",
        "5minute": "5m",
        "5m": "5m",
        "15minute": "15m",
        "15m": "15m",
        "30minute": "30m",
        "30m": "30m",
        "hourly": "1h",
        "1hour": "1h",
        "1h": "1h",
        "60m": "1h",
    }

    # 纯本地文件读取，不受远端配额约束，放宽限流避免拖慢批量读取
    DEFAULT_RATE_LIMIT: ClassVar[tuple[int, float]] = (100_000, 1.0)

    def __init__(
        self,
        data_dir: str | Path | None = None,
        dividend_type: str = "front",
        ex_rights_threshold: float = DEFAULT_EX_RIGHTS_THRESHOLD,
        rate_limit: tuple[int, float] | None = None,
        session_config: dict[str, Any] | None = None,
        search_roots: list[str | Path] | None = None,
    ) -> None:
        """初始化 QmtLocalProvider。

        Args:
            data_dir: QMT datadir 根目录（含 SH/SZ/BJ 子目录的那一层）。
                None 时按「环境变量 → 磁盘扫描」顺序自动发现，见
                :meth:`~ml4t.data.providers.qmt_datadir.QmtDataDir.discover`
            dividend_type: 复权类型，'front'=前复权, 'back'=后复权, 'none'=不复权
            ex_rights_threshold: 判定除权事件的最小价差（元），默认 0.01。
                不超过一个最小价位的差值视为舍入噪声，避免把 ±0.01 的
                噪声放大成假收益
            rate_limit: 已废弃，仅为兼容旧调用保留（本地文件读取不做限流）
            session_config: 已废弃，仅为兼容旧调用保留（本地读取无 HTTP 会话）
            search_roots: 自动发现时扫描的起始目录（默认为各本地磁盘根）

        Raises:
            FileNotFoundError: 未找到可用的 datadir
            ValueError: dividend_type 非法
        """
        if dividend_type not in ("front", "back", "none"):
            raise ValueError(
                f"unsupported dividend_type {dividend_type!r} (expected 'front'/'back'/'none')"
            )

        self._data_dir = QmtDataDir.discover(data_dir, search_roots)
        self._dividend_type = dividend_type
        self._ex_rights_threshold = ex_rights_threshold

        # 本地文件读取不经过 HTTP：跳过 BaseProvider 的限流器与 httpx 会话初始化，
        # 只保留熔断器（基类 fetch_ohlcv 模板会用到）。不初始化限流器时
        # _rate_limit_initialized 保持 False，_acquire_rate_limit() 直接放行。
        del rate_limit, session_config
        self.logger = structlog.get_logger(name=self.__class__.__name__)
        self._log_close_events = os.getenv("ML4T_LOG_PROVIDER_CLOSE", "0") == "1"
        self.init_circuit_breaker()

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "qmt_local"

    @property
    def data_dir(self) -> Path:
        """返回当前使用的 QMT datadir 根目录。"""
        return self._data_dir.root

    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities.

        Returns:
            支持分钟级数据、无需 API Key、无远端配额限制的能力描述
        """
        return ProviderCapabilities(
            supports_intraday=True,
            requires_api_key=False,
            rate_limit=self.DEFAULT_RATE_LIMIT,
        )

    # ------------------------------------------------------------------
    # 标的枚举
    # ------------------------------------------------------------------

    def available_symbols(
        self,
        period: str = "1d",
        market: str | None = None,
        traded_only: bool = False,
    ) -> list[str]:
        """列出本地已有缓存数据的标的（``'600000.SH'`` 形式，升序）。

        客户端会为大量未下载行情的品种写入零成交占位文件，``traded_only=True``
        可剔除这些占位标的。

        Args:
            period: 原生周期 '1d' / '1m'（聚合周期请查原生周期）
            market: 限定市场（SH/SZ/BJ），None 表示全部
            traded_only: 只保留首尾至少一条有成交的标的

        Returns:
            标的代码列表

        Raises:
            ValueError: period 不是原生周期
        """
        return self._data_dir.available_symbols(
            period=period, market=market, traded_only=traded_only
        )

    def coverage(
        self,
        period: str = "1d",
        market: str | None = None,
        stale_days: int = 30,
    ) -> list[CoverageStats]:
        """统计本地缓存覆盖情况，用于排查「还缺哪些数据」。

        Args:
            period: 原生周期 '1d' / '1m'
            market: 限定市场（SH/SZ/BJ），None 表示全部
            stale_days: 结束日落后最新日超过该天数的标的计为陈旧

        Returns:
            每个有数据的市场一条 :class:`CoverageStats`，按市场名升序
        """
        return self._data_dir.coverage(period=period, market=market, stale_days=stale_days)

    # ------------------------------------------------------------------
    # 板块 / 行业成分股（Sector/ 缓存）
    # ------------------------------------------------------------------

    def sector_kinds(self) -> list[str]:
        """列出本地板块分类目录（指数、申万 1/2/3 级、证监会、概念、地域、自选等）。"""
        return self._data_dir.sector_kinds()

    def sector_names(self, kind: str | None = None) -> list[str]:
        """列出板块标识（``'分类/板块名'`` 形式，升序）。

        Args:
            kind: 限定分类目录，None 表示全部分类
        """
        return self._data_dir.sector_names(kind=kind)

    def market_sector_names(self) -> list[str]:
        """列出市场分类板块（``'沪市基金'``、``'深市A股'``、``'创业板'``…）。

        这类板块在客户端由「市场 + 代码前缀」动态生成，datadir 没有对应文件，
        只能从本地已缓存的标的中筛选。
        """
        return self._data_dir.market_sector_names()

    def sector_symbols(self, sector: str, kind: str | None = None) -> list[str]:
        """读取板块成分股（``'600000.SH'`` 形式）。

        Args:
            sector: 板块名；可写成 ``'申万一级行业板块/SW1医药生物'`` 指定分类
            kind: 分类目录；``sector`` 已含 ``/`` 时忽略
        """
        return self._data_dir.sector_symbols(sector, kind=kind)

    def get_stock_list_in_sector(self, sectorname: str, realtime: int = 0) -> list[str]:
        """获取板块成分股（对齐 ``QmtProvider.get_stock_list_in_sector``）。

        先查 ``Sector/`` 板块文件；不是文件板块时回退到市场分类（``'沪市基金'``、
        ``'深市A股'``、``'创业板'`` 等），此时返回的是**本地已缓存**的标的。

        Args:
            sectorname: 板块名，与客户端左侧板块列表一致，如 ``'沪深300'``、
                ``'SW1医药生物'``、``'沪市基金'``、``'我的自选'``；重名时加
                分类前缀 ``'T行业/LED'``
            realtime: 本地缓存只有最新快照，忽略该参数

        Returns:
            成分股代码列表
        """
        del realtime  # 本地只有最新快照，无历史成分股
        return self._data_dir.sector_symbols(sectorname)

    def get_sector(self, sector: str, realtime: int = 0) -> list[str]:
        """获取指数成分股（对齐 ``QmtProvider.get_sector``）。

        与在线版不同，本地只有最新快照，没有历史成分股。

        Args:
            sector: 指数代码或名字，如 ``'000300.SH'``、``'SH000300'``、
                ``'沪深300'``；常见宽基指数走内置别名表，其余按文件名直查
            realtime: 本地缓存只有最新快照，忽略该参数

        Returns:
            成分股代码列表
        """
        del realtime  # 本地只有最新快照，无历史成分股
        return self._data_dir.sector_symbols(self._resolve_index_sector(sector))

    def get_industry(self, industry: str) -> list[str]:
        """获取行业成分股（对齐 ``QmtProvider.get_industry``）。

        Args:
            industry: 行业名，如 ``'SW1银行'``、``'CSRC1制造业'``

        Returns:
            成分股代码列表
        """
        return self._data_dir.sector_symbols(industry)

    def get_weight_in_index(self, indexcode: str, stockcode: str) -> float | None:
        """指数权重不可用：``Weight/`` 是 LevelDB，本 provider 不解析。"""
        raise NotImplementedError(
            "Index weights live in datadir/Weight (LevelDB); use QmtProvider for "
            "get_weight_in_index"
        )

    def _resolve_index_sector(self, sector: str) -> str:
        """把指数代码/名字映射为本地板块名。"""
        name = sector.strip()
        if "/" in name:
            return name
        code = name.split(".")[0].upper()
        if len(code) > 6 and code[:2] in ("SH", "SZ", "BJ"):
            code = code[2:]
        return INDEX_SECTOR_ALIASES.get(code, name)

    # ------------------------------------------------------------------
    # 取数
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> pl.DataFrame:
        """读取单个标的的 OHLCV 数据（覆盖基类模板）。

        校验与输出规范化与基类一致，但**不打 info 日志**、不做重试与熔断：
        本地文件读取既无远端配额，也不存在瞬时可恢复故障，重试只会把
        「文件被客户端独占写」这类错误延迟 4~10 秒才报出。

        Args:
            symbol: 证券代码，如 '600000.SH'（也接受 'SH600000'）
            start: 起始日期 'YYYY-MM-DD' (inclusive)
            end: 截止日期 'YYYY-MM-DD' (inclusive)
            frequency: 数据频率 (daily, minute, weekly, ...)

        Returns:
            标准 schema 的 Polars DataFrame
        """
        self._validate_inputs(symbol, start, end, frequency)
        data = self._fetch_and_transform_data(symbol, start, end, frequency)
        return self._validate_ohlcv(data, self.name)

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """读取本地缓存并转换为标准 schema。

        Args:
            symbol: 证券代码，如 '600000.SH'（也接受 'SH600000'）
            start: 起始日期 'YYYY-MM-DD' (inclusive)
            end: 截止日期 'YYYY-MM-DD' (inclusive)
            frequency: 数据频率 (daily, minute, weekly, ...)

        Returns:
            标准 schema 的 Polars DataFrame；区间内无数据时返回空 DataFrame

        Raises:
            ValueError: 频率或代码格式非法
            SymbolNotFoundError: 本地无该标的/该周期的缓存文件
            DataNotAvailableError: 缓存文件损坏或不可读
        """
        period = self.FREQUENCY_MAP.get(frequency.lower())
        if period is None:
            raise ValueError(
                f"unsupported frequency {frequency!r}; supported: {sorted(self.FREQUENCY_MAP)}"
            )

        market, code = split_symbol(symbol)
        canonical = f"{code}.{market}"

        try:
            bars = self._data_dir.read_bars(
                canonical,
                period=period,
                start=start,
                end=end,
                dividend_type=cast(DividendType, self._dividend_type),
                ex_rights_threshold=self._ex_rights_threshold,
            )
        except FileNotFoundError as exc:
            raise SymbolNotFoundError(
                self.name,
                canonical,
                details={
                    "period": period,
                    "data_dir": str(self._data_dir.root),
                    "hint": "本地无该标的/该周期的缓存文件，请先在 QMT 客户端下载数据",
                },
            ) from exc
        except QmtFormatError as exc:
            raise DataNotAvailableError(
                self.name,
                canonical,
                details={"stage": "parse", "period": period, "error": str(exc)},
            ) from exc
        except OSError as exc:  # 客户端独占写、权限不足等
            raise DataNotAvailableError(
                self.name,
                canonical,
                details={"stage": "io", "period": period, "error": str(exc)},
            ) from exc

        if bars.is_empty():
            return self._create_empty_dataframe()

        return self._transform_to_polars(bars, canonical)

    def _transform_to_polars(self, bars: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """解析层输出 → 标准 schema（补 symbol 列并调整列序）。

        Args:
            bars: :meth:`QmtDataDir.read_bars` 的输出
            symbol: 规范化后的证券代码，如 '600000.SH'

        Returns:
            标准 schema 的 Polars DataFrame
        """
        df = bars.with_columns(pl.lit(symbol).alias("symbol"))
        extra_cols = [c for c in df.columns if c not in _STD_COLUMNS]
        return df.select([*_STD_COLUMNS, *extra_cols])

    def _create_empty_dataframe(self) -> pl.DataFrame:
        """返回带完整 schema 的空 DataFrame（含额外列）。"""
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime(time_unit="ms", time_zone=TZ),
                "symbol": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "amount": pl.Float64,
                "suspend_flag": pl.Int8,
            }
        )

    def fetch_batch_ohlcv(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
        show_progress: bool = True,
    ) -> pl.DataFrame:
        """批量读取多个 symbol 的 OHLCV 数据。

        逐个读取本地文件：单个标的失败（无缓存、文件损坏）不会中断整体，
        失败的标的在结束时汇总告警。与 QmtProvider.fetch_batch_ohlcv
        返回结构一致（长格式、按 symbol + timestamp 升序）。

        Args:
            symbols: 证券代码列表
            start: 起始日期 'YYYY-MM-DD'
            end: 截止日期 'YYYY-MM-DD'
            frequency: 数据频率
            show_progress: 是否显示 tqdm 进度条（未安装 tqdm 时自动忽略）

        Returns:
            长格式 Polars DataFrame；全部失败时返回空 DataFrame
        """
        all_frames: list[pl.DataFrame] = []
        failed_symbols: list[str] = []

        progress: Any = symbols
        if show_progress and tqdm is not None:
            progress = tqdm(symbols, desc=f"QMT {frequency}", unit="sym")

        for symbol in progress:
            try:
                df = self._fetch_and_transform_data(symbol, start, end, frequency)
            except Exception as e:  # noqa: BLE001 - 单标的失败不中断批量
                logger.warning(
                    "Failed to read symbol from QMT datadir",
                    symbol=symbol,
                    error=str(e),
                )
                failed_symbols.append(symbol)
                continue

            if df.is_empty():
                failed_symbols.append(symbol)
                continue
            all_frames.append(df)

        if not all_frames:
            logger.error("No data read for any symbol", failed_symbols=failed_symbols)
            return self._create_empty_dataframe()

        result = pl.concat(all_frames).sort(["symbol", "timestamp"])

        if failed_symbols:
            logger.warning(
                "Some symbols failed",
                count=len(failed_symbols),
                total=len(symbols),
                symbols=failed_symbols[:10],
            )

        return result
