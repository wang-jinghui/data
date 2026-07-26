"""包股票 baostock A 股数据 Provider。

继承 BaseProvider，适配 baostock 的迭代式结果返回格式，
转换后输出标准长格式 Polars DataFrame。

数据流：
    bs.query_history_k_data_plus(code, fields, ...) → ResultSet
        → next() + get_row_data() → pd.DataFrame
        → 日期解析 + symbol 列 + 类型转换
        → pl.DataFrame[timestamp, symbol, open, high, low, close, volume, ...]

前置条件：无需额外客户端，pip install baostock 即可。
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
import polars as pl
import structlog

from ml4t.data.core.exceptions import DataNotAvailableError, SymbolNotFoundError
from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


class BaoStockProvider(BaseProvider):
    """包股票 baostock A 股数据 Provider。

    通过 baostock.query_history_k_data_plus() 获取 A 股历史 K 线数据，
    转换为 ml4t-data 标准 OHLCV 格式。

    额外保留列：amount（成交额）、turn（换手率）、pctChg（涨跌幅）、
    preclose（前收盘价）、tradestatus（交易状态）、isST（ST 标记）。

    Usage:
        >>> from ml4t.data import DataManager
        >>> dm = DataManager()
        >>> dm._provider_manager.register_provider("baostock", BaoStockProvider)
        >>> df = dm.fetch("sh.600000", "2020-01-01", "2024-12-31", provider="baostock")
    """

    FREQUENCY_MAP: ClassVar[dict[str, str]] = {
        "daily": "d",
        "1day": "d",
        "1d": "d",
        "weekly": "w",
        "1week": "w",
        "1w": "w",
        "monthly": "m",
        "1month": "m",
        "1mon": "m",
        "5minute": "5",
        "5m": "5",
        "15minute": "15",
        "15m": "15",
        "30minute": "30",
        "30m": "30",
        "60minute": "60",
        "60m": "60",
        "hourly": "60",
        "1hour": "60",
    }

    # 日线额外字段（分钟线不支持）
    DAILY_FIELDS: ClassVar[list[str]] = [
        "date", "code", "open", "high", "low", "close",
        "preclose", "volume", "amount", "adjustflag",
        "turn", "tradestatus", "pctChg", "isST",
    ]

    # 分钟线字段
    MINUTE_FIELDS: ClassVar[list[str]] = [
        "date", "time", "code", "open", "high", "low", "close",
        "volume", "amount", "adjustflag",
    ]

    def __init__(
        self,
        adjustflag: str = "3",
        rate_limit: tuple[int, float] | None = None,
    ) -> None:
        """初始化 BaoStockProvider。

        Args:
            adjustflag: 复权类型，'1'=后复权, '2'=前复权, '3'=不复权
            rate_limit: 限流配置 (calls, period_seconds)

        Raises:
            ImportError: 如果 baostock 未安装
        """
        try:
            import baostock as bs  # noqa: F401
        except ImportError:
            raise ImportError(
                "BaoStockProvider requires baostock. "
                "Install with: pip install baostock"
            )

        super().__init__(rate_limit=rate_limit)
        self._adjustflag = adjustflag
        self._logged_in = False

    def _ensure_login(self) -> None:
        """确保 baostock 会话已登录。"""
        if self._logged_in:
            return
        import baostock as bs

        lg = bs.login()
        if lg.error_code != "0":
            raise DataNotAvailableError(
                "baostock",
                "",
                details={
                    "error": f"login failed: {lg.error_code} - {lg.error_msg}"
                },
            )
        self._logged_in = True
        logger.debug("baostock login successful")

    def close(self) -> None:
        """关闭 baostock 会话。"""
        if self._logged_in:
            try:
                import baostock as bs
                bs.logout()
            except Exception:
                pass
            self._logged_in = False
        super().close()

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "baostock"

    @staticmethod
    def _is_minute_frequency(bs_frequency: str) -> bool:
        """判断是否为分钟线频率。"""
        return bs_frequency in ("5", "15", "30", "60")

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """获取并转换 baostock OHLCV 数据。

        Args:
            symbol: 股票代码，如 'sh.600000', 'sz.000001'
            start: 起始日期 'YYYY-MM-DD' (inclusive)
            end: 截止日期 'YYYY-MM-DD' (inclusive)
            frequency: 数据频率 (daily, minute, etc.)

        Returns:
            标准 schema 的 Polars DataFrame

        Raises:
            SymbolNotFoundError: 如果指定标的无数据返回
        """
        import baostock as bs

        self._ensure_login()

        bs_freq = self.FREQUENCY_MAP.get(frequency.lower(), "d")
        is_minute = self._is_minute_frequency(bs_freq)
        fields_str = ",".join(self.MINUTE_FIELDS if is_minute else self.DAILY_FIELDS)

        logger.info(
            "Fetching data from baostock",
            symbol=symbol,
            start=start,
            end=end,
            frequency=bs_freq,
            adjustflag=self._adjustflag,
        )

        try:
            rs = bs.query_history_k_data_plus(
                symbol,
                fields_str,
                start_date=start,
                end_date=end,
                frequency=bs_freq,
                adjustflag=self._adjustflag,
            )
        except Exception as e:
            logger.error("baostock query failed", symbol=symbol, error=str(e))
            raise DataNotAvailableError(
                "baostock",
                symbol,
                details={"start": start, "end": end, "error": str(e)},
            ) from e

        if rs.error_code != "0":
            raise DataNotAvailableError(
                "baostock",
                symbol,
                details={
                    "start": start,
                    "end": end,
                    "error_code": rs.error_code,
                    "error_msg": rs.error_msg,
                },
            )

        # 迭代获取结果
        data_list: list[list[Any]] = []
        while rs.error_code == "0" and rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            raise SymbolNotFoundError(
                "baostock",
                symbol,
                details={"start": start, "end": end, "frequency": frequency},
            )

        raw_df = pd.DataFrame(data_list, columns=rs.fields)

        # 清理 OHLC 全为空的行
        ohlc_cols = ["open", "high", "low", "close"]
        available_ohlc = [c for c in ohlc_cols if c in raw_df.columns]
        if available_ohlc:
            raw_df = raw_df.dropna(subset=available_ohlc, how="all")

        if raw_df.empty:
            raise SymbolNotFoundError(
                "baostock",
                symbol,
                details={"reason": "all OHLC rows are empty after cleanup"},
            )

        df = self._transform_to_polars(raw_df, symbol)

        logger.info(
            "Successfully fetched baostock data",
            symbol=symbol,
            rows=len(df),
        )
        return df

    def _transform_to_polars(self, raw_df: pd.DataFrame, symbol: str) -> pl.DataFrame:
        """将 baostock DataFrame 转换为标准 Polars DataFrame。

        Args:
            raw_df: baostock 返回的 DataFrame（所有列均为字符串类型）
            symbol: 股票代码（大写形式，用于 symbol 列）

        Returns:
            标准 schema 的 Polars DataFrame
        """
        df = pl.from_pandas(raw_df)

        # 判断是否有 time 列（分钟线）
        has_time = "time" in df.columns

        if has_time:
            # 分钟线: date + time → timestamp
            df = df.with_columns(
                pl.col("date")
                .str.concat(pl.lit(" "))
                .str.concat(pl.col("time"))
                .str.strptime(pl.Datetime("us", "Asia/Shanghai"), "%Y-%m-%d %H:%M:%S")
                .alias("timestamp")
            ).drop("date", "time")
        else:
            # 日/周/月线: date → timestamp (date 格式为 YYYY-MM-DD)
            df = df.with_columns(
                pl.col("date")
                .str.strptime(pl.Date, "%Y-%m-%d")
                .cast(pl.Datetime("us", "Asia/Shanghai"))
                .alias("timestamp")
            ).drop("date")

        # 拼 symbol 列
        df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))

        # OHLCV + 数值列类型转换（baostock 返回的全是 str）
        numeric_cols = [
            "open", "high", "low", "close", "volume", "amount",
            "turn", "pctChg", "preclose",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64))

        # tradestatus, adjustflag, isST 为整型
        int_cols = ["tradestatus", "adjustflag", "isST"]
        for col in int_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Int32))

        # code 列 — 原始返回中已有，与 symbol 列重复，移除
        if "code" in df.columns:
            df = df.drop("code")

        # 列排序：标准列在前，额外列在后
        std_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        extra_cols = [c for c in df.columns if c not in std_cols]
        df = df.select(std_cols + extra_cols)

        return df

    def fetch_batch_ohlcv(
        self,
        symbols: list[str],
        start: str,
        end: str,
        frequency: str = "daily",
    ) -> pl.DataFrame:
        """批量获取多个 symbol 的 OHLCV 数据。

        baostock 不支持原生批量查询，逐个获取后合并。

        Args:
            symbols: 股票代码列表，如 ['sh.600000', 'sz.000001']
            start: 起始日期 'YYYY-MM-DD'
            end: 截止日期 'YYYY-MM-DD'
            frequency: 数据频率

        Returns:
            长格式 Polars DataFrame（包含所有 symbol 的数据）
        """
        all_frames: list[pl.DataFrame] = []
        failed_symbols: list[str] = []

        for symbol in symbols:
            try:
                df = self._fetch_and_transform_data(symbol, start, end, frequency)
                all_frames.append(df)
            except (SymbolNotFoundError, DataNotAvailableError) as e:
                logger.warning(
                    "Failed to fetch symbol", symbol=symbol, error=str(e)
                )
                failed_symbols.append(symbol)
            except Exception as e:
                logger.warning(
                    "Unexpected error fetching symbol", symbol=symbol, error=str(e)
                )
                failed_symbols.append(symbol)

        if not all_frames:
            logger.error(
                "No data fetched for any symbol", failed_symbols=failed_symbols
            )
            return self._create_empty_dataframe()

        result = pl.concat(all_frames).sort(["symbol", "timestamp"])

        logger.info(
            "Batch fetch complete",
            successful_symbols=len(all_frames),
            failed_symbols=len(failed_symbols),
            total_rows=len(result),
        )

        if failed_symbols:
            logger.warning(
                "Some symbols failed",
                count=len(failed_symbols),
                symbols=failed_symbols[:10],
            )

        return result
