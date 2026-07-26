"""迅投 xtdata A股 ETF OHLCV Provider.

继承 BaseProvider，适配 xtdata 的 dict[str, DataFrame] 返回格式，
转换后输出标准长格式 Polars DataFrame。

数据流：
    xtdata.get_market_data_ex() → dict[symbol, DataFrame(index='YYYYMMDD')]
        → reset_index + 日期解析 + symbol 列 + 类型转换
        → pl.DataFrame[timestamp, symbol, open, high, low, close, volume, ...]

前置条件：迅投行情客户端已启动。
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
import polars as pl
import structlog

from ml4t.data.core.exceptions import DataNotAvailableError, SymbolNotFoundError
from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


class XtDataProvider(BaseProvider):
    """迅投 xtdata A 股 ETF 数据 Provider。

    通过 xtquant.xtdata.get_market_data_ex() 获取 A 股 ETF 日线数据，
    转换为 ml4t-data 标准 OHLCV 格式。

    额外保留列：amount（成交额）、suspendFlag（停牌标记）、preClose（前收盘）。

    Usage:
        >>> from ml4t.data import DataManager
        >>> dm = DataManager()
        >>> dm._provider_manager.register_provider("xtdata", XtDataProvider)
        >>> df = dm.fetch("510300.SH", "2020-01-01", "2026-07-01", provider="xtdata")
    """

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
        "hourly": "60m",
        "1hour": "60m",
        "60m": "60m",
    }

    def __init__(
        self,
        dividend_type: str = "front",
        fill_data: bool = True,
        rate_limit: tuple[int, float] | None = None,
    ) -> None:
        """初始化 XtDataProvider。

        Args:
            dividend_type: 复权类型，'front'=前复权, 'back'=后复权, 'none'=不复权
            fill_data: 是否填充非交易日数据
            rate_limit: 限流配置 (calls, period_seconds)

        Raises:
            ImportError: 如果 xtquant 未安装
        """
        try:
            from xtquant import xtdata  # noqa: F401
        except ImportError:
            raise ImportError(
                "XtDataProvider requires xtquant. "
                "请确保已安装 xtquant 且行情客户端已启动。"
            )

        super().__init__(rate_limit=rate_limit)
        self._dividend_type = dividend_type
        self._fill_data = fill_data

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "xtdata"

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """获取并转换 xtdata OHLCV 数据。

        Args:
            symbol: 股票代码，如 '510300.SH'
            start: 起始日期 'YYYY-MM-DD' (inclusive)
            end: 截止日期 'YYYY-MM-DD' (inclusive)
            frequency: 数据频率 (daily, minute, etc.)

        Returns:
            标准 schema 的 Polars DataFrame

        Raises:
            SymbolNotFoundError: 如果指定标的无数据返回
        """
        from xtquant import xtdata

        # 日期格式转换: "2025-01-01" → "20250101"
        start_xt = start.replace("-", "")
        end_xt = end.replace("-", "")
        period = self.FREQUENCY_MAP.get(frequency.lower(), "1d")

        logger.info(
            "Fetching data from xtdata",
            symbol=symbol,
            start=start_xt,
            end=end_xt,
            period=period,
            dividend_type=self._dividend_type,
        )

        try:
            raw_dict = xtdata.get_market_data_ex(
                stock_list=[symbol],
                period=period,
                start_time=start_xt,
                end_time=end_xt,
                count=-1,
                dividend_type=self._dividend_type,
                fill_data=self._fill_data,
            )
        except Exception as e:
            logger.error("xtdata call failed", symbol=symbol, error=str(e))
            raise DataNotAvailableError(
                "xtdata",
                symbol,
                details={"start": start, "end": end, "error": str(e)},
            ) from e

        if symbol not in raw_dict or raw_dict[symbol].empty:
            raise SymbolNotFoundError(
                "xtdata",
                symbol,
                details={"start": start, "end": end, "frequency": frequency},
            )

        raw_df = raw_dict[symbol]

        # 清理 OHLC 全为空的行（上市前填充行）
        ohlc_cols = ["open", "high", "low", "close"]
        available_ohlc = [c for c in ohlc_cols if c in raw_df.columns]
        if available_ohlc:
            raw_df = raw_df.dropna(subset=available_ohlc, how="all")

        if raw_df.empty:
            raise SymbolNotFoundError(
                "xtdata",
                symbol,
                details={"reason": "all OHLC rows are empty after cleanup"},
            )

        # 转换为 Polars
        df = self._transform_to_polars(raw_df, symbol)

        logger.info(
            "Successfully fetched xtdata data",
            symbol=symbol,
            rows=len(df),
        )
        return df

    def _transform_to_polars(self, raw_df: pd.DataFrame, symbol: str) -> pl.DataFrame:
        """将 xtdata pandas DataFrame 转换为标准 Polars DataFrame。

        Args:
            raw_df: xtdata 返回的 DataFrame，index 为 'YYYYMMDD' 字符串
            symbol: 股票代码

        Returns:
            标准 schema 的 Polars DataFrame
        """
        # reset_index() → 列名为 'index'（xtdata 的 index 无 name）
        df_pd = raw_df.reset_index()
        df = pl.from_pandas(df_pd)

        # index 列 → timestamp
        df = df.with_columns(
            pl.col("index").cast(pl.Utf8)
            .str.strptime(pl.Date, "%Y%m%d")
            .cast(pl.Datetime("us", "Asia/Shanghai"))
            .alias("timestamp")
        ).drop("index")

        # 拼 symbol 列
        df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))

        # OHLCV 类型转换
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        for col in ohlcv_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64))

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

        利用 xtdata 原生批量接口，一次请求获取所有标的数据，
        比逐个调用 fetch_ohlcv 更高效。

        Args:
            symbols: 股票代码列表
            start: 起始日期 'YYYY-MM-DD'
            end: 截止日期 'YYYY-MM-DD'
            frequency: 数据频率

        Returns:
            长格式 Polars DataFrame（包含所有 symbol 的数据）
        """
        from xtquant import xtdata

        start_xt = start.replace("-", "")
        end_xt = end.replace("-", "")
        period = self.FREQUENCY_MAP.get(frequency.lower(), "1d")

        logger.info(
            "Batch fetching from xtdata",
            total_symbols=len(symbols),
            start=start_xt,
            end=end_xt,
            period=period,
        )

        raw_dict = xtdata.get_market_data_ex(
            stock_list=symbols,
            period=period,
            start_time=start_xt,
            end_time=end_xt,
            count=-1,
            dividend_type=self._dividend_type,
            fill_data=self._fill_data,
        )

        all_frames: list[pl.DataFrame] = []
        failed_symbols: list[str] = []

        ohlc_cols = ["open", "high", "low", "close"]

        for symbol in symbols:
            if symbol not in raw_dict or raw_dict[symbol].empty:
                failed_symbols.append(symbol)
                continue

            raw_df = raw_dict[symbol]

            # 清理 OHLC 全空行
            available_ohlc = [c for c in ohlc_cols if c in raw_df.columns]
            if available_ohlc:
                raw_df = raw_df.dropna(subset=available_ohlc, how="all")
            if raw_df.empty:
                failed_symbols.append(symbol)
                continue

            try:
                df = self._transform_to_polars(raw_df, symbol)
                all_frames.append(df)
            except Exception as e:
                logger.warning("Failed to transform data", symbol=symbol, error=str(e))
                failed_symbols.append(symbol)

        if not all_frames:
            logger.error("No data fetched for any symbol", failed_symbols=failed_symbols)
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
