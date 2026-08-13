"""QMT HTTP 转发接口 A 股 ETF OHLCV Provider.

继承 BaseProvider，通过 QMT 内部 HTTP 转发服务（QMT_HTTP_API封装.py）的
/api/data/market_data_ex 接口获取数据，输出与 XtDataProvider 一致的
标准长格式 Polars DataFrame。

数据流：
    POST /api/data/market_data_ex
        → {"data": {symbol: {field: {index: value}}}}（DataFrame.to_dict 序列化）
        → pd.DataFrame.from_dict 重建
        → 'time' 列毫秒时间戳 → 日期解析（Asia/Shanghai）
        → symbol 列 + 类型转换
        → pl.DataFrame[timestamp, symbol, open, high, low, close, volume, ...]

前置条件：QMT 客户端已启动，且已在客户端内加载 QMT_HTTP_API封装.py
（默认 http://127.0.0.1:10086，认证 header X-Token）。

与 XtDataProvider 的差异：
    - 数据来自 HTTP 序列化结果而非本地 xtquant 库，无需安装 xtquant
    - ContextInfo.get_market_data_ex 返回 DataFrame 的 index 为毫秒时间戳
      （xtdata 为 'YYYYMMDD' 字符串），时间由 'time' 字段还原
    - 服务端未暴露历史数据下载接口，故不支持 download() / auto_download
    - 小时线周期为 '1h'（xtdata 为 '60m'）
    - 停牌填充 fill_data 由服务端固定为 True，客户端不可配置
"""

from __future__ import annotations

from typing import Any, ClassVar

import httpx
import pandas as pd
import polars as pl
import structlog

from ml4t.data.core.exceptions import DataNotAvailableError, NetworkError, SymbolNotFoundError
from ml4t.data.providers.base import BaseProvider

logger = structlog.get_logger()


class QmtProvider(BaseProvider):
    """QMT HTTP 转发接口 A 股 ETF 数据 Provider。

    通过 QMT 内部 HTTP 服务（QMT_HTTP_API封装.py）的 /api/data/market_data_ex
    接口获取 A 股 ETF 日线数据，反序列化后转换为 ml4t-data 标准 OHLCV 格式，
    输出与 XtDataProvider 保持一致。

    额外保留列：amount（成交额）。服务端字段不含 xtdata 特有的
    suspendFlag / preClose，故不输出。

    Usage:
        >>> from ml4t.data import DataManager
        >>> dm = DataManager()
        >>> dm._provider_manager.register_provider("qmt", QmtProvider)
        >>> df = dm.fetch("510300.SH", "2020-01-01", "2026-07-01", provider="qmt")
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
        "hourly": "1h",
        "1hour": "1h",
        "60m": "1h",
    }

    # 请求字段：OHLCV + amount（对齐 xtdata 保留列）+ time（毫秒时间戳，还原日期）
    FIELDS: ClassVar[list[str]] = [
        "open", "high", "low", "close", "volume", "amount", "time",
    ]

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:10086",
        token: str = "123456789",
        dividend_type: str = "front",
        rate_limit: tuple[int, float] | None = None,
        session_config: dict[str, Any] | None = None,
    ) -> None:
        """初始化 QmtProvider。

        Args:
            base_url: QMT HTTP 服务地址（QMT_HTTP_API封装.py 监听地址）
            token: 服务端认证 token（与 QMT_HTTP_API封装.py 中 TOKEN 一致）
            dividend_type: 复权类型，'front'=前复权, 'back'=后复权, 'none'=不复权
            rate_limit: 限流配置 (calls, period_seconds)
            session_config: 额外 httpx.Client 配置（如 timeout）
        """
        self._base_url = base_url.rstrip("/")
        self._dividend_type = dividend_type

        config = dict(session_config or {})
        config["headers"] = {"X-Token": token, **(config.get("headers") or {})}
        super().__init__(rate_limit=rate_limit, session_config=config)

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "qmt"

    def _request_market_data_ex(
        self, symbols: list[str], start: str, end: str, period: str
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """调用 QMT HTTP 服务 /api/data/market_data_ex 获取序列化行情。

        Args:
            symbols: 股票代码列表，如 ['510300.SH']
            start: 起始日期 'YYYYMMDD' (inclusive)
            end: 截止日期 'YYYYMMDD' (inclusive)
            period: QMT period 字符串（'1d' / '1w' / '1mon' / '1h' 等）

        Returns:
            {"data": {symbol: {field: {index: value}}}} 序列化结果

        Raises:
            NetworkError: HTTP 连接失败（QMT 客户端未启动/服务未加载）
            DataNotAvailableError: HTTP 错误状态或响应解析失败
        """
        url = f"{self._base_url}/api/data/market_data_ex"
        payload = {
            "fields": ",".join(self.FIELDS),
            "stock_code": ",".join(symbols),
            "period": period,
            "start_time": start,
            "end_time": end,
            "count": -1,
            "dividend_type": self._dividend_type,
        }
        symbol_hint = symbols[0] if len(symbols) == 1 else ""

        try:
            resp = self.session.post(url, json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = str(e.response.json().get("error", ""))
            except Exception:
                pass
            logger.error(
                "QMT HTTP request failed",
                status_code=e.response.status_code,
                error=detail,
                symbols=symbols,
            )
            raise DataNotAvailableError(
                "qmt",
                symbol_hint,
                details={
                    "stage": "http",
                    "status_code": e.response.status_code,
                    "error": detail,
                    "start": start,
                    "end": end,
                },
            ) from e
        except httpx.RequestError as e:
            logger.error("QMT HTTP connection failed", error=str(e))
            raise NetworkError(
                "qmt",
                message=f"QMT HTTP 服务连接失败: {e}（请确认 QMT 客户端已启动且已加载 HTTP 服务）",
                details={"start": start, "end": end},
            ) from e

        try:
            body = resp.json()
        except ValueError as e:
            logger.error("QMT HTTP response is not valid JSON", error=str(e))
            raise DataNotAvailableError(
                "qmt",
                symbol_hint,
                details={"stage": "parse", "error": str(e)},
            ) from e

        data = body.get("data")
        if not isinstance(data, dict):
            raise DataNotAvailableError(
                "qmt",
                symbol_hint,
                details={"stage": "parse", "error": "response missing 'data' field"},
            )
        return data

    @staticmethod
    def _rebuild_dataframe(
        serialized: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        """将服务端 DataFrame.to_dict() 序列化结果重建为 pandas DataFrame。

        Args:
            serialized: {field: {index: value}} 序列化字典

        Returns:
            重建后的 pandas DataFrame（index 为原始行序）
        """
        if not serialized:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(serialized)

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        """获取并转换 QMT HTTP OHLCV 数据。

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
        start_qmt = start.replace("-", "")
        end_qmt = end.replace("-", "")
        period = self.FREQUENCY_MAP.get(frequency.lower(), "1d")

        logger.info(
            "Fetching data from QMT HTTP",
            symbol=symbol,
            start=start_qmt,
            end=end_qmt,
            period=period,
            dividend_type=self._dividend_type,
        )

        raw_dict = self._request_market_data_ex([symbol], start_qmt, end_qmt, period)

        if symbol not in raw_dict or not raw_dict[symbol]:
            raise SymbolNotFoundError(
                "qmt",
                symbol,
                details={"start": start, "end": end, "frequency": frequency},
            )

        raw_df = self._rebuild_dataframe(raw_dict[symbol])

        # 清理 OHLC 全为空的行（停牌填充行等）
        ohlc_cols = ["open", "high", "low", "close"]
        available_ohlc = [c for c in ohlc_cols if c in raw_df.columns]
        if available_ohlc:
            raw_df = raw_df.dropna(subset=available_ohlc, how="all")

        if raw_df.empty:
            raise SymbolNotFoundError(
                "qmt",
                symbol,
                details={"reason": "all OHLC rows are empty after cleanup"},
            )

        df = self._transform_to_polars(raw_df, symbol)

        logger.info(
            "Successfully fetched QMT HTTP data",
            symbol=symbol,
            rows=len(df),
        )
        return df

    def _transform_to_polars(self, raw_df: pd.DataFrame, symbol: str) -> pl.DataFrame:
        """将 QMT HTTP 反序列化的 pandas DataFrame 转换为标准 Polars DataFrame。

        时间戳解析优先级：
        1. 'time' 列（毫秒时间戳）——服务端正常路径
        2. index 列（毫秒时间戳 / 'YYYYMMDD' / 标准时间字符串）——防御性兜底

        Args:
            raw_df: 重建的 pandas DataFrame
            symbol: 股票代码

        Returns:
            标准 schema 的 Polars DataFrame
        """
        df_pd = raw_df.reset_index()

        if "time" in df_pd.columns:
            # epoch 毫秒是 UTC 时刻：先标 UTC 再转 Asia/Shanghai
            ts = pd.to_datetime(
                df_pd["time"], unit="ms", errors="coerce", utc=True
            ).dt.tz_convert("Asia/Shanghai")
            df_pd["timestamp"] = ts
            df_pd = df_pd.drop(columns=["time"])
        else:
            idx = df_pd["index"].astype(str)
            if idx.str.fullmatch(r"\d{10,}").all():
                ts = pd.to_datetime(
                    idx.astype("int64"), unit="ms", errors="coerce", utc=True
                ).dt.tz_convert("Asia/Shanghai")
            elif idx.str.fullmatch(r"\d{8}").all():
                ts = pd.to_datetime(idx, format="%Y%m%d", errors="coerce").dt.tz_localize(
                    "Asia/Shanghai", nonexistent="NaT", ambiguous="NaT"
                )
            else:
                ts = pd.to_datetime(idx, errors="coerce")
                if ts.dt.tz is None:
                    ts = ts.dt.tz_localize(
                        "Asia/Shanghai", nonexistent="NaT", ambiguous="NaT"
                    )
                else:
                    ts = ts.dt.tz_convert("Asia/Shanghai")
            df_pd["timestamp"] = ts
        df_pd = df_pd.drop(columns=["index"])

        df = pl.from_pandas(df_pd)

        # 过滤时间戳解析失败的行（NaT）
        df = df.filter(pl.col("timestamp").is_not_null())

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

        服务端原生支持多标的，一次 HTTP 请求获取所有标的数据，
        与 XtDataProvider 的批量行为对齐。

        Args:
            symbols: 股票代码列表
            start: 起始日期 'YYYY-MM-DD'
            end: 截止日期 'YYYY-MM-DD'
            frequency: 数据频率

        Returns:
            长格式 Polars DataFrame（包含所有 symbol 的数据）
        """
        start_qmt = start.replace("-", "")
        end_qmt = end.replace("-", "")
        period = self.FREQUENCY_MAP.get(frequency.lower(), "1d")

        logger.info(
            "Batch fetching from QMT HTTP",
            total_symbols=len(symbols),
            start=start_qmt,
            end=end_qmt,
            period=period,
        )

        raw_dict = self._request_market_data_ex(symbols, start_qmt, end_qmt, period)

        all_frames: list[pl.DataFrame] = []
        failed_symbols: list[str] = []

        ohlc_cols = ["open", "high", "low", "close"]

        for symbol in symbols:
            serialized = raw_dict.get(symbol)
            if not serialized:
                failed_symbols.append(symbol)
                continue

            raw_df = self._rebuild_dataframe(serialized)

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
