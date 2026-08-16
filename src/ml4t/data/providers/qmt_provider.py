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

此外提供与 ContextInfo 同名的数据查询接口（get_stock_name / get_open_date /
get_last_volume / get_sector / get_industry / get_stock_list_in_sector /
get_weight_in_index / get_risk_free_rate / get_trading_dates / get_longhubang /
get_total_share），经由服务端 /api/data/* 对应端点转发。
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
        timeout: float | None = None,
    ) -> None:
        """初始化 QmtProvider。

        Args:
            base_url: QMT HTTP 服务地址（QMT_HTTP_API封装.py 监听地址）
            token: 服务端认证 token（与 QMT_HTTP_API封装.py 中 TOKEN 一致）
            dividend_type: 复权类型，'front'=前复权, 'back'=后复权, 'none'=不复权
            rate_limit: 限流配置 (calls, period_seconds)
            session_config: 额外 httpx.Client 配置（如 timeout）
            timeout: 请求超时秒数，覆盖默认 30s。大批量/长历史请求时
                服务端耗时可达数十秒（实测上千标的 20 年数据约 94s），
                建议按需调大，如 timeout=180。与 session_config["timeout"]
                等价，同时给出时以本参数为准
        """
        self._base_url = base_url.rstrip("/")
        self._dividend_type = dividend_type

        config = dict(session_config or {})
        if timeout is not None:
            config["timeout"] = timeout
        config["headers"] = {"X-Token": token, **(config.get("headers") or {})}
        super().__init__(rate_limit=rate_limit, session_config=config)

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "qmt"

    def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        symbol_hint: str = "",
    ) -> dict[str, Any]:
        """发送 POST JSON 请求并解析响应（数据查询接口统一入口）。

        Args:
            endpoint: API 路径（如 '/api/data/stock_name'）
            payload: JSON 请求体
            symbol_hint: 错误日志中的标的提示

        Returns:
            解析后的响应字典

        Raises:
            NetworkError: HTTP 连接失败（QMT 客户端未启动/服务未加载）
            DataNotAvailableError: HTTP 错误状态或响应解析失败
        """
        url = f"{self._base_url}{endpoint}"
        self._acquire_rate_limit()
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
                endpoint=endpoint,
            )
            raise DataNotAvailableError(
                "qmt",
                symbol_hint,
                details={
                    "stage": "http",
                    "status_code": e.response.status_code,
                    "error": detail,
                    "endpoint": endpoint,
                },
            ) from e
        except httpx.RequestError as e:
            logger.error(
                "QMT HTTP connection failed", error=str(e), endpoint=endpoint
            )
            raise NetworkError(
                "qmt",
                message=f"QMT HTTP 服务连接失败: {e}（请确认 QMT 客户端已启动且已加载 HTTP 服务）",
                details={"endpoint": endpoint},
            ) from e

        try:
            return resp.json()
        except ValueError as e:
            logger.error(
                "QMT HTTP response is not valid JSON",
                error=str(e),
                endpoint=endpoint,
            )
            raise DataNotAvailableError(
                "qmt",
                symbol_hint,
                details={"stage": "parse", "error": str(e), "endpoint": endpoint},
            ) from e

    @staticmethod
    def _get_field_or_raise(
        body: dict[str, Any], key: str, symbol_hint: str = ""
    ) -> Any:
        """从响应中提取字段，缺失则抛 DataNotAvailableError。

        Args:
            body: 响应字典
            key: 字段名
            symbol_hint: 错误日志中的标的提示

        Returns:
            字段值（可能为 None，表示服务端调用失败）

        Raises:
            DataNotAvailableError: 响应缺少指定字段
        """
        if key not in body:
            raise DataNotAvailableError(
                "qmt",
                symbol_hint,
                details={"stage": "parse", "error": f"response missing '{key}' field"},
            )
        return body[key]

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
        symbol_hint = symbols[0] if len(symbols) == 1 else ""
        body = self._post_json(
            "/api/data/market_data_ex",
            {
                "fields": ",".join(self.FIELDS),
                "stock_code": ",".join(symbols),
                "period": period,
                "start_time": start,
                "end_time": end,
                "count": -1,
                "dividend_type": self._dividend_type,
            },
            symbol_hint,
        )
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

    @staticmethod
    def _clean_invalid_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
        """清理无效行情行：OHLC 全 NaN / 全 0 / volume=0 的常量填充行。

        服务端 fill_data=True 时对本地库无数据的代码返回填充行而非报错：
        - 完全无效代码：OHLC 全 0
        - 有效代码但本地缺历史：OHLC 为最新价常量且 volume=0
        dropna 无法识别这些填充行，需显式过滤，避免静默返回假数据。

        Args:
            raw_df: 重建的 pandas DataFrame

        Returns:
            清理后的 pandas DataFrame
        """
        ohlc_cols = ["open", "high", "low", "close"]
        available_ohlc = [c for c in ohlc_cols if c in raw_df.columns]
        if not available_ohlc:
            return raw_df

        n_before = len(raw_df)

        # 1) OHLC 全 NaN 行（停牌/上市前填充行）
        df = raw_df.dropna(subset=available_ohlc, how="all")

        # 2) OHLC 全 0 行（无效代码填充）
        zero_mask = pd.Series(True, index=df.index)
        for col in available_ohlc:
            zero_mask &= df[col].fillna(0) == 0

        # 3) volume=0 且 OHLC 恒定（本地缺历史的常量价格填充）
        const_mask = pd.Series(False, index=df.index)
        if "volume" in df.columns:
            first = df[available_ohlc[0]].fillna(0)
            same = pd.Series(True, index=df.index)
            for col in available_ohlc[1:]:
                same &= df[col].fillna(0) == first
            const_mask = (df["volume"].fillna(0) == 0) & same

        df = df[~(zero_mask | const_mask)]

        dropped = n_before - len(df)
        if dropped:
            logger.warning(
                "Dropped filled rows (QMT local data may be incomplete)",
                dropped=dropped,
                remaining=len(df),
            )
        return df

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

        # 清理无效行情行：OHLC 全 NaN（停牌填充）或全 0（无效代码填充）
        raw_df = self._clean_invalid_rows(raw_df)

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

        for symbol in symbols:
            serialized = raw_dict.get(symbol)
            if not serialized:
                failed_symbols.append(symbol)
                continue

            raw_df = self._rebuild_dataframe(serialized)

            # 清理无效行情行：OHLC 全 NaN（停牌填充）或全 0（无效代码填充）
            raw_df = self._clean_invalid_rows(raw_df)
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

    # ------------------------------------------------------------------
    # ContextInfo 同名数据查询接口（经由服务端 /api/data/* 转发）
    # 方法名与 QMT ContextInfo 保持一致，返回类型与原方法对齐。
    # 服务端 safe_call 调用失败时关键字段为 None：返回 None 并记录 warning。
    # ------------------------------------------------------------------

    def get_stock_name(self, stockcode: str) -> str | None:
        """根据代码获取股票名称（ContextInfo.get_stock_name）。

        Args:
            stockcode: 股票代码，如 '600000.SH'

        Returns:
            股票名称；服务端调用失败返回 None
        """
        body = self._post_json(
            "/api/data/stock_name", {"stockcode": stockcode}, stockcode
        )
        name = self._get_field_or_raise(body, "name", stockcode)
        if name is None:
            logger.warning("QMT get_stock_name returned None", stockcode=stockcode)
        return name

    def get_open_date(self, stockcode: str) -> int | None:
        """根据代码获取上市时间（ContextInfo.get_open_date）。

        服务端 get_open_date 为 QMT 策略环境内置全局函数，HTTP 服务（运行于
        init）中不可用，故改由 /api/data/instrumentdetail 的 OpenDate 字段
        （IPO 日期，见官方文档 get_instrumentdetail）获取。

        Args:
            stockcode: 股票代码，如 '600000.SH'

        Returns:
            上市日期整数 'YYYYMMDD'（如 19991110）；查询失败返回 None
        """
        body = self._post_json(
            "/api/data/instrumentdetail", {"stockcode": stockcode}, stockcode
        )
        detail = body.get("detail")
        open_date = detail.get("OpenDate") if isinstance(detail, dict) else None
        if open_date is None:
            logger.warning("QMT get_open_date returned None", stockcode=stockcode)
        return open_date

    def get_last_volume(self, stockcode: str) -> float | None:
        """获取最新流通股本（ContextInfo.get_last_volume）。

        Args:
            stockcode: 股票代码，如 '600000.SH'

        Returns:
            最新流通股本（股）；服务端调用失败返回 None
        """
        body = self._post_json(
            "/api/data/last_volume", {"stockcode": stockcode}, stockcode
        )
        last_volume = self._get_field_or_raise(body, "last_volume", stockcode)
        if last_volume is None:
            logger.warning(
                "QMT get_last_volume returned None", stockcode=stockcode
            )
        return last_volume

    def get_total_share(self, stockcode: str) -> float | None:
        """获取总股本（ContextInfo.get_total_share）。

        Args:
            stockcode: 股票代码，如 '600000.SH'

        Returns:
            总股本（股）；服务端调用失败返回 None
        """
        body = self._post_json(
            "/api/data/total_share", {"stockcode": stockcode}, stockcode
        )
        total_share = self._get_field_or_raise(body, "total_share", stockcode)
        if total_share is None:
            logger.warning("QMT get_total_share returned None", stockcode=stockcode)
        return total_share

    def get_sector(self, sector: str, realtime: int = 0) -> list[str]:
        """获取指数成分股（ContextInfo.get_sector）。

        Args:
            sector: 指数代码，如 '000300.SH'（沪深300）
            realtime: 是否实时获取，0=历史快照，1=实时

        Returns:
            成分股代码列表
        """
        body = self._post_json(
            "/api/data/sector", {"sector": sector, "realtime": realtime}, sector
        )
        return self._get_field_or_raise(body, "stocks", sector)

    def get_industry(self, industry: str) -> list[str]:
        """获取行业成分股（ContextInfo.get_industry）。

        Args:
            industry: 行业分类名，形式为 '分类代码+行业名'，如 'CSRC1金融业'、
                'SW1银行'（分类代码 CSRC=证监会，SW=申万；行业名可先通过
                get_industry_name_of_stock 反查）

        Returns:
            成分股代码列表
        """
        body = self._post_json(
            "/api/data/industry", {"industry": industry}, industry
        )
        return self._get_field_or_raise(body, "stocks", industry)

    def get_stock_list_in_sector(self, sectorname: str, realtime: int = 0) -> list[str]:
        """获取板块成分股（ContextInfo.get_stock_list_in_sector）。

        Args:
            sectorname: 板块名（无空格，与客户端左侧板块列表一致），如
                '沪深300'、'中证500'、'上证50'、'我的自选'
            realtime: 毫秒级时间戳，0 表示取最新成分股（服务端当前版本
                忽略该参数，仅返回最新成分股）

        Returns:
            成分股代码列表
        """
        body = self._post_json(
            "/api/data/stock_list_in_sector",
            {"sectorname": sectorname, "realtime": realtime},
            sectorname,
        )
        return self._get_field_or_raise(body, "stocks", sectorname)

    def get_weight_in_index(self, indexcode: str, stockcode: str) -> float | None:
        """获取股票在指数中的权重（ContextInfo.get_weight_in_index）。

        Args:
            indexcode: 指数代码，如 '000300.SH'
            stockcode: 股票代码，如 '600000.SH'

        Returns:
            权重，单位 %，如 1.6134 表示 1.6134%；不在指数内或调用失败返回 None
        """
        body = self._post_json(
            "/api/data/weight_in_index",
            {"indexcode": indexcode, "stockcode": stockcode},
            stockcode,
        )
        weight = self._get_field_or_raise(body, "weight", stockcode)
        if weight is None:
            logger.warning(
                "QMT get_weight_in_index returned None",
                indexcode=indexcode,
                stockcode=stockcode,
            )
        return weight

    def get_risk_free_rate(self, index: int = -1) -> float | None:
        """获取无风险利率（ContextInfo.get_risk_free_rate）。

        官方文档：用十年期国债收益率 CGB10Y 作无风险利率。

        Args:
            index: K 线索引号（barpos），-1 表示最新

        Returns:
            无风险利率；服务端调用失败返回 None
        """
        body = self._post_json("/api/data/risk_free_rate", {"index": index})
        rate = self._get_field_or_raise(body, "risk_free_rate")
        if rate is None:
            logger.warning("QMT get_risk_free_rate returned None", index=index)
        return rate

    def get_trading_dates(
        self,
        stockcode: str,
        start_date: str,
        end_date: str,
        count: int = -1,
        period: str = "1d",
    ) -> list:
        """获取交易日列表（ContextInfo.get_trading_dates）。

        Args:
            stockcode: 证券代码，如 '510300.SH'
            start_date: 起始日期，'YYYY-MM-DD' 或 'YYYYMMDD'
            end_date: 截止日期，'YYYY-MM-DD' 或 'YYYYMMDD'
            count: 返回数量，-1 表示区间内全部
            period: 周期（'1d' / '1w' / '1mon' 等）

        Returns:
            交易日列表；period 为日线时返回 ['20170101', ...] 字符串，
            其他周期返回时间戳。注意：官方文档注明该接口在 init 函数中
            不可用，而 HTTP 服务运行于 init，服务端可能返回空列表
        """
        body = self._post_json(
            "/api/data/trading_dates",
            {
                "stockcode": stockcode,
                "start_date": start_date.replace("-", ""),
                "end_date": end_date.replace("-", ""),
                "count": count,
                "period": period,
            },
            stockcode,
        )
        return self._get_field_or_raise(body, "dates", stockcode)

    def get_longhubang(
        self, stock_list: list[str], startTime: str, endTime: str
    ) -> Any:
        """获取龙虎榜数据（ContextInfo.get_longhubang）。

        Args:
            stock_list: 股票代码列表
            startTime: 起始日期 'YYYYMMDD'
            endTime: 截止日期 'YYYYMMDD'

        Returns:
            服务端返回 DataFrame.to_dict() 序列化结果时重建为 pandas
            DataFrame；否则返回原始值

        Raises:
            DataNotAvailableError: 服务端返回 error 或 HTTP 错误
        """
        joined = ",".join(stock_list)
        body = self._post_json(
            "/api/data/longhubang",
            {"stock_list": joined, "startTime": startTime, "endTime": endTime},
            joined,
        )
        if "error" in body:
            raise DataNotAvailableError(
                "qmt",
                joined,
                details={"stage": "server", "error": body["error"]},
            )
        data = body.get("data")
        if isinstance(data, dict) and data:
            # DataFrame.to_dict() 序列化：{列名: {行索引: 值}}
            return self._rebuild_dataframe(data)
        return data
