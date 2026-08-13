"""QmtProvider 测试（mock QMT HTTP 服务端）。

QMT HTTP 服务（QMT_HTTP_API封装.py）仅在 QMT 客户端内运行，这里用
httpx.MockTransport 模拟 /api/data/market_data_ex 的序列化响应格式：

    {"data": {code: {field: {index: value}}}}（DataFrame.to_dict 序列化）

覆盖点：请求参数构造、时间戳解析（time 列 / index 兜底）、复权参数、
批量取数、OHLC 空行清理、异常映射（SymbolNotFound / DataNotAvailable /
NetworkError）与端到端校验。
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx
import pandas as pd
import pytest

from ml4t.data.providers.qmt_provider import QmtProvider

# 2025-01-01 00:00:00 +08:00 → epoch 毫秒
TS_BASE = 1735660800000
DAY_MS = 86_400_000

FIELDS = ["time", "open", "high", "low", "close", "volume", "amount"]


# ---------------------------------------------------------------------------
# Fake QMT HTTP 服务端
# ---------------------------------------------------------------------------


class FakeQmtServer:
    """模拟 QMT_HTTP_API封装.py 的 /api/data/market_data_ex 服务。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.requests: list[httpx.Request] = []
        self.local: dict[str, dict[str, dict[str, Any]]] = {}
        self.http_error: tuple[int, str] | None = None
        self.connection_error = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.connection_error:
            raise httpx.ConnectError("connection refused", request=request)
        if self.http_error is not None:
            code, msg = self.http_error
            return httpx.Response(code, json={"error": msg, "status_code": code})
        payload = self.last_payload()
        symbols = [s.strip() for s in payload["stock_code"].split(",")]
        data = {
            sym: self.local[sym]
            for sym in symbols
            if sym in self.local and self.local[sym]
        }
        return httpx.Response(200, json={"data": data})

    def last_payload(self) -> dict[str, Any]:
        return json.loads(self.last_request().content) if self.requests else {}

    def last_request(self) -> httpx.Request:
        assert self.requests, "no request captured"
        return self.requests[-1]


def _make_serialized_bars(dates: list[str]) -> dict[str, dict[str, Any]]:
    """构造服务端 DataFrame.to_dict() 序列化格式（index 为 'YYYYMMDD' 字符串）。"""
    rows: list[dict[str, Any]] = []
    for i, d in enumerate(dates):
        ts_ms = TS_BASE + i * DAY_MS
        rows.append(
            {
                "time": ts_ms,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1000,
                "amount": 1_000_000.0,
                # 记录日期以便断言（不进入序列化）
                "_date": d,
            }
        )
    return {f: {rows[i]["_date"]: rows[i][f] for i in range(len(rows))} for f in FIELDS}


def _make_zero_bars(dates: list[str]) -> dict[str, dict[str, Any]]:
    """构造服务端对无效代码返回的全 0 填充序列化数据（真实 fill_data 行为）。"""
    rows: list[dict[str, Any]] = []
    for i, d in enumerate(dates):
        rows.append(
            {
                "time": TS_BASE + i * DAY_MS,
                "open": 0.0,
                "high": 0.0,
                "low": 0.0,
                "close": 0.0,
                "volume": 0,
                "amount": 0.0,
                "_date": d,
            }
        )
    return {f: {rows[i]["_date"]: rows[i][f] for i in range(len(rows))} for f in FIELDS}


def _make_constant_bars(dates: list[str], price: float = 7.98) -> dict[str, dict[str, Any]]:
    """构造本地缺历史时服务端的常量价格填充序列化数据（OHLC 恒定 + volume=0）。"""
    rows: list[dict[str, Any]] = []
    for i, d in enumerate(dates):
        rows.append(
            {
                "time": TS_BASE + i * DAY_MS,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0,
                "amount": 0.0,
                "_date": d,
            }
        )
    return {f: {rows[i]["_date"]: rows[i][f] for i in range(len(rows))} for f in FIELDS}


@pytest.fixture
def fake_server() -> FakeQmtServer:
    server = FakeQmtServer()
    server.local["510300.SH"] = _make_serialized_bars(
        ["20250101", "20250102", "20250103"]
    )
    server.local["510500.SH"] = _make_serialized_bars(["20250101", "20250102"])
    return server


@pytest.fixture
def provider(fake_server):
    """每个测试使用干净的 fake 服务端。"""
    p = QmtProvider(
        session_config={"transport": httpx.MockTransport(fake_server.handler)}
    )
    yield p
    p.close()


# ---------------------------------------------------------------------------
# 取数与请求构造
# ---------------------------------------------------------------------------


class TestFetch:
    def test_fetch_ohlcv_basic(self, provider):
        """取数：输出标准 schema、3 行、时间戳为 Asia/Shanghai。"""
        df = provider._fetch_and_transform_data(
            "510300.SH", "2025-01-01", "2025-01-03", "daily"
        )

        assert len(df) == 3
        assert list(df.columns[:7]) == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert df["symbol"].to_list() == ["510300.SH"] * 3
        assert df["timestamp"].dt.date().to_list() == [
            date(2025, 1, 1),
            date(2025, 1, 2),
            date(2025, 1, 3),
        ]
        assert df["amount"].to_list() == [1_000_000.0] * 3

    def test_request_payload(self, provider, fake_server):
        """请求构造：字段、代码、周期、日期范围、count、复权方式。"""
        provider._fetch_and_transform_data(
            "510300.SH", "2025-01-01", "2025-01-03", "daily"
        )

        req = fake_server.last_request()
        payload = fake_server.last_payload()
        assert str(req.url).endswith("/api/data/market_data_ex")
        assert payload["fields"] == "open,high,low,close,volume,amount,time"
        assert payload["stock_code"] == "510300.SH"
        assert payload["period"] == "1d"
        assert payload["start_time"] == "20250101"
        assert payload["end_time"] == "20250103"
        assert payload["count"] == -1
        assert payload["dividend_type"] == "front"
        assert req.headers["x-token"] == "123456789"

    def test_dividend_type_configurable(self):
        """复权方式可通过构造参数配置。"""
        server = FakeQmtServer()
        server.local["510300.SH"] = _make_serialized_bars(["20250101"])
        p = QmtProvider(
            dividend_type="none",
            session_config={"transport": httpx.MockTransport(server.handler)},
        )
        try:
            p._fetch_and_transform_data("510300.SH", "2025-01-01", "2025-01-02", "daily")
        finally:
            p.close()
        assert server.last_payload()["dividend_type"] == "none"

    def test_custom_base_url_no_double_slash(self):
        """base_url 尾斜杠会被去除。"""
        p = QmtProvider(
            base_url="http://127.0.0.1:10086/",
            session_config={"transport": httpx.MockTransport(FakeQmtServer().handler)},
        )
        try:
            assert p._base_url == "http://127.0.0.1:10086"
        finally:
            p.close()

    def test_custom_token_header(self):
        """自定义 token 写入 X-Token header。"""
        server = FakeQmtServer()
        server.local["510300.SH"] = _make_serialized_bars(["20250101"])
        p = QmtProvider(
            token="abc123",
            session_config={"transport": httpx.MockTransport(server.handler)},
        )
        try:
            p._fetch_and_transform_data("510300.SH", "2025-01-01", "2025-01-02", "daily")
        finally:
            p.close()
        assert server.last_request().headers["x-token"] == "abc123"


# ---------------------------------------------------------------------------
# 时间戳解析
# ---------------------------------------------------------------------------


class TestTimestampParsing:
    def test_time_column_used(self, provider):
        """主路径：'time' 列毫秒时间戳还原日期（Asia/Shanghai）。"""
        df = provider._fetch_and_transform_data(
            "510300.SH", "2025-01-01", "2025-01-03", "daily"
        )
        assert df["timestamp"].dt.date().to_list() == [
            date(2025, 1, 1),
            date(2025, 1, 2),
            date(2025, 1, 3),
        ]
        # time 列不应出现在输出
        assert "time" not in df.columns

    def test_index_fallback_ms_timestamp(self, provider):
        """兜底：无 time 列，index 为毫秒时间戳字符串。"""
        bars = _make_serialized_bars(["20250101", "20250102"])
        # index 改写为毫秒时间戳（旧格式兜底场景）
        ms_bars = {
            f: {str(TS_BASE + i * DAY_MS): v for i, v in enumerate(vals.values())}
            for f, vals in bars.items()
        }
        ms_bars.pop("time")  # 模拟服务端未返回 time 字段
        provider.session._transport.handler = _handler_with(ms_bars)

        df = provider._fetch_and_transform_data(
            "510300.SH", "2025-01-01", "2025-01-02", "daily"
        )

        assert df["timestamp"].dt.date().to_list() == [date(2025, 1, 1), date(2025, 1, 2)]

    def test_index_fallback_yyyymmdd(self, provider):
        """兜底：index 为 'YYYYMMDD' 字符串（xtdata 风格）。"""
        bars = {
            "open": {"20250101": 10.0, "20250102": 10.0},
            "high": {"20250101": 11.0, "20250102": 11.0},
            "low": {"20250101": 9.0, "20250102": 9.0},
            "close": {"20250101": 10.5, "20250102": 10.5},
            "volume": {"20250101": 1000, "20250102": 1000},
        }
        provider.session._transport.handler = _handler_with(bars)

        df = provider._fetch_and_transform_data(
            "510300.SH", "2025-01-01", "2025-01-02", "daily"
        )

        assert df["timestamp"].dt.date().to_list() == [date(2025, 1, 1), date(2025, 1, 2)]

    def test_nan_timestamp_rows_dropped(self, provider):
        """time 列含无效值 → 对应行被过滤。"""
        bars = _make_serialized_bars(["20250101", "20250102"])
        bars["time"]["invalid"] = float("nan")  # 额外一行无效时间戳
        bars["open"]["invalid"] = 10.0
        bars["high"]["invalid"] = 11.0
        bars["low"]["invalid"] = 9.0
        bars["close"]["invalid"] = 10.5
        bars["volume"]["invalid"] = 1000
        bars["amount"]["invalid"] = 1_000_000.0
        provider.session._transport.handler = _handler_with(bars, allow_nan=True)

        df = provider._fetch_and_transform_data(
            "510300.SH", "2025-01-01", "2025-01-02", "daily"
        )

        assert len(df) == 2


# ---------------------------------------------------------------------------
# OHLC 空行清理
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_ohlc_nan_rows_dropped(self, provider):
        """OHLC 全为空的行（停牌填充行）被清理。"""
        bars = _make_serialized_bars(["20250101", "20250102", "20250103"])
        for f in ("open", "high", "low", "close"):
            bars[f]["somekey"] = float("nan")
        bars["time"]["somekey"] = TS_BASE + 5 * DAY_MS
        bars["volume"]["somekey"] = 0
        bars["amount"]["somekey"] = 0.0
        provider.session._transport.handler = _handler_with(bars, allow_nan=True)

        df = provider._fetch_and_transform_data(
            "510300.SH", "2025-01-01", "2025-01-03", "daily"
        )

        assert len(df) == 3

    def test_all_zero_ohlc_rows_dropped(self, provider):
        """服务端对无效代码返回全 0 填充行（真实 fill_data 行为）→ 单标的抛 SymbolNotFoundError。"""
        provider.session._transport.handler = _handler_with(
            _make_zero_bars(["20250101", "20250102", "20250103"])
        )

        from ml4t.data.core.exceptions import SymbolNotFoundError

        with pytest.raises(SymbolNotFoundError):
            provider._fetch_and_transform_data(
                "BAD.SZ", "2025-01-01", "2025-01-03", "daily"
            )

    def test_fetch_batch_zero_filled_symbol_skipped(self, fake_server, provider):
        """批量：无效代码返回全 0 填充 → 不计入结果集。"""
        fake_server.local["BAD.SZ"] = _make_zero_bars(["20250101", "20250102", "20250103"])

        df = provider.fetch_batch_ohlcv(
            ["510300.SH", "BAD.SZ"], "2025-01-01", "2025-01-03", "daily"
        )

        assert set(df["symbol"].to_list()) == {"510300.SH"}
        assert len(df) == 3

    def test_constant_price_fill_rows_dropped(self, provider):
        """本地缺历史：常量价格 + volume=0 填充行被清理 → 单标的抛 SymbolNotFoundError。"""
        provider.session._transport.handler = _handler_with(
            _make_constant_bars(["20250101", "20250102", "20250103"])
        )

        from ml4t.data.core.exceptions import SymbolNotFoundError

        with pytest.raises(SymbolNotFoundError):
            provider._fetch_and_transform_data(
                "510500.SH", "2025-01-01", "2025-01-03", "daily"
            )

    def test_fetch_batch_constant_fill_symbol_skipped(self, fake_server, provider):
        """批量：本地缺历史的常量价格填充标的 → 不计入结果集。"""
        fake_server.local["510500.SH"] = _make_constant_bars(["20250101", "20250102"])

        df = provider.fetch_batch_ohlcv(
            ["510300.SH", "510500.SH"], "2025-01-01", "2025-01-03", "daily"
        )

        assert set(df["symbol"].to_list()) == {"510300.SH"}
        assert len(df) == 3

    def test_volume_zero_suspension_row_kept(self, provider):
        """OHLC 非恒定且 volume=0 的行（如真实停牌行）不被误删。"""
        bars = _make_serialized_bars(["20250101", "20250102", "20250103"])
        # 中间行模拟停牌：volume=0 但 OHLC 各异（非恒定）
        bars["open"]["20250102"] = 10.1
        bars["high"]["20250102"] = 10.2
        bars["low"]["20250102"] = 9.9
        bars["close"]["20250102"] = 10.0
        bars["volume"]["20250102"] = 0
        provider.session._transport.handler = _handler_with(bars)

        df = provider._fetch_and_transform_data(
            "510300.SH", "2025-01-01", "2025-01-03", "daily"
        )

        assert len(df) == 3


# ---------------------------------------------------------------------------
# 异常映射
# ---------------------------------------------------------------------------


class TestErrors:
    def test_symbol_not_found(self, fake_server, provider):
        """服务端无该标的数据 → SymbolNotFoundError。"""
        fake_server.local.pop("510300.SH")

        from ml4t.data.core.exceptions import SymbolNotFoundError

        with pytest.raises(SymbolNotFoundError):
            provider._fetch_and_transform_data(
                "510300.SH", "2025-01-01", "2025-01-03", "daily"
            )

    def test_http_error_raises_data_not_available(self, fake_server, provider):
        """服务端 HTTP 500 → DataNotAvailableError。"""
        fake_server.http_error = (500, "获取扩展行情失败")

        from ml4t.data.core.exceptions import DataNotAvailableError

        with pytest.raises(DataNotAvailableError):
            provider._fetch_and_transform_data(
                "510300.SH", "2025-01-01", "2025-01-03", "daily"
            )

    def test_connection_error_raises_network_error(self, fake_server, provider):
        """连接失败（QMT 未启动）→ NetworkError。"""
        fake_server.connection_error = True

        from ml4t.data.core.exceptions import NetworkError

        with pytest.raises(NetworkError):
            provider._fetch_and_transform_data(
                "510300.SH", "2025-01-01", "2025-01-03", "daily"
            )

    def test_response_missing_data_field(self, provider):
        """响应缺少 data 字段 → DataNotAvailableError。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": 1})

        provider.session._transport.handler = handler

        from ml4t.data.core.exceptions import DataNotAvailableError

        with pytest.raises(DataNotAvailableError):
            provider._fetch_and_transform_data(
                "510300.SH", "2025-01-01", "2025-01-03", "daily"
            )


# ---------------------------------------------------------------------------
# 批量取数与端到端
# ---------------------------------------------------------------------------


class TestBatchAndEndToEnd:
    def test_fetch_batch_ohlcv(self, provider, fake_server):
        """批量取数：一次请求返回所有标的数据，缺失标的跳过。"""
        df = provider.fetch_batch_ohlcv(
            ["510300.SH", "510500.SH", "BAD.SZ"],
            "2025-01-01",
            "2025-01-03",
            "daily",
        )

        assert set(df["symbol"].to_list()) == {"510300.SH", "510500.SH"}
        assert len(df) == 5  # 3 + 2
        # 批量应只发一次请求
        assert len(fake_server.requests) == 1

    def test_fetch_batch_all_failed_returns_empty(self, fake_server, provider):
        """全部标的失败 → 返回空标准 DataFrame。"""
        fake_server.local.clear()

        df = provider.fetch_batch_ohlcv(
            ["510300.SH"], "2025-01-01", "2025-01-03", "daily"
        )

        assert df.is_empty()
        assert list(df.columns[:7]) == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def test_fetch_ohlcv_end_to_end(self, provider):
        """走 BaseProvider 模板方法：校验（OHLC 关系/排序/去重）通过。"""
        df = provider.fetch_ohlcv("510300.SH", "2025-01-01", "2025-01-03", "daily")

        assert len(df) == 3
        assert list(df.columns[:7]) == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert df["timestamp"].dt.date().to_list() == [
            date(2025, 1, 1),
            date(2025, 1, 2),
            date(2025, 1, 3),
        ]


def _handler_with(bars: dict[str, dict[str, Any]], allow_nan: bool = False):
    """构造仅返回单标的序列化数据的 handler。

    allow_nan=True 时模拟真实服务端 json.dumps 默认输出（NaN 字面量），
    httpx 的 Response(json=...) 走严格模式无法序列化 NaN。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        symbols = [s.strip() for s in payload["stock_code"].split(",")]
        data = {sym: bars for sym in symbols}
        if allow_nan:
            return httpx.Response(
                200,
                content=json.dumps({"data": data}, allow_nan=True).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
        return httpx.Response(200, json={"data": data})

    return handler
