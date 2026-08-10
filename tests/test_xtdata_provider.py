"""XtDataProvider 测试（mock xtquant）。

xtquant 仅在装有迅投 QMT 客户端的机器上可用，这里用 FakeXtData 模拟
xtquant.xtdata 的本地数据库行为：

- get_market_data_ex: 从模拟本地库按 [start, end] 过滤返回（count 截断）
- download_history_data: 记录调用并将缺失日期补齐写入模拟本地库

覆盖点：本地已覆盖跳过下载、缺口触发增量下载、auto_download 开关、
会话内下载缓存、download() 公共方法、失败降级与异常映射。
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from types import ModuleType

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 注入 fake xtquant 模块（必须在 import provider 之前）
# ---------------------------------------------------------------------------


class FakeXtData:
    """模拟 xtquant.xtdata 的本地数据库。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.local: dict[str, pd.DataFrame] = {}
        self.download_calls: list[dict[str, str | bool]] = []
        self.fail_download_symbols: set[str] = set()
        self.fail_probe_symbols: set[str] = set()
        self.no_progress_symbols: set[str] = set()

    # -- xtquant 接口 ------------------------------------------------------

    def get_market_data_ex(
        self,
        stock_list: list[str],
        period: str = "1d",
        start_time: str = "",
        end_time: str = "",
        count: int = -1,
        dividend_type: str = "front",
        fill_data: bool = True,
    ) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}
        for symbol in stock_list:
            if symbol in self.fail_probe_symbols:
                raise RuntimeError("probe failed")
            df = self.local.get(symbol)
            if df is None or df.empty:
                result[symbol] = pd.DataFrame()
                continue
            mask = (df.index >= start_time) & (df.index <= end_time)
            sub = df[mask]
            if count != -1 and len(sub) > count:
                sub = sub.tail(count)
            result[symbol] = sub.copy()
        return result

    def download_history_data(
        self,
        stock_code: str,
        period: str = "1d",
        start_time: str = "",
        end_time: str = "",
        incrementally: bool = True,
    ) -> None:
        self.download_calls.append(
            {
                "code": stock_code,
                "period": period,
                "start_time": start_time,
                "end_time": end_time,
                "incrementally": incrementally,
            }
        )
        if stock_code in self.fail_download_symbols:
            raise RuntimeError("download failed")
        if stock_code in self.no_progress_symbols:
            return  # 模拟下载成功但本地无新增（未来日期/节假日）

        df = self.local.setdefault(stock_code, _make_bars([]))
        start_dt = datetime.strptime(start_time, "%Y%m%d")
        end_dt = datetime.strptime(end_time, "%Y%m%d")
        existing = set(df.index)
        new_dates = [
            (start_dt + timedelta(days=i)).strftime("%Y%m%d")
            for i in range((end_dt - start_dt).days + 1)
            if (start_dt + timedelta(days=i)).strftime("%Y%m%d") not in existing
        ]
        if new_dates:
            self.local[stock_code] = pd.concat([df, _make_bars(new_dates)]).sort_index()


def _make_bars(dates: list[str]) -> pd.DataFrame:
    """按日期列表生成 OHLC 关系合法的 K 线行。"""
    if not dates:
        return pd.DataFrame(
            {
                "open": pd.Series(dtype=float),
                "high": pd.Series(dtype=float),
                "low": pd.Series(dtype=float),
                "close": pd.Series(dtype=float),
                "volume": pd.Series(dtype=float),
                "amount": pd.Series(dtype=float),
                "suspendFlag": pd.Series(dtype=int),
                "preClose": pd.Series(dtype=float),
            }
        )
    df = pd.DataFrame(
        {
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1000.0,
            "amount": 1_000_000.0,
            "suspendFlag": 0,
            "preClose": 10.0,
        },
        index=pd.Index(dates),
    )
    df.index.name = None
    return df


_xtquant = ModuleType("xtquant")
fake_xtdata = FakeXtData()
_xtquant.xtdata = fake_xtdata
sys.modules["xtquant"] = _xtquant
sys.modules["xtquant.xtdata"] = fake_xtdata

from ml4t.data.providers.xtdata_provider import XtDataProvider  # noqa: E402


@pytest.fixture
def provider():
    """每个测试使用干净的 fake 本地库。"""
    fake_xtdata.reset()
    return XtDataProvider()


# ---------------------------------------------------------------------------
# 取数路径：探测与增量下载
# ---------------------------------------------------------------------------


class TestEnsureDownloaded:
    def test_fetch_no_download_when_local_covered(self, provider):
        """本地数据已覆盖请求区间 → 不触发下载。"""
        fake_xtdata.local["510300.SH"] = _make_bars(["20250101", "20250102", "20250103"])

        df = provider._fetch_and_transform_data(
            "510300.SH", "2025-01-01", "2025-01-03", "daily"
        )

        assert fake_xtdata.download_calls == []
        assert len(df) == 3

    def test_fetch_downloads_when_gap(self, provider):
        """本地数据截止早于请求 end → 增量下载缺口，参数正确。"""
        fake_xtdata.local["510300.SH"] = _make_bars(["20250101", "20250102", "20250103"])

        df = provider._fetch_and_transform_data(
            "510300.SH", "2025-01-01", "2025-01-06", "daily"
        )

        assert len(fake_xtdata.download_calls) == 1
        call = fake_xtdata.download_calls[0]
        assert call["code"] == "510300.SH"
        assert call["period"] == "1d"
        assert call["start_time"] == "20250104"  # 本地最后日期 + 1 天
        assert call["end_time"] == "20250106"
        assert len(df) == 6

    def test_fetch_downloads_when_no_local_data(self, provider):
        """本地无任何数据 → 全区间下载。"""
        df = provider._fetch_and_transform_data(
            "510300.SH", "2025-01-01", "2025-01-05", "daily"
        )

        assert len(fake_xtdata.download_calls) == 1
        call = fake_xtdata.download_calls[0]
        assert call["start_time"] == "20250101"
        assert call["end_time"] == "20250105"
        assert len(df) == 5

    def test_fetch_auto_download_disabled(self, provider):
        """auto_download=False → 不下载，仅读本地已有数据。"""
        p = XtDataProvider(auto_download=False)
        fake_xtdata.local["510300.SH"] = _make_bars(["20250101"])

        df = p._fetch_and_transform_data("510300.SH", "2025-01-01", "2025-01-05", "daily")

        assert fake_xtdata.download_calls == []
        assert len(df) == 1

    def test_download_no_progress_only_warns(self, provider):
        """下载无进展（未来日期/节假日）→ 不抛异常，返回已有数据。"""
        fake_xtdata.local["510300.SH"] = _make_bars(["20250101"])
        fake_xtdata.no_progress_symbols.add("510300.SH")

        df = provider._fetch_and_transform_data(
            "510300.SH", "2025-01-01", "2025-01-05", "daily"
        )

        assert len(fake_xtdata.download_calls) == 1
        assert len(df) == 1

    def test_download_failure_raises_data_not_available(self, provider):
        """下载抛异常 → DataNotAvailableError。"""
        from ml4t.data.core.exceptions import DataNotAvailableError

        fake_xtdata.local["510300.SH"] = _make_bars(["20250101"])
        fake_xtdata.fail_download_symbols.add("510300.SH")

        with pytest.raises(DataNotAvailableError):
            provider._fetch_and_transform_data(
                "510300.SH", "2025-01-01", "2025-01-05", "daily"
            )

    def test_probe_failure_raises_data_not_available(self, provider):
        """探测抛异常（如客户端未启动）→ DataNotAvailableError。"""
        from ml4t.data.core.exceptions import DataNotAvailableError

        fake_xtdata.fail_probe_symbols.add("510300.SH")

        with pytest.raises(DataNotAvailableError):
            provider._fetch_and_transform_data(
                "510300.SH", "2025-01-01", "2025-01-05", "daily"
            )


# ---------------------------------------------------------------------------
# 会话内下载缓存
# ---------------------------------------------------------------------------


class TestSessionCache:
    def test_prevents_repeat_download(self, provider):
        """同标的同区间第二次取数 → 不再触发下载。"""
        fake_xtdata.local["510300.SH"] = _make_bars(["20250101"])

        provider._fetch_and_transform_data("510300.SH", "2025-01-01", "2025-01-05", "daily")
        provider._fetch_and_transform_data("510300.SH", "2025-01-01", "2025-01-05", "daily")

        assert len(fake_xtdata.download_calls) == 1

    def test_hits_for_earlier_end(self, provider):
        """已确保到 01-05，再请求更早的 end → 命中缓存。"""
        fake_xtdata.local["510300.SH"] = _make_bars(["20250101"])

        provider._fetch_and_transform_data("510300.SH", "2025-01-01", "2025-01-05", "daily")
        provider._fetch_and_transform_data("510300.SH", "2025-01-02", "2025-01-04", "daily")

        assert len(fake_xtdata.download_calls) == 1

    def test_extends_for_later_end(self, provider):
        """请求更晚的 end → 重新探测并补齐新缺口。"""
        fake_xtdata.local["510300.SH"] = _make_bars(["20250101"])

        provider._fetch_and_transform_data("510300.SH", "2025-01-01", "2025-01-05", "daily")
        provider._fetch_and_transform_data("510300.SH", "2025-01-01", "2025-01-08", "daily")

        assert len(fake_xtdata.download_calls) == 2
        assert fake_xtdata.download_calls[1]["start_time"] == "20250106"

    def test_cache_is_per_instance(self, provider):
        """缓存不跨实例共享：新实例面对同样的本地库缺口会再次下载。

        注意：本地库本身是全局共享的（模拟迅投客户端），所以两次 fetch
        之间把本地库重置回缺口状态，以验证新实例不会命中旧实例的缓存。
        """
        other = XtDataProvider()
        fake_xtdata.local["510300.SH"] = _make_bars(["20250101"])

        provider._fetch_and_transform_data("510300.SH", "2025-01-01", "2025-01-05", "daily")
        # 模拟新会话：本地库仍有缺口（未下载），但新实例缓存为空
        fake_xtdata.local["510300.SH"] = _make_bars(["20250101"])
        other._fetch_and_transform_data("510300.SH", "2025-01-01", "2025-01-05", "daily")

        assert len(fake_xtdata.download_calls) == 2


# ---------------------------------------------------------------------------
# download() 公共方法
# ---------------------------------------------------------------------------


class TestDownloadMethod:
    def test_download_batch(self, provider):
        """批量预下载多只标的，返回每只标的最后数据日期。"""
        fake_xtdata.local["510300.SH"] = _make_bars(["20250101"])

        results = provider.download(
            ["510300.SH", "510500.SH"], "2025-01-01", "2025-01-05", "daily"
        )

        assert results == {"510300.SH": "20250105", "510500.SH": "20250105"}
        assert len(fake_xtdata.download_calls) == 2

    def test_download_accepts_single_symbol(self, provider):
        """download 接受单个字符串。"""
        results = provider.download("510300.SH", "2025-01-01", "2025-01-03", "daily")

        assert results == {"510300.SH": "20250103"}
        assert len(fake_xtdata.download_calls) == 1

    def test_download_failure_returns_none_and_continues(self, provider, monkeypatch):
        """单只标的下载失败 → 返回 None，不中断其他标的。"""
        import ml4t.data.providers.xtdata_provider as mod

        orig = mod.XtDataProvider._ensure_downloaded

        def flaky(self, symbol, start, end, period):
            if symbol == "BAD.SH":
                raise RuntimeError("client not connected")
            return orig(self, symbol, start, end, period)

        monkeypatch.setattr(mod.XtDataProvider, "_ensure_downloaded", flaky)

        results = provider.download(
            ["BAD.SH", "510300.SH"], "2025-01-01", "2025-01-03", "daily"
        )

        assert results == {"BAD.SH": None, "510300.SH": "20250103"}

    def test_download_skips_covered_symbols(self, provider):
        """已覆盖的标的在 download() 中自动跳过。"""
        fake_xtdata.local["510300.SH"] = _make_bars(
            ["20250101", "20250102", "20250103"]
        )

        provider.download(["510300.SH"], "2025-01-01", "2025-01-03", "daily")

        assert fake_xtdata.download_calls == []


# ---------------------------------------------------------------------------
# 批量取数与端到端
# ---------------------------------------------------------------------------


class TestBatchAndEndToEnd:
    def test_fetch_batch_ohlcv_downloads_missing(self, provider):
        """批量取数：缺失标的先补齐，再统一取数。"""
        fake_xtdata.local["510300.SH"] = _make_bars(["20250101"])

        df = provider.fetch_batch_ohlcv(
            ["510300.SH", "510500.SZ"], "2025-01-01", "2025-01-03", "daily"
        )

        assert {c["code"] for c in fake_xtdata.download_calls} == {"510300.SH", "510500.SZ"}
        assert df["symbol"].n_unique() == 2
        assert len(df) == 6

    def test_fetch_batch_ohlcv_download_failure_continues(self, provider, monkeypatch):
        """批量取数：下载失败的标的不中断其他标的。"""
        import ml4t.data.providers.xtdata_provider as mod

        orig = mod.XtDataProvider._ensure_downloaded

        def flaky(self, symbol, start, end, period):
            if symbol == "BAD.SH":
                raise RuntimeError("boom")
            return orig(self, symbol, start, end, period)

        monkeypatch.setattr(mod.XtDataProvider, "_ensure_downloaded", flaky)
        fake_xtdata.local["510300.SH"] = _make_bars(["20250101"])

        df = provider.fetch_batch_ohlcv(
            ["BAD.SH", "510300.SH"], "2025-01-01", "2025-01-03", "daily"
        )

        assert set(df["symbol"].to_list()) == {"510300.SH"}
        assert len(df) == 3

    def test_fetch_ohlcv_end_to_end(self, provider):
        """走 BaseProvider 模板方法：下载补齐 + 校验通过。"""
        fake_xtdata.local["510300.SH"] = _make_bars(["20250101", "20250102"])

        df = provider.fetch_ohlcv("510300.SH", "2025-01-01", "2025-01-03", "daily")

        assert len(df) == 3  # 下载补齐了 01-03
        assert list(df.columns[:7]) == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
