"""QmtProvider vs XtDataProvider 真实数据对比（分阶段采集，最终验证）。

QMT 客户端同一时刻只允许一种数据访问模式（xtquant 本地库 / HTTP 服务），
因此采集必须分阶段执行：

    1) QMT 客户端切换为 xtquant 本地库模式后采集基准数据：
       python examples/qmt_xtdata_compare.py save-xtdata <out.parquet> [start] [end]

    2) 切换为 HTTP 服务模式（运行 QMT_HTTP_API封装.py）后采集对比数据：
       python examples/qmt_xtdata_compare.py save-qmt <out.parquet> [start] [end]

    3) 对比两个文件（无需 QMT 客户端）：
       python examples/qmt_xtdata_compare.py compare <xt.parquet> <qmt.parquet>

对比标的：000001.SH / 399001.SZ / 399006.SZ（日频数据本地完整）。
保存文件统一 schema：[date(YYYY-MM-DD), symbol, open, high, low, close, volume, amount]，
比较时按 (symbol, date) 对齐，避免两侧时区差异（xtdata 输出 UTC，qmt 输出 Asia/Shanghai）。

示例（PowerShell）：
    python examples/qmt_xtdata_compare.py save-xtdata $env:TEMP/qmt_xt_xtdata.parquet
    python examples/qmt_xtdata_compare.py save-qmt $env:TEMP/qmt_xt_qmt.parquet
    python examples/qmt_xtdata_compare.py compare $env:TEMP/qmt_xt_xtdata.parquet $env:TEMP/qmt_xt_qmt.parquet
"""

from __future__ import annotations

import sys

import polars as pl

from ml4t.data.providers.qmt_provider import QmtProvider
from ml4t.data.providers.xtdata_provider import XtDataProvider

SYMBOLS = ["000001.SH", "399001.SZ", "399006.SZ"]
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2026-08-13"
FREQUENCY = "daily"

# 双方共有且语义一致的数值列（xtdata 额外的 suspendFlag/preClose、qmt 无，不参与）
COMPARE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def _save_frame(df: pl.DataFrame, path: str, tag: str) -> None:
    """提取 date 列并按统一 schema 保存 parquet。"""
    out = df.with_columns(pl.col("timestamp").dt.date().cast(pl.Utf8).alias("date"))
    keep = ["date", "symbol"] + [c for c in COMPARE_COLS if c in out.columns]
    out.select(keep).sort(["symbol", "date"]).write_parquet(path)
    print(f"[{tag}] {len(out)} 行 -> {path}")
    print(f"[{tag}] 标的: {out['symbol'].unique().to_list()}")


def cmd_save_xtdata(path: str, start: str, end: str) -> None:
    """阶段 1：xtquant 本地库模式采集（基准）。"""
    print(f"拉取 XtDataProvider: {start} ~ {end}, {SYMBOLS}")
    p = XtDataProvider(auto_download=False)
    try:
        df = p.fetch_batch_ohlcv(SYMBOLS, start, end, FREQUENCY)
        _save_frame(df, path, "xtdata")
    finally:
        p.close()


def cmd_save_qmt(path: str, start: str, end: str) -> None:
    """阶段 2：QMT HTTP 服务模式采集（对比）。"""
    print(f"拉取 QmtProvider: {start} ~ {end}, {SYMBOLS}")
    p = QmtProvider()
    try:
        df = p.fetch_batch_ohlcv(SYMBOLS, start, end, FREQUENCY)
        _save_frame(df, path, "qmt")
    finally:
        p.close()


def cmd_compare(xt_path: str, qmt_path: str) -> None:
    """阶段 3：对比两个文件（不访问 QMT 客户端）。"""
    xt = pl.read_parquet(xt_path)
    qmt = pl.read_parquet(qmt_path)

    merged = xt.join(
        qmt,
        on=["symbol", "date"],
        how="full",
        suffix="_qmt",
        coalesce=True,
    ).sort(["symbol", "date"])

    print(f"xtdata 文件: {len(xt)} 行, qmt 文件: {len(qmt)} 行")

    for symbol in sorted(merged["symbol"].unique().to_list()):
        sub = merged.filter(pl.col("symbol") == symbol)
        only_xt = sub.filter(pl.col("open_qmt").is_null())
        only_qmt = sub.filter(pl.col("open").is_null())
        common = sub.filter(pl.col("open").is_not_null() & pl.col("open_qmt").is_not_null())

        print(f"\n=== {symbol} ===")
        print(f"  共同 {len(common)} 天 | 仅 xtdata {len(only_xt)} 天 | 仅 qmt {len(only_qmt)} 天")
        if len(only_xt):
            print(f"    仅 xtdata: {only_xt['date'].to_list()[:5]}")
        if len(only_qmt):
            print(f"    仅 qmt: {only_qmt['date'].to_list()[:5]}")

        if common.is_empty():
            print("  无共同日期，跳过数值对比")
            continue

        print("  共同日最大绝对差:")
        for col in COMPARE_COLS:
            col_qmt = f"{col}_qmt"
            if col not in common.columns or col_qmt not in common.columns:
                continue
            diff = (common[col] - common[col_qmt]).abs()
            max_abs = diff.max() if len(diff) else None
            flag = "  <-- 差异!" if max_abs is not None and max_abs > 1e-9 else ""
            val = "N/A" if max_abs is None else f"{max_abs:.6g}"
            print(f"    {col:8s}: {val}{flag}")

        rels = (
            (common["close"] - common["close_qmt"]).abs() / common["close"].abs()
        ).filter(common["close"] != 0)
        if len(rels):
            print(f"  close 最大相对差: {rels.max():.2e}, 平均: {rels.mean():.2e}")

    print("\n对比完成 ✅")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    cmd = args[0]
    if cmd == "save-xtdata":
        path = args[1]
        start = args[2] if len(args) > 2 else DEFAULT_START
        end = args[3] if len(args) > 3 else DEFAULT_END
        cmd_save_xtdata(path, start, end)
    elif cmd == "save-qmt":
        path = args[1]
        start = args[2] if len(args) > 2 else DEFAULT_START
        end = args[3] if len(args) > 3 else DEFAULT_END
        cmd_save_qmt(path, start, end)
    elif cmd == "compare":
        if len(args) < 3:
            print("用法: compare <xt.parquet> <qmt.parquet>")
            sys.exit(1)
        cmd_compare(args[1], args[2])
    else:
        print(f"未知命令: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
