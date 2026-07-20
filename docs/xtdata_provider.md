# XtDataProvider — 迅投 A 股数据接入

> 通过 xtquant.xtdata 获取 A 股 / ETF 市场数据，接入 ml4t-data 标准 OHLCV 管线。

---

## 概述

`XtDataProvider` 继承自 `BaseProvider`，将 xtdata 的 `dict[str, DataFrame]` 批量返回格式转换为 ml4t-data 标准的长格式 Polars DataFrame。

- **文件位置**：`src/ml4t/data/providers/xtdata_provider.py`
- **数据源**：迅投 xtquant（需行情客户端运行）
- **支持标的**：A 股、ETF、指数等 xtdata 覆盖的所有品种
- **数据频率**：日线、周线、月线、分钟线

---

## 环境要求

| 依赖 | 版本要求 | 说明 |
|---|---|---|
| Python | 3.11 | xtquant .pyd 最高支持 cp311 |
| xtquant | 从 QMT 客户端获取 | 需拷贝至 conda 环境 site-packages |
| polars | >=0.20.0 | ml4t-data 核心依赖 |
| pandas | >=2.0.0 | xtdata 返回 pandas DataFrame |

### 环境初始化

```bash
# 创建环境
conda create -n ml4t python=3.11 -y
conda activate ml4t

# 安装 ml4t-data 依赖
cd research/ML4T/data
pip install -e ".[dev]"

# 拷贝 xtquant
```

---

## 快速开始

### 基本用法

```python
from ml4t.data.providers.xtdata_provider import XtDataProvider

provider = XtDataProvider()

# 获取单个标的
df = provider.fetch_ohlcv("510300.SH", "2025-01-01", "2025-06-30", frequency="daily")
print(df.head())
```

### 批量获取

```python
# 批量获取（利用 xtdata 原生批量接口，更高效）
symbols = ["510300.SH", "510500.SH", "159915.SZ"]
df = provider.fetch_batch_ohlcv(symbols, "2025-01-01", "2025-06-30")
print(f"Total rows: {len(df)}")
print(f"Symbols: {df['symbol'].unique().to_list()}")
```

### 通过 DataManager 注册使用

```python
from ml4t.data import DataManager

dm = DataManager()
dm._provider_manager.register_provider("xtdata", XtDataProvider)

# 标准接口调用
df = dm.fetch("510300.SH", "2025-01-01", "2025-06-30", provider="xtdata")
```

---

## 数据格式

### 输入参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `symbol` | str | 股票代码，如 `510300.SH`、`000001.SZ` |
| `start` | str | 起始日期 `YYYY-MM-DD`（inclusive） |
| `end` | str | 截止日期 `YYYY-MM-DD`（inclusive） |
| `frequency` | str | `daily` / `weekly` / `monthly` / `minute` 等 |

### 输出 Schema

| 列名 | 类型 | 说明 |
|---|---|---|
| `timestamp` | Datetime[μs, UTC] | 交易日期 |
| `symbol` | String | 股票代码（大写） |
| `open` | Float64 | 开盘价 |
| `high` | Float64 | 最高价 |
| `low` | Float64 | 最低价 |
| `close` | Float64 | 收盘价 |
| `volume` | Float64 | 成交量 |
| `amount` | Float64 | 成交额（额外列） |
| `settelementPrice` | Float64 | 结算价（额外列） |
| `openInterest` | Int64 | 持仓量（额外列） |
| `preClose` | Float64 | 前收盘价（额外列） |
| `suspendFlag` | Int32 | 停牌标记，1=停牌（额外列） |

> **注意**：`settelementPrice` 的拼写是 xtdata 原始返回，非笔误。

---

## 初始化参数

```python
XtDataProvider(
    dividend_type="front",  # 复权: 'front'=前复权, 'back'=后复权, 'none'=不复权
    fill_data=True,         # 是否填充非交易日
    rate_limit=None,        # 限流: (calls, period_seconds)
)
```

---

## 频率映射

| 传入值 | xtdata period |
|---|---|
| `daily` / `1day` / `1d` | `1d` |
| `weekly` / `1week` / `1w` | `1w` |
| `monthly` / `1month` / `1mon` | `1mon` |
| `minute` / `1minute` / `1m` | `1m` |
| `5minute` / `5m` | `5m` |
| `15minute` / `15m` | `15m` |
| `30minute` / `30m` | `30m` |
| `hourly` / `1hour` / `60m` | `60m` |

---

## 数据处理流程

```
xtdata.get_market_data_ex(stock_list, period, start_time, end_time)
    │
    ▼
dict[symbol] → pd.DataFrame(index='YYYYMMDD', cols=[open,high,...])
    │
    ├─→ dropna(subset=ohlc, how='all')    # 清除上市前空值行
    ├─→ reset_index()                      # index → 'index' 列
    ├─→ pl.from_pandas()                   # pandas → polars
    ├─→ strptime('index' → timestamp)      # 日期解析
    ├─→ pl.lit(symbol).alias('symbol')     # 添加 symbol 列
    ├─→ cast(Float64)                      # OHLCV 类型统一
    └─→ select(std_cols + extra_cols)      # 标准列在前
    │
    ▼
pl.DataFrame[timestamp, symbol, open, high, low, close, volume, ...]
```

---

## 常见问题

### Q: 报错 "无法连接行情服务"

确保迅投 QMT 或极速交易版客户端已启动并连接行情。

### Q: 报错 ImportError: No module named 'xtquant.IPythonApiClient'

xtquant 的 `.pyd` 编译文件只支持 Python 3.6~3.11，不支持 3.12+。请使用 Python 3.11 环境。

### Q: 额外列会丢失吗？

不会。BaseProvider 的 `_validate_ohlcv` 只要求 6 列（timestamp, symbol, open, high, low, close, volume）存在，不会删除额外列。

---

## 相关文档

- [ADAPTATION_PLAN.md](./ADAPTATION_PLAN.md) — 适配计划与设计决策
- [base.py](../src/ml4t/data/providers/base.py) — BaseProvider 基类
