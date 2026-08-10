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
    auto_download=True,     # 取数前自动探测并增量下载缺失数据
)
```

---

## 自动下载机制

`get_market_data_ex` 只能读取迅投本地库中**已下载**的数据：未下载过的标的返回空，
下载截止日早于请求 `end` 时，无论传什么日期都只会返回截至上次下载的数据。

默认 `auto_download=True` 时，每次取数前自动执行：

1. **探测**：轻量查询本地库最后数据日期（`count=1`）
2. **补齐**：存在缺口时调用 `download_history_data` 增量下载（串行，只补缺口不重下全量）
3. **取数**：下载完成后再调用 `get_market_data_ex` 返回完整区间数据

同一进程内相同标的与频率的下载结果会被缓存，避免重复触发下载；
程序重启后缓存清空，重新探测（探测开销极小）。

### 批量预下载

收盘后可用 `download()` 显式批量更新本地库，之后正式取数时探测即可命中，零额外开销：

```python
provider = XtDataProvider()
# 返回 dict[symbol, 本地最后数据日期 'YYYYMMDD']，失败或范围内无数据为 None
results = provider.download(
    ["510300.SH", "510500.SH", "159915.SZ"],
    "2025-01-01",
    "2025-06-30",
    frequency="daily",
)
```

若希望完全在迅投客户端中手动管理下载、取数只读本地库，可关闭自动下载：

```python
provider = XtDataProvider(auto_download=False)
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

### Q: 取到的数据只到某一天，没有更新到请求的截止日？

通常是迅投本地库未下载到请求日期。默认 `auto_download=True` 会自动探测并增量补齐；
若关闭了自动下载（或想手动控制），请在迅投客户端中下载数据后重试。

### Q: 额外列会丢失吗？

不会。BaseProvider 的 `_validate_ohlcv` 只要求 6 列（timestamp, symbol, open, high, low, close, volume）存在，不会删除额外列。

---

## 相关文档

- [ADAPTATION_PLAN.md](./ADAPTATION_PLAN.md) — 适配计划与设计决策
- [base.py](../src/ml4t/data/providers/base.py) — BaseProvider 基类
