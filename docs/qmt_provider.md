# QmtProvider — QMT HTTP 转发接口数据接入

> 通过 QMT 客户端内部运行的 HTTP 转发服务（`QMT_HTTP_API封装.py`）获取 A 股 / ETF 市场数据，接入 ml4t-data 标准 OHLCV 管线。

---

## 概述

`QmtProvider` 继承自 `BaseProvider`，将 QMT HTTP 服务返回的 `dict[str, DataFrame.to_dict()]` 序列化数据反序列化，转换为 ml4t-data 标准的长格式 Polars DataFrame。输出结果与 `XtDataProvider` 保持一致，区别在于数据来源是 HTTP 接口而非本地 xtquant 库。

- **文件位置**：`src/ml4t/data/providers/qmt_provider.py`
- **数据源**：QMT 客户端内加载的 HTTP 服务（`QMT_HTTP_API封装.py`，默认 `http://127.0.0.1:10086`）
- **支持标的**：A 股、ETF、指数等 QMT 覆盖的所有品种
- **数据频率**：日线、周线、月线、分钟线、小时线

---

## 环境要求

| 依赖 | 版本要求 | 说明 |
|---|---|---|
| Python | 与 ml4t-data 一致 | 无需安装 xtquant（区别于 XtDataProvider） |
| QMT 客户端 | 最新版 | 行情客户端需启动并连接行情 |
| `QMT_HTTP_API封装.py` | 仓库内脚本 | 需在 QMT 客户端内加载运行 |
| httpx | ml4t-data 核心依赖 | HTTP 客户端 |

### 服务端启动

`QMT_HTTP_API封装.py` 必须在 QMT 客户端**内部**运行（依赖客户端内的 `ContextInfo` 环境）：

1. 打开迅投 QMT 客户端，新建一个 Python 策略
2. 将 `QMT_HTTP_API封装.py` 内容加载到策略中并运行
3. 确认服务监听在 `http://127.0.0.1:10086`（默认端口，可用 `PORT` 常量修改）

> 客户端侧 `QmtProvider` 的 `base_url` 与 `token` 必须与服务端配置一致（默认 `http://127.0.0.1:10086`、`X-Token: 123456789`）。

---

## 快速开始

### 基本用法

```python
from ml4t.data.providers.qmt_provider import QmtProvider

provider = QmtProvider()

# 获取单个标的
df = provider.fetch_ohlcv("510300.SH", "2025-01-01", "2025-06-30", frequency="daily")
print(df.head())
```

### 批量获取

```python
# 批量获取（服务端原生支持多标的，一次 HTTP 请求返回全部）
symbols = ["510300.SH", "510500.SH", "159915.SZ"]
df = provider.fetch_batch_ohlcv(symbols, "2025-01-01", "2025-06-30")
print(f"Total rows: {len(df)}")
print(f"Symbols: {df['symbol'].unique().to_list()}")
```

### 通过 DataManager 注册使用

```python
from ml4t.data import DataManager
from ml4t.data.providers.qmt_provider import QmtProvider

dm = DataManager()
dm._provider_manager.register_provider("qmt", QmtProvider)

# 标准接口调用
df = dm.fetch("510300.SH", "2025-01-01", "2025-06-30", provider="qmt")
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
| `timestamp` | Datetime[μs, Asia/Shanghai] | 交易日期（由服务端 `time` 毫秒时间戳还原） |
| `symbol` | String | 股票代码（大写） |
| `open` | Float64 | 开盘价 |
| `high` | Float64 | 最高价 |
| `low` | Float64 | 最低价 |
| `close` | Float64 | 收盘价 |
| `volume` | Float64 | 成交量 |
| `amount` | Float64 | 成交额（额外列） |
| `settelementPrice` | Float64 | 结算价（额外列，服务端返回时保留） |
| `openInterest` | Int64 | 持仓量（额外列，服务端返回时保留） |

> **注意**：
> - `settelementPrice` 的拼写是 QMT 原始返回，非笔误。
> - 服务端字段不含 xtdata 特有的 `suspendFlag` / `preClose`，故不输出。
> - `timestamp` 为 **Asia/Shanghai 时区**（与 XtDataProvider 的 UTC 不同），`dt.date()` 与 A 股交易日历一致。

---

## 初始化参数

```python
QmtProvider(
    base_url="http://127.0.0.1:10086",  # HTTP 服务地址（与服务端 PORT 一致）
    token="123456789",                  # 认证 token（与服务端 TOKEN 一致）
    dividend_type="front",              # 复权: 'front'=前复权, 'back'=后复权, 'none'=不复权
    rate_limit=None,                    # 限流: (calls, period_seconds)
    session_config=None,                # 额外 httpx.Client 配置（如 timeout）
)
```

---

## 与 XtDataProvider 的差异

| 维度 | XtDataProvider | QmtProvider |
|---|---|---|
| 数据来源 | 本地 xtquant 库直接返回 DataFrame | HTTP 接口返回 `DataFrame.to_dict()` 序列化数据 |
| xtquant 依赖 | 必需（仅支持 Python 3.6~3.11） | 不需要 |
| 时间戳 | index 为 `'YYYYMMDD'` 字符串，输出 UTC | 服务端 `time` 毫秒时间戳列还原，输出 Asia/Shanghai |
| 小时线周期 | `60m` | `1h` |
| `download()` / `auto_download` | 支持（自动增量补齐本地库） | 不支持（服务端未暴露下载接口） |
| 停牌填充 `fill_data` | 客户端可配置 | 服务端固定为 True，客户端不可配置 |
| 额外列 | `suspendFlag`、`preClose` 等 | 无 `suspendFlag` / `preClose` |

> **数据可用性同样依赖本地库**：服务端的 `get_market_data_ex` 与 xtdata 一样只读取 QMT 客户端本地**已下载**的数据。qmt 无法自动触发下载，需在 QMT 客户端中先下载好目标区间数据。

---

## 频率映射

| 传入值 | QMT period |
|---|---|
| `daily` / `1day` / `1d` | `1d` |
| `weekly` / `1week` / `1w` | `1w` |
| `monthly` / `1month` / `1mon` | `1mon` |
| `minute` / `1minute` / `1m` | `1m` |
| `5minute` / `5m` | `5m` |
| `15minute` / `15m` | `15m` |
| `30minute` / `30m` | `30m` |
| `hourly` / `1hour` / `60m` | `1h`（注意：xtdata 为 `60m`） |

---

## 数据处理流程

```
POST /api/data/market_data_ex  (fields, stock_code, period, start_time, end_time, count=-1, dividend_type)
    │
    ▼
{"data": {symbol: {field: {index: value}}}}    # DataFrame.to_dict() 序列化
    │
    ▼
pd.DataFrame.from_dict(serialized)             # 反序列化重建
    │
    ├─→ dropna(subset=ohlc, how='all')         # 清除 OHLC 全空行（停牌填充行等）
    ├─→ 'time' 列毫秒时间戳                     # 先标 UTC 再转 Asia/Shanghai
    ├─→ pl.from_pandas()                       # pandas → polars
    ├─→ filter(timestamp is_not_null)          # 过滤时间戳解析失败行
    ├─→ pl.lit(symbol).alias('symbol')         # 添加 symbol 列
    ├─→ cast(Float64)                          # OHLCV 类型统一
    └─→ select(std_cols + extra_cols)          # 标准列在前
    │
    ▼
pl.DataFrame[timestamp, symbol, open, high, low, close, volume, ...]
```

---

## 常见问题

### Q: 报错 NetworkError: QMT HTTP 服务连接失败？

`QmtProvider` 无法连接到 HTTP 服务。请确认：

1. QMT 客户端已启动并连接行情
2. `QMT_HTTP_API封装.py` 已在客户端内加载运行
3. `base_url` 端口与客户端侧 `PORT` 常量一致（默认 10086）

### Q: 报错 DataNotAvailableError / HTTP 状态码错误？

服务端返回了错误状态（如 401 token 不匹配、400 参数错误）。请检查 `token` 是否与服务端 `TOKEN` 常量一致，以及请求参数是否合法。响应体中的 `error` 字段会带在异常 `details` 中。

### Q: 取到的数据只到某一天，没有更新到请求的截止日？

服务端的 `get_market_data_ex` 只读取 QMT 客户端本地**已下载**的数据。qmt 没有 `download()` 能力，请在 QMT 客户端中先下载目标区间数据后重试。

### Q: 某个标的一直报 SymbolNotFoundError？

服务端返回中不存在该标的，或该标的在区间内所有行的 OHLC 均为空（如停牌填充行）。请确认代码格式正确（如 `510300.SH`）且客户端本地已下载该标的。

### Q: 额外列会丢失吗？

不会。BaseProvider 的 `_validate_ohlcv` 只要求 6 列（timestamp, symbol, open, high, low, close, volume）存在，服务端返回的其余列（如 `amount`、`settelementPrice`、`openInterest`）会原样保留。

### Q: 批量获取时某个标的失败会中断吗？

不会。`fetch_batch_ohlcv` 一次 HTTP 请求获取全部标的，单个标的缺失或转换失败只计入 `failed_symbols`（日志警告），成功标的正常返回；全部失败时返回空表。

---

## 相关文档

- [xtdata_provider.md](./xtdata_provider.md) — XtDataProvider（输出对齐基准）
- `QMT\QMT_HTTP_API封装.py` — 服务端实现（需在 QMT 客户端内加载，独立工作区）
- `QMT\QMT_Client_Demo.py` — requests 客户端 Demo
- [base.py](../src/ml4t/data/providers/base.py) — BaseProvider 基类
