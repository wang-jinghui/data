# QmtLocalProvider — QMT 本地缓存离线接入

> 直接解析 QMT（含 MiniQMT）客户端缓存目录 `datadir` 中的二进制 K 线文件，**不要求 QMT 客户端在线、不要求安装 xtquant**，接入 ml4t-data 标准 OHLCV 管线。

---

## 概述

`QmtLocalProvider` 继承自 `BaseProvider`，读取 QMT 客户端已经下载到本地的历史行情缓存，将 64 字节定长二进制记录解析为 ml4t-data 标准长格式 Polars DataFrame。输出与 `QmtProvider` / `XtDataProvider` 保持一致。

- **文件位置**：`src/ml4t/data/providers/qmt_local_provider.py`（Provider）+ `src/ml4t/data/providers/qmt_datadir.py`（解析层）
- **数据源**：QMT 客户端缓存目录 `datadir`（`<客户端目录>\datadir` 或 `<客户端目录>\userdata_mini\datadir`）
- **支持标的**：客户端本地已有缓存的 A 股、ETF、指数、可转债等（`SH` / `SZ` / `BJ`）
- **数据频率**：日线（原生）、1 分钟（原生）、5/15/30 分钟与小时线（由 1 分钟聚合）、周/月线（由日线聚合）
- **依赖**：仅 `numpy` + `polars`（均为 ml4t-data 核心依赖）

`QmtLocalProvider` 与 `QmtProvider` 互补：前者离线读缓存、速度快、可批量；后者在线取数、数据更新鲜、可查询证券元数据。

---

## 环境要求

| 依赖 | 版本要求 | 说明 |
|---|---|---|
| Python | 与 ml4t-data 一致 | 无需安装 xtquant |
| numpy | ml4t-data 核心依赖 | 零拷贝解析定长记录 |
| QMT 客户端 | 已安装并**曾经下载过**目标数据 | 取数时**不需要**客户端处于运行状态 |
| QMT HTTP 服务 | 不需要 | 区别于 `QmtProvider` |

### 定位 datadir

`QmtDataDir.discover()` 按以下优先级定位缓存目录：

1. 显式传入 `data_dir`
2. 环境变量 `ML4T_QMT_DATA_DIR`，其次 `QMT_DATA_DIR`
3. 在各本地磁盘根下查找名字含 `qmt` 的目录，尝试 `<dir>\datadir` 与 `<dir>\userdata_mini\datadir`

```powershell
# 推荐：显式设置环境变量，避免全盘扫描
$env:ML4T_QMT_DATA_DIR = "D:\国金证券QMT交易端\datadir"
```

---

## 快速开始

### 自动发现

```python
from ml4t.data.providers.qmt_local_provider import QmtLocalProvider

provider = QmtLocalProvider()          # 自动发现 datadir
print(provider.data_dir)               # D:\国金证券QMT交易端\datadir

df = provider.fetch_ohlcv("000001.SZ", "2024-01-01", "2024-06-30", frequency="daily")
print(df.head())
```

### 显式指定缓存目录

```python
provider = QmtLocalProvider(
    data_dir=r"D:\国金证券QMT交易端\datadir",
    dividend_type="front",
)
```

### 查看本地已有标的

```python
# 所有非空的缓存文件
symbols = provider.available_symbols(period="1d", market="SH")
print(len(symbols), symbols[:5])       # 例如 23834 ['000001.SH', '000002.SH', ...]
```

⚠️ 客户端会为大量**未下载行情**的品种写入「代码占位」文件：收盘价恒为 100.000、成交量为 0（例如 `230000.SH`、`190000.SZ`）。它们会被当作有效标的列出。加 `traded_only=True` 可剔除：

```python
# 只保留首尾至少一条记录有成交的标的（多花几秒做小随机读）
traded = provider.available_symbols(period="1d", traded_only=True)
print(len(traded))                     # 同一份缓存：45123 → 7400
```

### 检查缓存覆盖与新鲜度

`coverage()` 逐文件读首尾记录，汇总标的数、记录条数分位数、起止日期、陈旧标的数，用来回答「本地还缺哪些数据」：

```python
for stats in provider.coverage(period="1d", stale_days=30):
    print(
        stats.market, stats.symbols, stats.traded,
        f"{stats.first_date}~{stats.last_date}",
        f"rows={stats.rows_median:.0f}", f"stale={stats.stale}",
    )
```

真实缓存（4.5 万个日线文件）输出：

```text
SH 23834 2568 1990-12-19~2026-10-22 rows=695  stale=22720
SZ 21279 4827 1991-01-05~2026-10-22 rows=652  stale=20229
BJ    10    5 2024-01-15~2026-08-14 rows=2    stale=6
```

字段：`symbols` 有效文件数、`traded` 有成交标的数、`rows_min/median/max` 记录条数、`first_date` 最早起始日、`last_date` 最新结束日、`median_last_date` 结束日中位数、`stale` 结束日落后超过 `stale_days` 的标的数、`unreadable` 被占用/无权限的文件数。日线全量扫描约 7 秒。

按品种细分新鲜度需要自己读代码前缀（`60`/`68`/`51`/`00`/`30`/`15` 等），`coverage()` 只按市场汇总。

### 批量读取

```python
symbols = provider.get_stock_list_in_sector("沪市基金")
df = provider.fetch_batch_ohlcv(symbols, "2024-01-01", "2024-06-30", "daily")
print(df.group_by("symbol").len())
```

默认显示 tqdm 进度条（`QMT daily: 100%|██████████| 1117/1117 [00:12<00:00, 88.8sym/s]`），调试或 CI 里可关闭：

```python
df = provider.fetch_batch_ohlcv(symbols, "2024-01-01", "2024-06-30", "daily", show_progress=False)
```

单个标的缺缓存不会中断整体：正常读取**零日志**，结束时若有失败标的只打一条汇总 warning（`Some symbols failed count=2 total=23 symbols=[...]`）；全部失败时返回空表并打 error。

### 日志与进度

本地读取的正常路径不打任何日志，噪音已全部收敛：

| 场景 | 输出 |
|---|---|
| 构造 provider | 无（不初始化限流器 / httpx 会话） |
| `fetch_ohlcv()` 单标的 | 无 |
| `fetch_batch_ohlcv()` 全部成功 | 仅进度条 |
| `fetch_batch_ohlcv()` 有失败 | 进度条 + 1 条 warning（含失败数、总数、前 10 个标的） |
| 全部失败 | 1 条 error（`No data read for any symbol`） |

进度条依赖 `tqdm`（可选依赖，缺失时自动忽略，不影响功能）。tqdm 写 **stderr**，因此 `2>&1` 重定向时会与数据流混在一起。

### 通过 DataManager 注册使用

```python
from ml4t.data import DataManager
from ml4t.data.providers.qmt_local_provider import QmtLocalProvider

dm = DataManager()
dm._provider_manager.register_provider("qmt_local", QmtLocalProvider)

df = dm.fetch("000001.SZ", "2024-01-01", "2024-06-30", provider="qmt_local")
```

---

## 数据格式

### 输入参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `symbol` | str | 证券代码，如 `000001.SZ`、`600004.SH`；也接受 `SH600004`、`SH.600004` |
| `start` | str | 起始日期 `YYYY-MM-DD`（inclusive） |
| `end` | str | 截止日期 `YYYY-MM-DD`（inclusive） |
| `frequency` | str | `daily` / `weekly` / `monthly` / `minute` / `5m` / `15m` / `30m` / `1h` 等 |

> **代码必须带市场标识**：`000001` 在 `SH` 是上证指数、在 `SZ` 是平安银行，无法推断，缺少后缀会直接报 `ValueError`。

### 输出 Schema

| 列名 | 类型 | 说明 |
|---|---|---|
| `timestamp` | Datetime[ms, Asia/Shanghai] | 交易时间（日线为当日 00:00，分钟线为 bar 结束时刻） |
| `symbol` | String | 规范化后的证券代码，如 `000001.SZ` |
| `open` | Float64 | 开盘价 |
| `high` | Float64 | 最高价 |
| `low` | Float64 | 最低价 |
| `close` | Float64 | 收盘价 |
| `volume` | Float64 | 成交量（**手**） |
| `amount` | Float64 | 成交额（**元**，额外列） |
| `suspend_flag` | Int8 | 停牌标记：`volume == 0 且 amount == 0` 时为 1（额外列） |

> **注意**：
> - `volume` 单位为「手」、`amount` 单位为「元」，与 xtdata 输出一致，**均不做缩放**。
> - `timestamp` 为 **Asia/Shanghai 时区**（与 `QmtProvider` 一致，与 `XtDataProvider` 的 UTC 不同）。
> - `suspend_flag` 只是「当日无成交」的机械标记，不代表交易所口径的停牌状态。

---

## 初始化参数

```python
QmtLocalProvider(
    data_dir=None,                        # datadir 路径；None 表示自动发现
    dividend_type="front",                # 复权: 'front'=前复权, 'back'=后复权, 'none'=不复权
    ex_rights_threshold=0.01,             # 判定除权事件的最小价差（元）
    rate_limit=None,                      # 已废弃，仅为兼容旧调用保留
    session_config=None,                  # 已废弃，仅为兼容旧调用保留
    search_roots=None,                    # 自动发现的扫描起点（默认各本地磁盘根）
)
```

> 本地读取不经过 HTTP，初始化时**不会**创建 `httpx.Client`、也**不会**注册全局限流器（`rate_limit` / `session_config` 仅保留签名兼容）。`fetch_ohlcv()` 同时覆盖了基类模板：保留入参/输出校验，但去掉重试与熔断——本地读取没有远端配额，「文件被客户端独占写」这类错误重试只会把报错延迟 4~10 秒。

---

## datadir 目录结构

```
<datadir>/
├── SH/                        # 上交所
│   ├── 86400/                 # 日线
│   │   └── 600004.DAT
│   └── 60/                    # 1 分钟线
│       └── 600004.DAT
├── SZ/                        # 深交所（同上）
├── BJ/                        # 北交所（同上）
└── 0/                         # tick 数据（<symbol>/<yyyymmdd>.dat，本 provider 不解析）
```

枚举标的时会过滤两类文件：

- `<code>_<4位>.DAT` 这类板块/财务附属文件（如 `000001_9000.DAT`）
- 不足一条记录（64 字节）的空文件

### 二进制布局

日线与分钟线共用 64 字节定长记录（小端）：

| 偏移 | 类型 | 字段 | 说明 |
|---|---|---|---|
| 0 | u32 | reserved | 保留 |
| 4 | u32 | reserved | 保留 |
| 8 | u32 | timestamp | Unix 秒 |
| 12 | u32 | open | 开盘价 ×1000 |
| 16 | u32 | high | 最高价 ×1000 |
| 20 | u32 | low | 最低价 ×1000 |
| 24 | u32 | close | 收盘价 ×1000 |
| 28 | u32 | reserved | 保留 |
| 32 | u32 | volume | 成交量（手） |
| 36 | u32 | reserved | 保留 |
| 40 | u64 | amount | 成交额（元） |
| 48 | u32 | openInterest | 持仓量 |
| 52 | u32 | reserved | 保留 |
| 56 | u32 | reserved | 保留 |
| 60 | u32 | preClose | 除权参考价 ×1000 |

> 尾部不足 64 字节的残留字节会被忽略；时间戳重复时保留最后一条（客户端重写场景）。

---

## 复权口径

文件中的 `preClose` 是**交易所除权参考价**，即 `(昨收 - 每股派息) / (1 + 送转比例)`，因此：

- 逐日因子：`f_t = preClose_t / close_{t-1}`，非除权日为 1
- 前复权：`adj_t = price_t × Π_{s>t} f_s`（**最新一根价格不变**，与 xtdata 前复权一致）
- 后复权：`adj_t = price_t / Π_{s<=t} f_s`（**最早一根价格不变**）
- 复权只缩放 OHLC，不改变 `volume` / `amount`

**噪声阈值**：`|preClose_t - close_{t-1}|` 不超过一个最小价位（0.01 元）的差值视为舍入噪声，默认不判定为除权事件。比较在整数毫元上进行——float64 下 `10.01 - 10.0 = 0.010000000000000497 > 0.01`，用浮点直接比较多会把「每 10 股派 0.1 元」这类噪声误当成除权，注入约 0.1% 的假收益。

代价是「价差恰好 0.01 元的真实除权」也会被漏掉。这类情形与噪声在数据上不可区分，如确需捕获可调小 `ex_rights_threshold`（如 `0.0005`），但要接受噪声带来的偏差。

---

## 周期派生

| 周期 | 来源 | 说明 |
|---|---|---|
| `1d` | 原生 `86400/` | 直接解析 |
| `1m` | 原生 `60/` | 直接解析 |
| `5m` / `15m` / `30m` / `1h` | `1m` 聚合 | 09:30 集合竞价归一到 09:31，按 bar **结束**时间标注，每日 48 / 16 / 8 / 4 根 |
| `1w` / `1mon` | `1d` 聚合 | 周以周一为起点；标签取区间内**最后一个交易日** |

聚合前的日期过滤发生在原始周期上，因此**首根/末根聚合 bar 可能不完整**（例如 `start` 落在月中时的首根月线）。需要完整 bar 时请把 `start` 对齐到周初/月初。

---

## 板块 / 行业成分股

QMT 客户端会把板块成分股缓存在 `datadir/Sector/<分类>/<板块名>`，内容是**逗号分隔的 `code.market` 纯文本**。本 provider 可直接解析，方法与 [QmtProvider](./qmt_provider.md) 同名同签名。

| 方法 | 参数形式 | 数据源 |
|---|---|---|
| `get_stock_list_in_sector(sectorname)` | 板块名，如 `'沪深300'`、`'SW1医药生物'`、`'我的自选'`、`'沪市基金'` | `Sector/*/<板块名>`，未命中则按代码前缀筛选 |
| `get_sector(sector)` | 指数代码或名称，如 `'000300.SH'`、`'SH000300'`、`'沪深300'` | `Sector/指数成分股板块/` |
| `get_industry(industry)` | 行业名，如 `'SW1银行'`、`'CSRC1制造业'` | `Sector/申万*行业板块/`、`Sector/证监会*行业板块/` |
| `get_weight_in_index(...)` | — | ❌ 不支持（`Weight/` 是 LevelDB） |

辅助枚举：

```python
provider.sector_kinds()          # 13 个分类：D概念 / T概念 / T行业 / 地域 / 我的 /
                                 # 指数成分股板块 / 申万一~三级 / 证监会一~二级 …
provider.sector_names()          # 2841 个板块，形如 '指数成分股板块/沪深300'
provider.sector_names(kind="申万一级行业板块")
provider.sector_symbols("申万一级行业板块/SW1医药生物")
```

真实缓存上的输出：

```python
provider.get_sector("000300.SH")            # 300 只
provider.get_sector("000905.SH")            # 500 只（中证500）
provider.get_industry("CSRC1制造业")         # 3324 只
provider.get_stock_list_in_sector("我的自选") # 3 只
```

**重名板块**：`沪深300` 同时存在于 `D概念` 和 `指数成分股板块`。`sector_symbols()` 在重名时**优先 `指数成分股板块`**；其他情况会抛 `ValueError` 并列出候选，此时需加分类前缀：

```python
provider.sector_symbols("T行业/LED")
provider.sector_symbols("医药生物", kind="T概念")
```

**代码别名**：本地板块文件以中文名命名，而 `get_sector()` 惯例传指数代码。内置了 11 个宽基指数别名（`000001`→`上证指数`、`000300`→`沪深300`、`000905`→`中证500`、`000852`→`中证1000`、`000016`→`上证50`、`000688`→`科创50`、`000906`→`中证800`、`000009`/`000010`/`399001`/`399006`）。未收录的代码会按文件名直查（例如 `'000809'`）。

**市场分类板块**（`'沪市基金'`、`'深市A股'`、`'创业板'` …）：客户端左侧这类分类由「市场 + 代码前缀」动态生成，datadir 里**没有对应文件**（`Sector/` 下的「基金重仓」「参股基金」是概念板块，不是市场分类）。本 provider 内置 17 个规则，按缓存标的的代码前缀筛选：

```python
provider.market_sector_names()                 # 17 个分类名
provider.get_stock_list_in_sector("沪市基金")  # 本地已缓存的沪市基金
provider.get_sector("沪市基金")                # 同上（get_sector 也会回退）
```

| 分类 | 代码前缀规则 |
|---|---|
| 沪深A股 / 沪市A股 / 深市A股 | SH `600/601/603/605/688/689`；SZ `000/001/002/003/300/301/302` |
| 沪市主板 / 深市主板 | SH `600/601/603/605`；SZ `000/001/002/003` |
| 创业板 / 科创板 / 北交所 | SZ `300/301/302`；SH `688/689`；BJ 全部 |
| 沪深基金 / 沪市基金 / 深市基金 | SH `50/51/52/53/55/56/58`；SZ `15/16/18` |
| 沪深指数 / 沪市指数 / 深市指数 | SH `000/880/999`；SZ `399/980` |
| 沪深转债 / 沪市转债 / 深市转债 | SH `110/111/113/118/132`；SZ `123/125/126/127/128/129` |

注意三点：

1. 返回的是**本地已缓存**的标的，不是全市场清单（在线版返回全量），补齐缓存后数量会增加。
2. 同名文件板块优先：若 `Sector/` 下真有叫 `沪市基金` 的板块文件，以文件为准。
3. **债券**（国债/企业债/公司债）没有内置分类：`12x`/`13x` 段里转债与公司债混编，规则不唯一，建议用 `sector_symbols()` 读自定义板块文件。

真实缓存上的结果：

```
沪市基金 1117   深市基金 1050   沪深基金 2167
沪市A股   350   深市A股   311   沪深A股   661
深市指数  369   沪市指数  221   沪深指数  590
深市转债  161   沪市转债  153   沪深转债  314
```

典型用法——按指数成分股批量取数：

```python
symbols = provider.get_sector("000300.SH")
df = provider.fetch_batch_ohlcv(symbols, "2024-01-01", "2024-12-31", "daily")
```

---

## 与 QmtProvider / XtDataProvider 的差异

| 维度 | XtDataProvider | QmtProvider | QmtLocalProvider |
|---|---|---|---|
| 数据来源 | 本地 xtquant 库 | QMT 内 HTTP 服务 | datadir 二进制缓存 |
| 客户端需运行 | 是 | 是 | **否** |
| xtquant 依赖 | 必需 | 不需要 | 不需要 |
| 数据新鲜度 | 取决于本地库 | 取决于本地库 | 取决于**最后一次下载** |
| 时间戳时区 | UTC | Asia/Shanghai | Asia/Shanghai |
| 小时线周期 | `60m` | `1h` | `1h`（含 `60m` 别名） |
| `download()` | 支持 | 不支持 | 不支持（只读缓存） |
| `suspendFlag` | 有 | 无 | `suspend_flag` |
| `amount` | 有 | 有 | 有 |
| ContextInfo 查询 | 支持 | 支持 | 部分支持（`Sector/` 纯文本：板块/行业成分股；市场分类按代码前缀推导；股票名称/指数权重不支持） |
| tick 数据 | 支持 | 支持 | 不解析（仅 64 字节 K 线） |

---

## 频率映射

| 传入值 | 周期 |
|---|---|
| `daily` / `1day` / `1d` | `1d` |
| `weekly` / `1week` / `1w` | `1w`（由日线聚合） |
| `monthly` / `1month` / `1mon` | `1mon`（由日线聚合） |
| `minute` / `1minute` / `1m` | `1m` |
| `5minute` / `5m` | `5m`（由 1m 聚合） |
| `15minute` / `15m` | `15m`（由 1m 聚合） |
| `30minute` / `30m` | `30m`（由 1m 聚合） |
| `hourly` / `1hour` / `1h` / `60m` | `1h`（由 1m 聚合） |

未知频率直接报 `ValueError`，**不会静默降级为日线**。

---

## 数据处理流程

```
<datadir>/<SH|SZ|BJ>/<86400|60>/<code>.DAT
    │
    ▼
np.frombuffer(payload, dtype=KLINE_DTYPE)      # 64 字节定长记录，零拷贝
    │
    ├─→ timestamp(u32 秒) → Datetime[ms]       # 先标 UTC 再转 Asia/Shanghai
    ├─→ OHLC / preClose ÷ 1000                 # 价格还原为元
    ├─→ volume / amount 保持原值                # 手 / 元，不缩放
    ├─→ unique(keep='last') + sort              # 去重、升序
    ├─→ suspend_flag = (volume==0 & amount==0)
    │
    ├─→ preClose 推导因子 → 前/后复权            # 只缩放 OHLC
    ├─→ 按日期区间过滤（inclusive）
    ├─→ 周期派生（1m → 5m/15m/30m/1h；1d → 1w/1mon）
    │
    ▼
pl.DataFrame[timestamp, open, high, low, close, volume, amount, suspend_flag]
    │
    ▼
+ symbol 列 + 标准列序 → pl.DataFrame[timestamp, symbol, open, ..., suspend_flag]
```

---

## 常见问题

### Q: 报错 FileNotFoundError: QMT datadir not found？

没有找到缓存目录。请显式传入 `data_dir`，或设置环境变量 `ML4T_QMT_DATA_DIR`。自动扫描只识别名字含 `qmt` 的客户端目录。

### Q: 报错 SymbolNotFoundError，但我明明有这只股票？

本地没有该标的**该周期**的缓存文件。注意两点：

1. 日线（`86400/`）与分钟线（`60/`）是分开下载的，只下载了日线时查分钟线必然缺失
2. 报错 `details` 里带有 `period` 与 `data_dir`，可直接据此确认路径

### Q: 数据只到几天/几个月前，没有更新？

本 provider **只读缓存**，不会触发下载。请在 QMT 客户端中补充下载后再读取。

### Q: 分钟数据只有最近几个月？

QMT 客户端本地通常只保留近期分钟数据，历史分钟线需要客户端预先下载，且不同标的覆盖区间可能不同。

### Q: 前复权价格和我印象里的不一样？

前复权以**本地最新一根**为基准（末端不变）。因此同一标的在不同时间点读到的前复权价格序列会随新的除权事件整体平移，这是前复权口径的固有性质。需要可复现的价格请用 `dividend_type="none"` 或 `"back"`。

### Q: 某些日线 `volume = 0`，但 `close` 不是 0？

这些是停牌/无成交日，文件里保留为 `preClose` 填充行，`suspend_flag` 会标为 1。是否需要剔除由使用方决定，provider 不做过滤。

### Q: `available_symbols()` 返回 4.5 万个，但我只关心几千只股票？

客户端为每个品种都建了占位文件（价格恒 100.000、成交量 0），其中真正有行情的只有几千个。用 `available_symbols(traded_only=True)` 过滤，或用 `coverage()` 看每个市场的 `traded` 数量。

### Q: 怎么知道还缺哪些数据？

```python
provider.coverage("1d")    # 各市场：标的数 / 有成交数 / 起止日期 / 陈旧数
provider.coverage("1m")    # 分钟线覆盖通常极小
```

关注 `traded`（有效标的数）、`median_last_date`（整体新鲜度）、`stale`（落后标的数）。具体到品种（股票/ETF/指数/可转债）需按代码前缀自行统计。

### Q: `get_weight_in_index()` 报 NotImplementedError？

指数权重存在 `datadir/Weight/` 的 LevelDB 中（`.ldb` / `.log` / `MANIFEST-*`），纯 Python 解析需要 leveldb 绑定，不在本 provider 范围内。需要指数权重请用 [QmtProvider](./qmt_provider.md)。

### Q: 传入 `rate_limit=(...)` 会生效吗？

不会。本地文件读取没有远端配额，限流器与 HTTP 会话都不再初始化，参数仅为兼容旧调用保留；`session_config` 同理被忽略。

### Q: 批量读取时看不到任何日志，是没跑吗？

正常。逐标的日志已移除，成功时只有进度条；失败标的在结束时汇总成一条 warning。需要逐标的细节可先 `show_progress=False` 再自己包一层循环统计。

### Q: 报 `ValueError: Sector 'xxx' is ambiguous`？

同名板块存在于多个分类。除 `指数成分股板块` 会自动优先外，其余需加分类前缀：`sector_symbols("T行业/LED")`，或用 `sector_names()` 先确认唯一写法。

### Q: `get_stock_list_in_sector("沪市基金")` 抛 FileNotFoundError？

升级到支持市场分类的版本后不会再抛——`Sector/` 下确实没有这个文件（只有「基金重仓」这类概念板块）。请在 `Sector/` 找不到时依赖内置规则：`get_stock_list_in_sector("沪市基金")` 会按 SH 的 `50/51/52/53/55/56/58` 前缀从本地缓存筛选。若仍报错，说明该名称不在内置的 17 个分类中，用 `market_sector_names()` 确认。

### Q: 市场分类板块的结果比客户端少？

这是预期行为：本地只能从**已缓存**的标的里筛，而 QMT 客户端显示的是全市场清单。补齐行情缓存后重新调用即可，数量会自动增加。

### Q: 文件读取报 DataNotAvailableError（stage=parse / stage=io）？

- `stage=parse`：文件不足一条完整记录，通常是客户端正在写入或下载中断
- `stage=io`：文件被客户端独占写或权限不足，可在客户端退出后重试

---

## 相关文档

- [qmt_provider.md](./qmt_provider.md) — QmtProvider（在线 HTTP 接口）
- [xtdata_provider.md](./xtdata_provider.md) — XtDataProvider（输出对齐基准）
- [base.py](../src/ml4t/data/providers/base.py) — BaseProvider 基类
- [qmt_datadir.py](../src/ml4t/data/providers/qmt_datadir.py) — 本地缓存解析层
