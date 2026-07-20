# 适配计划：自定义市场数据接入 ml4t-data

> **状态**：规划中
> **创建日期**：2026-07-20
> **目标**：将自有市场数据管道的日频 OHLCV 数据接入 ml4t-data 库，复用其存储、校验、增量更新管线。

---

## 一、适配方案概述

### 核心结论

**零侵入改造**。通过继承 `BaseProvider` 并注册到 `ProviderManager`，不修改 ml4t-data 任何源码。

### 架构位置

```
┌─────────────────────────────────────────────┐
│            你的自定义 Provider               │
│  MyMarketProvider(BaseProvider)             │
│                                             │
│  _fetch_and_transform_data(symbol, ...)     │
│    → 调用自有管道获取 dict[str, DataFrame]   │
│    → 取 dict[symbol] → 转标准长格式          │
│    → 返回 pl.DataFrame                       │
└──────────┬──────────────────────────────────┘
           │ 实现 OHLCVProvider 协议
           ▼
┌─────────────────────────────────────────────┐
│          ml4t-data (库，不改)                │
│                                             │
│  DataManager → FetchManager                 │
│    → ProviderManager                        │
│      → register_provider("my_market", ...)  │
│      → Provider.fetch_ohlcv(symbol, ...)    │
│        → _validate_ohlcv()                  │
│        → StorageManager                     │
│          → HiveStorage (Parquet)            │
└─────────────────────────────────────────────┘
```

---

## 二、数据格式差异

### 自有数据格式

| 维度 | 描述 |
|---|---|
| 数据结构 | `dict[str, DataFrame]`，key 为 symbol |
| DataFrame 索引 | 日期格式 `20260701`（整型或字符串） |
| DataFrame 列 | OHLCV 核心列 + 额外业务列（具体待确认） |
| 数据频率 | 日频 |
| 获取方式 | 批量接口，一次返回多个 symbol |

### ml4t-data 期望格式

| 维度 | 描述 |
|---|---|
| 数据结构 | 单个 `pl.DataFrame` |
| 必需列 | `timestamp` (Datetime), `symbol` (Utf8), `open`, `high`, `low`, `close`, `volume` (Float64) |
| 可选列 | 额外列保留，不做约束 |
| 数据频率 | 由调用方指定（daily/hourly/minute 等） |

### 转换规则

```
dict[str, DataFrame]          →  按 symbol 参数取对应 DataFrame
index (20260701)              →  pl.col.cast(Utf8) → str.strptime("%Y%m%d") → cast(Datetime)
DataFrame 自身列              →  保留（额外列）
symbol (dict 的 key)          →  写入 pl.lit(symbol).alias("symbol") 列
拼上 timestamp + symbol 列    →  标准长格式 DataFrame
```

---

## 三、改造内容清单

### 3.1 需要新建的文件

| 文件 | 位置 | 说明 |
|---|---|---|
| `my_market_provider.py` | 外部项目内（或 `src/ml4t/data/providers/`） | 自定义 Provider 类 |
| `test_my_market_provider.py` | 测试目录 | Provider 单元测试 |

### 3.2 需要提供的信息

以下信息在实现 Provider 之前必须确认：

- [ ] **列名映射表**：自有 DataFrame 的列名 → 标准 OHLCV 列名
  - 例：`"o" → "open"`, `"h" → "high"`, ...
  - 如果列名已经是 `open/high/low/close/volume`，无需映射
- [ ] **索引列名**：`DataFrame.reset_index()` 后日期索引的列名是什么
- [ ] **获取接口**：自有管道的调用方式（函数签名、返回类型、认证方式）
- [ ] **额外列清单**：除 OHLCV 外有哪些列，是否需要保留入库
- [ ] **symbol 规范**：自有 symbol 的格式（大小写、分隔符），是否需要统一转大写

### 3.3 不需要修改的内容

| 模块 | 原因 |
|---|---|
| `providers/base.py` | 继承即可，模板方法已完整 |
| `providers/mixins/validation.py` | 只要求 6 列存在，额外列保留 |
| `core/schemas.py` | 校验不拒绝额外列 |
| `storage/backend.py` | 按 `timestamp` 列分区，与额外列无关 |
| `managers/fetch_manager.py` | 单 symbol 调用，天然隔离 |
| `managers/provider_manager.py` | `register_provider()` 支持动态注册 |

---

## 四、技术实现要点

### 4.1 Provider 类骨架

```python
"""自定义市场 OHLCV 数据 Provider。

继承 BaseProvider，适配自有数据管道的 dict[str, DataFrame] 格式，
转换后输出标准长格式 Polars DataFrame。

数据流：
    自有管道 → dict[symbol] → DataFrame(index=20260701, cols=[...])
        → reset_index + 日期解析 + symbol 列 + 列名映射
        → pl.DataFrame[timestamp, symbol, open, high, low, close, volume, ...]
"""

from __future__ import annotations

import polars as pl
from ml4t.data.providers.base import BaseProvider


class MyMarketProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "my_market"

    def _fetch_and_transform_data(
        self, symbol: str, start: str, end: str, frequency: str
    ) -> pl.DataFrame:
        # 第 1 步：调用自有管道
        raw_dict: dict[str, DataFrame] = self._get_data(symbol, start, end)

        # 第 2 步：取对应 symbol 的 DataFrame
        df = raw_dict[symbol]

        # 第 3 步：index → timestamp 列
        df = df.reset_index()
        df = df.with_columns(
            pl.col("<index_col>").cast(pl.Utf8)
            .str.strptime(pl.Date, "%Y%m%d")
            .cast(pl.Datetime("us", "UTC"))
            .alias("timestamp")
        )

        # 第 4 步：列名映射（如需要）
        # df = df.rename({"o": "open", "h": "high", ...})

        # 第 5 步：拼上 symbol 列
        df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))

        # 第 6 步：转为 Polars（如原始是 pandas）
        if not isinstance(df, pl.DataFrame):
            df = pl.from_polars(df)

        # 第 7 步：类型转换
        for col in ["open", "high", "low", "close", "volume"]:
            df = df.with_columns(pl.col(col).cast(pl.Float64))

        return df

    # ---- 私有方法：自有管道封装 ----

    def _get_data(
        self, symbol: str, start: str, end: str
    ) -> dict:
        """调用自有数据管道，返回 dict[symbol, DataFrame]。"""
        ...
```

### 4.2 注册与使用

```python
from ml4t.data import DataManager

dm = DataManager()
dm._provider_manager.register_provider("my_market", MyMarketProvider)

# 单 symbol
df = dm.fetch("SYM_A", "2025-01-01", "2026-07-20", provider="my_market")

# 批量（由 BatchManager 并行分发到 fetch_ohlcv）
df_all = dm.batch_load(
    ["SYM_A", "SYM_B", "SYM_C"],
    "2025-01-01", "2026-07-20",
    provider="my_market",
    max_workers=4,
)

# 带存储
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.storage.backend import StorageConfig

storage = HiveStorage(StorageConfig(base_path="./market_data"))
dm = DataManager(storage=storage)
dm._provider_manager.register_provider("my_market", MyMarketProvider)

# 初始加载 + 存储
dm.load("SYM_A", "2025-01-01", "2026-07-20", provider="my_market")

# 增量更新
dm.update("SYM_A", provider="my_market")
```

### 4.3 调用链路

```
用户调用: dm.fetch("SYM_A", start, end, provider="my_market")
    │
    ▼
FetchManager.fetch("SYM_A", ...)
    │
    ▼
ProviderManager.get_provider("my_market")  → MyMarketProvider 实例
    │
    ▼
MyMarketProvider.fetch_ohlcv("SYM_A", ...)  # BaseProvider 模板方法
    │
    ├─→ _validate_inputs("SYM_A", start, end)          # 输入校验
    ├─→ _acquire_rate_limit()                           # 限流
    ├─→ _fetch_and_transform_data("SYM_A", ...)         # ★ 你的实现
    │       │
    │       ├─→ _get_data(...)                          # 调自有管道
    │       ├─→ dict["SYM_A"]                           # 取单 symbol
    │       ├─→ index → timestamp                       # 日期解析
    │       └─→ 拼列 + 转类型                            # 格式对齐
    │
    └─→ _validate_ohlcv(df)                               # OHLCV 校验 + 去重
              │
              ├─→ 检查 REQUIRED_COLUMNS 存在
              ├─→ OHLC 不变式 (high >= low 等)
              ├─→ sort("timestamp")
              └─→ unique(subset=["timestamp"])            # 单 symbol，安全
```

---

## 五、风险评估

| 风险 | 等级 | 说明 | 应对 |
|---|---|---|---|
| 去重误杀 | **无** | `fetch_ohlcv` 单 symbol 调用，不同 symbol 不会出现在同一 DataFrame | 遵循单 symbol 约定 |
| 额外列丢失 | **无** | 校验只要求 6 列存在，不删除额外列 | 无需处理 |
| 日期解析失败 | 低 | `20260701` 格式固定，`strptime("%Y%m%d")` 稳定 | 异常处理 + 日志 |
| 列名不匹配 | 低 | 取决于确认的映射表 | 明确映射后无风险 |
| 管道接口变更 | 中 | 自有管道 API 可能变动 | 封装在 `_get_data` 内，改动隔离 |

---

## 六、测试计划

| 测试项 | 说明 |
|---|---|
| 单 symbol 获取 | 验证 dict → DataFrame 转换正确 |
| 日期解析 | 边界日期（1月1日、12月31日、闰年）正确转为 timestamp |
| 空结果 | 管道返回空 dict 或空 DataFrame 时返回空 schema |
| 额外列保留 | 输出 DataFrame 包含所有原始额外列 |
| OHLCV 校验 | 通过 BaseProvider 继承的校验管线 |
| 批量获取 | 通过 batch_load 验证多 symbol 并发安全 |
| 存储写入 | 写入 Hive 分区，验证文件结构与可读性 |
| 增量更新 | 验证 update() 能正确检测 gap 并回填 |

---

## 七、修改记录

> 每次修改在此记录，保持倒序排列（最新在最上）。

### 2026-07-20 — 初始规划

| 项目 | 内容 |
|---|---|
| 修改人 | — |
| 修改内容 | 创建适配计划文档 |
| 分析结论 | 零侵入方案可行，继承 `BaseProvider` 即可 |
| 关键发现 | `fetch_ohlcv` 单 symbol 调用 → 去重逻辑不影响多 symbol 数据 |
| 待确认 | 列名映射、索引列名、管道调用接口、额外列清单、symbol 格式规范 |
| 文件变动 | 无（仅本文档） |

---

## 八、待办事项

- [ ] 确认自有 DataFrame 列名 → 标准列名映射
- [ ] 确认 `reset_index()` 后索引列名
- [ ] 确认自有管道调用方式（函数签名 / 认证 / 返回值）
- [ ] 确认额外列清单及是否入库
- [ ] 确认 symbol 格式规范
- [ ] 实现 `MyMarketProvider` 类
- [ ] 编写单元测试
- [ ] 集成验证（fetch / batch_load / load / update）
- [ ] 性能验证（批量场景）
