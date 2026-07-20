# 适配计划：xtdata 市场数据接入 ml4t-data

> **状态**：已实现（初版）
> **创建日期**：2026-07-20
> **目标**：通过 xtquant.xtdata 将 A 股 / ETF 日频 OHLCV 数据接入 ml4t-data，复用其存储、校验、增量更新管线。
>
> 详细使用文档见 [xtdata_provider.md](./xtdata_provider.md)

---

## 适配方案

**零侵入改造**。继承 `BaseProvider` 并注册到 `ProviderManager`，不修改 ml4t-data 任何源码。

### 数据格式转换

| 维度 | xtdata 返回 | ml4t-data 要求 | 转换方式 |
|---|---|---|---|
| 结构 | `dict[str, DataFrame]` | 单个 `pl.DataFrame` | 按 symbol 取 DataFrame |
| 索引 | `'YYYYMMDD'` 字符串 | `timestamp` Datetime 列 | `strptime("%Y%m%d")` |
| 列名 | `open/high/low/close/volume` | 同 | 无需映射 |
| symbol | dict 的 key | `symbol` 列 | `pl.lit(symbol.upper())` |
| 额外列 | `amount, preClose, suspendFlag` 等 | 保留 | 自动保留 |

### 不需要修改的 ml4t-data 模块

`providers/base.py`、`mixins/validation.py`、`core/schemas.py`、`storage/backend.py`、`managers/fetch_manager.py`、`managers/provider_manager.py` — 继承即可，模板方法已完整。

---

## 风险评估

| 风险 | 等级 | 说明 |
|---|---|---|
| 去重误杀 | **无** | 单 symbol 调用，天然隔离 |
| 额外列丢失 | **无** | 校验只要求 6 列存在 |
| 日期解析失败 | 低 | 格式固定，strptime 稳定 |
| 管道接口变更 | 中 | 封装在 `_fetch_and_transform_data` 内，改动隔离 |

---

## 修改记录

### 2026-07-21 — 实现 XtDataProvider 并验证通过

| 项目 | 内容 |
|---|---|
| 实现文件 | `src/ml4t/data/providers/xtdata_provider.py` |
| 确认事项 | 列名无需映射；`reset_index()` 后列名为 `index`；额外列保留 |
| 验证结果 | 单 symbol (510300.SH) OK；批量 (2 标的) OK；OHLCV 校验通过 |
| 环境要求 | Python 3.11（xtquant .pyd 最高支持 cp311） |
| 文件变动 | 新增 `xtdata_provider.py`、`docs/xtdata_provider.md` |

### 2026-07-20 — 初始规划

| 项目 | 内容 |
|---|---|
| 分析结论 | 零侵入方案可行，继承 `BaseProvider` 即可 |
| 关键发现 | `fetch_ohlcv` 单 symbol 调用 → 去重逻辑不影响多 symbol 数据 |

---

## 待办事项

- [x] 确认列名映射（无需映射）
- [x] 确认索引列名（`index`）
- [x] 确认获取接口（`xtdata.get_market_data_ex`）
- [x] 确认额外列清单
- [x] 确认 symbol 格式（`510300.SH`）
- [x] 实现 `XtDataProvider`
- [x] 冒烟测试验证
- [ ] 编写正式单元测试
- [ ] 集成验证（DataManager fetch / load / update）
- [ ] 性能验证（大量标的批量场景）
