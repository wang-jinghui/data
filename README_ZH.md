# ml4t-data

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/ml4t-data)](https://pypi.org/project/ml4t-data/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

面向量化研究工作流的统一市场数据获取与存储库。

## ML4T 库生态系统的一部分

本库是支持 [Machine Learning for Trading](https://www.ml4trading.io) 一书中所述"机器学习交易"工作流的六个相互关联的库之一：

![ML4T 库生态系统](docs/images/ml4t_ecosystem_workflow_color.png)

它们共同覆盖了数据基础设施、特征工程、建模、信号评估、策略回测和实盘部署等完整环节。

## 本库功能

量化研究需要从多个来源一致且可复现地获取市场数据。ml4t-data 提供：

- 以 `DataManager` 作为统一接口：在所有数据源之间实现获取、存储、更新和查询
- 20+ 个数据源适配器，覆盖股票、加密货币、期货、外汇、宏观经济、预测市场和因子数据
- 自动以 Hive 分区的 Parquet 格式存储数据，并跟踪元数据
- 通过 CLI 支持增量更新、缺口检测和回填
- 内置数据校验（OHLC 不变量检查、去重、异常检测）
- 期货模块，支持 CME/ICE 批量下载与连续合约构建
- COT 模块，支持获取 CFTC 交易者持仓每周报告
- 弹性机制：频率限制、指数退避重试、缺口检测

本库旨在支持持续的研究工作流，而非一次性下载。数据存储在本地，跟踪其时效性，并可借助 DuckDB 或 Polars 等工具进行查询。

![ml4t-data 架构](docs/images/ml4t_data_architecture_print.jpeg)

## 安装

```bash
pip install ml4t-data
```

## 快速开始

### DataManager（统一接口）

```python
from ml4t.data import DataManager

dm = DataManager()

# 获取并存储数据
dm.fetch("AAPL", "2020-01-01", "2024-12-31", provider="yahoo")

# 从本地存储加载
data = dm.load("AAPL", "2020-01-01", "2024-12-31")

# 批量加载多个标的
prices = dm.batch_load(["AAPL", "MSFT", "GOOGL"], "2020-01-01", "2024-12-31")

# 增量更新
dm.update("AAPL")

# 列出已存储的标的
symbols = dm.list_symbols()
metadata = dm.get_metadata("AAPL")
```

### 直接访问数据源

所有数据源适配器实现统一的接口：

```python
from ml4t.data.providers import YahooFinanceProvider, CoinGeckoProvider, FREDProvider

# 股票数据
provider = YahooFinanceProvider()
data = provider.fetch_ohlcv("AAPL", "2020-01-01", "2024-12-31")

# 加密货币
crypto = CoinGeckoProvider().fetch_ohlcv("bitcoin", "2024-01-01", "2024-12-31")

# 宏观经济数据
fred = FREDProvider().fetch_series("GDP", "2020-01-01", "2024-12-31")

# A 股数据（BaoStock）
from ml4t.data.providers.baostock_provider import BaoStockProvider
bs = BaoStockProvider()
data = bs.fetch_ohlcv("sh.600000", "2024-01-01", "2024-12-31")

# A 股 5 分钟线
minute_data = bs.fetch_ohlcv("sz.000001", "2024-07-01", "2024-12-31", frequency="5m")

# 批量获取多只 A 股
batch = bs.fetch_batch_ohlcv(["sh.600000", "sz.000001", "sh.601398"], "2024-01-01", "2024-12-31")
```

## 数据源

### 无需 API 密钥

| 数据源 | 覆盖范围 |
|--------|----------|
| Yahoo Finance | 美国/全球股票、ETF、加密货币、外汇 |
| CoinGecko | 10,000+ 种加密货币 |
| FRED | 850,000 条经济序列 |
| FXMacroData | 外汇宏观发布、经济日历、COT、大宗商品、市场情绪 |
| Fama-French | 学术因子数据 |
| AQR | 研究因子（QMJ、BAB、HML Devil、VME 等） |
| Wiki Prices | 历史美股数据（1962-2018，静态快照） |
| Kalshi | 预测市场合约 |
| Polymarket | 预测市场历史/订单簿快照 |
| Binance Public | 加密货币批量数据下载 |
| NASDAQ ITCH Sample | Tick 级别示例数据 |
| BaoStock | A 股日线/分钟线（1990-12-19 至今） |

### 受限访问

| 数据源 | 覆盖范围 |
|--------|----------|
| XtData (迅投) | A 股、ETF、指数（需 QMT 行情客户端） |

### 需要认证或有调用量限制的 API

| 数据源 | 覆盖范围 |
|--------|----------|
| Alpaca | 美股 + 加密货币（免费 IEX 行情） |
| EODHD | 60+ 个全球交易所 |
| Tiingo | 注重数据质量的美股 |
| Twelve Data | 多资产覆盖 |
| Databento | CME/ICE 期货；OPRA 期权 |
| Massive | 美股、期权、期货、外汇、加密货币 |
| Finnhub | 70+ 个全球交易所 |
| Binance | 加密货币交易所数据 |
| OKX | 加密货币永续合约与资金费率 |
| CryptoCompare | 加密货币市场数据 |
| OANDA | 外汇经纪商数据 |

## 专用模块

### 期货

支持 CME/ICE 产品的批量下载与连续合约构建：

```python
from ml4t.data.futures import FuturesDownloader, ContinuousContractBuilder

# 通过 Databento 批量下载（父级符号体系）
downloader = FuturesDownloader(config)
downloader.download()  # 下载 ES, NQ, CL, GC 等合约

# 以可配置的移仓逻辑构建连续合约
builder = ContinuousContractBuilder()
continuous = builder.build(contracts_df, roll_method="volume")
```

面向本书（ML4T）的简化接口，支持数据概览分析：

```python
from ml4t.data.futures import FuturesDataManager

fm = FuturesDataManager.from_config("config.yaml")
fm.download_all()
data = fm.load_ohlcv("ES")
profile = fm.generate_profile("ES")
```

### COT（交易者持仓报告）

获取 CFTC 期货市场的周度持仓数据：

```python
from ml4t.data.cot import COTFetcher, create_cot_features, combine_cot_ohlcv_pit

fetcher = COTFetcher(config)
cot_data = fetcher.fetch_product("ES", start_year=2015, end_year=2024)

# 与 OHLCV 进行时间点组合（无未来信息泄露）
combined = combine_cot_ohlcv_pit(cot_data, ohlcv_data)

# 从 COT 数据生成特征
features = create_cot_features(cot_data)
```

### 书籍数据管理器

为 ML4T 书籍工作流提供的简化接口：

```python
from ml4t.data.etfs import ETFDataManager
from ml4t.data.crypto import CryptoDataManager

# 通过 Yahoo Finance 获取 50 只多样化 ETF
etf_dm = ETFDataManager.from_config("config.yaml")
etf_dm.download_all()
aapl = etf_dm.load_ohlcv("AAPL")

# 通过 Binance Public 获取加密货币指数
crypto_dm = CryptoDataManager.from_config("config.yaml")
crypto_dm.download_premium_index()
```

## 命令行工具（CLI）

```bash
# 获取指定标的
ml4t-data fetch -s AAPL -s MSFT -s GOOGL --provider yahoo --start 2020-01-01

# 增量更新
ml4t-data update --symbol AAPL

# 校验数据质量
ml4t-data validate --symbol AAPL --anomalies

# 查看存储状态
ml4t-data status --detailed

# 列出可用数据
ml4t-data list-data

# 导出为 CSV/JSON/Excel
ml4t-data export --symbol AAPL --format-type csv --output aapl.csv

# 获取标的信息
ml4t-data info --symbol AAPL
```

基于配置的批量更新：

```yaml
storage:
  path: ~/data/market

datasets:
  sp500_daily:
    provider: yahoo
    symbols_file: symbols/sp500.txt
    frequency: daily
    start_date: 2015-01-01

  crypto:
    provider: coingecko
    symbols: [bitcoin, ethereum, solana]
    frequency: daily
    start_date: 2020-01-01
```

## 数据存储

### 配置存储位置

DataManager 的存储功能由 `storage` 参数控制，有三种方式指定存储路径：

**方式一：编程方式指定（推荐快速上手）**

```python
from ml4t.data import DataManager
from ml4t.data.storage.hive import HiveStorage
from ml4t.data.storage.backend import StorageConfig

storage = HiveStorage(StorageConfig(base_path="./market_data"))
dm = DataManager(storage=storage)

# 获取并保存到 ./market_data/<provider>/<frequency>/symbol=<name>/data.parquet
dm.load("AAPL", "2020-01-01", "2024-12-31", provider="yahoo")
```

**方式二：配置文件（YAML）指定**

```yaml
# config.yaml
storage:
  path: ~/data/market
```

```python
dm = DataManager(config_path="config.yaml")
```

**方式三：环境变量指定**

设置 `ML4T_DATA_ROOT` 环境变量，默认使用当前工作目录下的 `data/` 子目录。

### 存储格式

数据以 Hive 分区的 Parquet 格式存储：

```
~/data/market/
├── yahoo/daily/symbol=AAPL/data.parquet
├── yahoo/daily/symbol=MSFT/data.parquet
└── coingecko/daily/symbol=bitcoin/data.parquet
```

可使用 DuckDB 或 Polars 进行查询：

```python
import duckdb

result = duckdb.execute("""
    SELECT * FROM read_parquet('~/data/market/yahoo/daily/**/*.parquet')
    WHERE symbol IN ('AAPL', 'MSFT')
    AND date >= '2024-01-01'
""").pl()
```

### 常用存储操作

```python
# 获取并保存
dm.load("AAPL", "2020-01-01", "2024-12-31", provider="yahoo")

# 导入已有 DataFrame
from ml4t.data.providers.baostock_provider import BaoStockProvider
bs = BaoStockProvider()
data = bs.fetch_ohlcv("sh.600000", "2024-01-01", "2024-12-31")
dm.import_data(data, "sh.600000", provider="baostock")

# 增量更新
dm.update("sh.600000", provider="baostock")

# 列出已存储的标的
dm.list_symbols()

# 查看元数据（时间范围、行数等）
dm.get_metadata("sh.600000")

# 批量更新所有已存储数据
dm.update_all(provider="baostock")
```

## 数据校验

```python
from ml4t.data.validation import OHLCVValidator, ValidationReport

validator = OHLCVValidator()
report = validator.validate(data)
# 检查：high >= low, high >= open/close, low <= open/close
# 检测：重复数据、缺口、异常值
```

异常检测：

```python
from ml4t.data.anomaly import AnomalyManager, ReturnOutlierDetector, VolumeSpikeDetector

manager = AnomalyManager([
    ReturnOutlierDetector(),
    VolumeSpikeDetector(),
])
report = manager.detect(data)
```

## 文档

- [入门指南](docs/user-guide/getting-started.md) — 快速上手
- [配置说明](docs/user-guide/configuration.md) — YAML 配置参考
- [存储](docs/user-guide/storage.md) — Hive 分区与存储后端
- [增量更新](docs/user-guide/incremental-updates.md) — 更新策略与缺口检测
- [数据质量](docs/user-guide/data-quality.md) — 校验与异常检测
- [CLI 参考](docs/user-guide/cli-reference.md) — 命令行接口
- [数据源选择指南](docs/provider-selection-guide.md) — 如何选择数据源
- [创建数据源](docs/creating_a_provider.md) — 扩展新的数据源

## 技术特点

- **基于 Polars**：全程使用原生 Polars DataFrame，性能优异
- **一致的 Schema**：所有数据源返回统一的列结构
- **异步支持**：异步数据源与批量操作，支持并行下载
- **元数据跟踪**：最后更新时间、行数、日期范围
- **弹性机制**：频率限制、指数退避重试、缺口检测
- **多后端存储**：支持本地文件系统、S3 和内存存储
- **类型安全**：完整的类型注解

## 相关库

- **ml4t-engineer**：特征工程与技术指标
- **ml4t-diagnostic**：信号评估与统计校验
- **ml4t-backtest**：事件驱动的回测框架
- **ml4t-live**：实盘交易与券商集成

## 开发

```bash
git clone https://github.com/ml4t/data.git ml4t-data
cd ml4t-data
uv sync
uv run pytest tests/ -q
uv run ty check
```

## 许可证

MIT 许可证 — 详见 [LICENSE](LICENSE)。
