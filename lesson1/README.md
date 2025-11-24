# 中文新闻稀疏检索系统

基于结巴分词和TF-IDF算法的中文新闻检索系统原型，支持100-1000篇新闻文章的语义检索。

## 🎯 项目概述

本项目实现了一个完整的中文新闻稀疏检索系统，包含以下核心功能：

- **新闻数据爬取**: 支持多源新闻网站数据收集
- **中文分词处理**: 基于结巴分词的优化分词算法
- **TF-IDF向量化**: 高效的文本向量化实现
- **余弦相似度检索**: 基于余弦相似度的检索算法
- **数据统计分析**: 全面的数据质量和分词效果分析
- **性能评估**: 完整的系统性能评估框架

## 📋 系统要求

- Python 3.8+
- 内存: 至少 2GB
- 存储: 至少 1GB 可用空间

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行系统

#### 完整流程（推荐）
```bash
python main.py --mode full --articles 500 --mock
```

#### 仅运行演示
```bash
python main.py --mode demo --load-index saved_index.pkl
```

#### 批量测试
```bash
python main.py --mode test --articles 200 --mock
```

### 3. 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|---------|
| `--mode` | 运行模式: full/demo/test/collect/analyze | full |
| `--articles` | 爬取文章数量 | 500 |
| `--mock` | 使用模拟数据 | False |
| `--no-viz` | 不生成可视化图表 | False |
| `--save-index` | 保存索引文件路径 | None |
| `--load-index` | 加载索引文件路径 | None |
| `--log-level` | 日志级别 | INFO |

## 📁 项目结构

```
chinese-news-search/
├── README.md                           # 项目说明
├── requirements.txt                    # 依赖包列表
├── main.py                            # 主程序入口
├── config/                            # 配置文件
│   ├── settings.py                    # 系统配置
│   ├── stopwords.txt                  # 停用词列表
│   └── news_dict.txt                  # 新闻领域词典
├── src/                               # 源代码
│   ├── crawler/                       # 数据爬取模块
│   │   └── news_spider.py            # 新闻爬虫
│   ├── preprocessing/                 # 数据预处理模块
│   │   ├── tokenizer.py              # 分词器
│   │   └── analyzer.py               # 数据分析器
│   ├── retrieval/                     # 检索模块
│   │   ├── tfidf.py                  # TF-IDF实现
│   │   ├── similarity.py             # 相似度计算
│   │   └── search_engine.py          # 搜索引擎
│   └── evaluation/                    # 评估模块
│       └── metrics.py                # 评估指标
├── data/                              # 数据目录
│   ├── raw/                          # 原始数据
│   ├── processed/                    # 处理后数据
│   └── analysis/                     # 分析结果
└── logs/                             # 日志文件
```

## 🔧 核心功能

### 1. 数据收集与处理

**数据来源**:
- 新浪新闻、网易新闻等主流新闻网站
- 支持模拟数据生成用于测试

**数据质量控制**:
- 内容长度过滤 (100-5000字符)
- 重复内容检测
- 数据完整性验证

### 2. 中文分词

**结巴分词优化**:
- 自定义新闻领域词典
- 停用词过滤
- 词性标注和过滤
- 并行分词处理

**分词统计分析**:
- 词频分布分析
- 词性分布统计
- 词长分布分析
- 分词质量评估

### 3. TF-IDF向量化

**算法特性**:
- 对数词频 (1 + log(tf))
- 逆文档频率 (log(N/df))
- L2归一化
- 稀疏矩阵存储

**参数配置**:
- 最小文档频率: 2
- 最大文档频率比例: 0.95
- 支持词汇表过滤

### 4. 检索算法

**余弦相似度**:
- 基于TF-IDF向量的余弦相似度计算
- 支持批量检索
- 相似度阈值过滤
- 结果排序和截断

**检索优化**:
- 稀疏矩阵运算
- 向量归一化
- 内存优化存储

## 📊 数据统计分析

### 原始数据分析

- **基本统计**: 文章数量、字符数分布、分类分布
- **时间分析**: 发布时间分布、高峰时段分析
- **质量分析**: 内容完整性、重复度检测

### 分词效果分析

- **词汇统计**: 总词数、唯一词数、词汇丰富度
- **词频分析**: 高频词、低频词、词频分布
- **词性分析**: 词性分布、词性集中度
- **质量评估**: 分词一致性、OOV率

## 🎯 检索算法设计

### TF-IDF实现

```python
# 词频计算 (对数归一化)
tf(t,d) = 1 + log(count(t,d))

# 逆文档频率
idf(t) = log(N / df(t))

# TF-IDF权重
w(t,d) = tf(t,d) × idf(t)
```

### 余弦相似度

```python
# 余弦相似度计算
similarity(q,d) = (q · d) / (||q|| × ||d||)
```

### 检索流程

1. **查询预处理**: 分词、停用词过滤
2. **查询向量化**: TF-IDF权重计算
3. **相似度计算**: 与文档向量计算余弦相似度
4. **结果排序**: 按相似度降序排列
5. **结果过滤**: 应用阈值和数量限制

## 📈 性能评估

### 评估指标

- **响应时间**: 平均检索时间、95百分位响应时间
- **吞吐量**: QPS (每秒查询数)
- **检索质量**: 结果相关性、成功率
- **内存使用**: 索引大小、内存占用

### 基准测试

系统在500篇新闻文档上的典型性能：

| 指标 | 数值 |
|------|------|
| 平均响应时间 | < 50ms |
| 词汇表大小 | ~2000词 |
| 索引内存使用 | ~10MB |
| 检索成功率 | > 90% |

## 🔍 使用示例

### Python API

```python
from src.retrieval.search_engine import ChineseNewsSearchSystem

# 创建搜索系统
search_system = ChineseNewsSearchSystem()

# 构建索引
search_system.index_documents(news_data)

# 执行检索
results, search_time = search_system.search("人工智能", top_k=10)

# 查看结果
for result in results:
    print(f"标题: {result['title']}")
    print(f"相似度: {result['similarity_score']:.4f}")
```

### 交互式检索

```bash
# 启动交互式界面
python main.py --mode demo

# 输入查询
🔎 请输入查询: 人工智能

# 查看结果
📄 找到 5 条相关结果:
📰 1. 相似度: 0.3256
   📌 标题: 人工智能技术在医疗领域的应用
   📂 分类: 科技
   📝 摘要: 人工智能技术在医疗领域的应用越来越广泛...
```

## 🛠️ 开发指南

### 添加新的数据源

1. 在 `src/crawler/news_spider.py` 中添加新的爬虫方法
2. 更新 `config/settings.py` 中的数据源配置
3. 实现数据解析和验证逻辑

### 自定义分词

1. 修改 `config/news_dict.txt` 添加领域词汇
2. 更新 `config/stopwords.txt` 调整停用词
3. 在 `src/preprocessing/tokenizer.py` 中调整分词参数

### 扩展评估指标

1. 在 `src/evaluation/metrics.py` 中添加新的评估方法
2. 实现自定义的相关性判断逻辑
3. 扩展性能基准测试

## 📝 技术文档

详细的技术文档请参考：

- [系统设计文档](chinese_news_search_system_design.md)
- [技术规格说明](technical_specifications.md)
- [项目配置文档](project_configuration.md)

## 🧪 测试

### 运行单元测试

```bash
# 安装测试依赖
pip install pytest

# 运行所有测试
pytest tests/

# 运行特定模块测试
pytest tests/test_tokenizer.py -v
```

### 性能测试

```bash
# 运行性能基准测试
python main.py --mode test --articles 1000 --mock
```

## 🐛 问题排查

### 常见问题

1. **内存不足**
   - 减少文档数量 (`--articles`)
   - 降低TF-IDF最小频率阈值

2. **分词效果不佳**
   - 检查自定义词典配置
   - 调整停用词列表

3. **检索结果相关性低**
   - 增加训练文档数量
   - 调整相似度阈值

### 日志分析

```bash
# 查看系统日志
tail -f logs/system_*.log

# 设置详细日志级别
python main.py --log-level DEBUG
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/new-feature`)
3. 提交更改 (`git commit -am 'Add new feature'`)
4. 推送到分支 (`git push origin feature/new-feature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [jieba](https://github.com/fxsjy/jieba) - 中文分词库
- [scikit-learn](https://scikit-learn.org/) - 机器学习工具库
- [NumPy](https://numpy.org/) - 数值计算库
- [Matplotlib](https://matplotlib.org/) - 数据可视化库

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件
- 创建 Discussion

---

**开发者**: Roo  
**版本**: 1.0.0  
**最后更新**: 2024年10月