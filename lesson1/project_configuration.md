# 项目配置文档

## 1. 项目依赖 (requirements.txt)

```txt
# 核心分词库
jieba==0.42.1

# 数值计算和科学计算
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.3.0

# 数据处理
pandas==2.0.3

# 网络爬虫
requests==2.31.0
beautifulsoup4==4.12.2
lxml==4.9.3

# 数据可视化
matplotlib==3.7.1
seaborn==0.12.2

# 开发和测试工具
jupyter==1.0.0
pytest==7.4.0
black==23.7.0
flake8==6.0.0

# 其他实用工具
tqdm==4.65.0
python-dateutil==2.8.2
psutil==5.9.5
```

## 2. 项目结构配置

```
chinese-news-search/
├── README.md
├── requirements.txt
├── config/
│   ├── settings.py
│   ├── stopwords.txt
│   └── news_dict.txt
├── data/
│   ├── raw/              # 原始爬取数据
│   ├── processed/        # 处理后的数据
│   └── analysis/         # 分析结果和图表
├── src/
│   ├── __init__.py
│   ├── crawler/
│   │   ├── __init__.py
│   │   └── news_spider.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   └── analyzer.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── tfidf.py
│   │   ├── similarity.py
│   │   └── search_engine.py
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py
├── tests/
│   ├── test_crawler.py
│   ├── test_tokenizer.py
│   ├── test_tfidf.py
│   └── test_search.py
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_analysis.ipynb
│   ├── 03_segmentation_analysis.ipynb
│   ├── 04_retrieval_evaluation.ipynb
│   └── 05_system_demo.ipynb
├── main.py
└── demo.py
```

## 3. 系统配置 (config/settings.py)

```python
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录配置
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANALYSIS_DIR = DATA_DIR / "analysis"

# 创建必要的目录
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, ANALYSIS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 爬虫配置
CRAWLER_CONFIG = {
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    },
    "request_delay": 1,  # 请求间隔(秒)
    "timeout": 10,       # 请求超时(秒)
    "max_retries": 3,    # 最大重试次数
    "max_articles": 1000 # 最大爬取文章数
}

# 新闻源配置
NEWS_SOURCES = {
    "sina": {
        "base_url": "https://news.sina.com.cn",
        "categories": ["domestic", "international", "finance", "sports", "tech"],
        "enabled": True
    },
    "netease": {
        "base_url": "https://news.163.com",
        "categories": ["domestic", "world", "finance", "sports", "tech"],
        "enabled": False  # 可选启用
    }
}

# 分词配置
TOKENIZATION_CONFIG = {
    "enable_parallel": True,
    "parallel_processes": 4,
    "min_word_length": 2,
    "enable_pos_tagging": True,
    "custom_dict_path": PROJECT_ROOT / "config" / "news_dict.txt",
    "stopwords_path": PROJECT_ROOT / "config" / "stopwords.txt"
}

# TF-IDF配置
TFIDF_CONFIG = {
    "min_df": 2,          # 最小文档频率
    "max_df": 0.95,       # 最大文档频率比例
    "use_log_tf": True,   # 使用对数TF
    "normalize": True     # L2归一化
}

# 检索配置
RETRIEVAL_CONFIG = {
    "default_top_k": 10,
    "similarity_threshold": 0.01,
    "max_query_length": 100,
    "enable_query_expansion": False
}

# 评估配置
EVALUATION_CONFIG = {
    "test_query_count": 20,
    "evaluation_metrics": ["precision", "recall", "f1", "map"],
    "random_seed": 42
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": PROJECT_ROOT / "logs" / "system.log"
}
```

## 4. 停用词配置 (config/stopwords.txt)

```txt
# 基础停用词
的
了
在
是
我
有
和
就
不
人
都
一
一个
上
也
很
到
说
要
去
你
会
着
没有
看
好
自己
这
那
什么
时候
如果
但是
因为
所以
而且
然后
这样
那样
可以
应该
需要
能够
已经
还是
或者
以及
对于
关于
通过
根据
按照
由于
为了
虽然
尽管
除了
另外
同时
首先
其次
最后
总之
因此
所以
然而
不过
此外
而且
并且
或者
以及

# 时间词
今天
昨天
明天
现在
以前
以后
最近
目前
当前
过去
未来
今年
去年
明年
本月
上月
下月

# 数量词
一些
很多
少数
大量
全部
部分
大部分
小部分
许多
几个
一点
一些

# 标点符号
，
。
！
？
；
：
""
''
（
）
【
】
《
》
、
…
—
·
```

## 5. 新闻词典配置 (config/news_dict.txt)

```txt
# 新闻专业术语
人工智能 5 n
机器学习 5 n
深度学习 5 n
神经网络 5 n
大数据 5 n
云计算 5 n
物联网 5 n
区块链 5 n
5G网络 5 n
智能制造 5 n

# 疫情相关
新冠肺炎 5 n
新冠病毒 5 n
疫情防控 5 n
核酸检测 5 n
疫苗接种 5 n
隔离管控 5 n
健康码 5 n
方舱医院 5 n

# 环保相关
碳达峰 5 n
碳中和 5 n
碳排放 5 n
新能源 5 n
可再生能源 5 n
节能减排 5 n
绿色发展 5 n
生态环境 5 n

# 经济相关
数字经济 5 n
共享经济 5 n
平台经济 5 n
供给侧改革 5 n
高质量发展 5 n
双循环 5 n
自贸区 5 n
营商环境 5 n

# 政治相关
全面小康 5 n
脱贫攻坚 5 n
乡村振兴 5 n
共同富裕 5 n
治理体系 5 n
治理能力 5 n
法治建设 5 n

# 国际关系
一带一路 5 n
人类命运共同体 5 n
多边主义 5 n
全球化 5 n
贸易保护主义 5 n
地缘政治 5 n
```

## 6. 主程序配置 (main.py)

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from config.settings import *
from src.crawler.news_spider import NewsDataCollector
from src.preprocessing.tokenizer import OptimizedChineseTokenizer
from src.preprocessing.analyzer import RawDataAnalyzer, TokenizationAnalyzer
from src.retrieval.search_engine import ChineseNewsSearchSystem
from src.evaluation.metrics import RetrievalEvaluator

def setup_logging():
    """设置日志"""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG["level"]),
        format=LOGGING_CONFIG["format"],
        handlers=[
            logging.FileHandler(LOGGING_CONFIG["file_path"], encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """主程序入口"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== 中文新闻稀疏检索系统启动 ===")
    
    # 1. 数据收集
    logger.info("开始数据收集...")
    collector = NewsDataCollector()
    news_data = collector.collect_news(max_articles=CRAWLER_CONFIG["max_articles"])
    
    # 2. 数据分析
    logger.info("开始数据分析...")
    data_analyzer = RawDataAnalyzer(news_data)
    data_stats = data_analyzer.comprehensive_analysis()
    
    # 3. 分词分析
    logger.info("开始分词分析...")
    tokenizer = OptimizedChineseTokenizer()
    token_analyzer = TokenizationAnalyzer(tokenizer)
    token_stats = token_analyzer.analyze_tokenization_results(news_data)
    
    # 4. 构建检索系统
    logger.info("构建检索系统...")
    search_system = ChineseNewsSearchSystem()
    search_system.index_documents(news_data)
    
    # 5. 系统评估
    logger.info("开始系统评估...")
    evaluator = RetrievalEvaluator(search_system)
    test_queries = evaluator.create_test_queries()
    eval_results = evaluator.evaluate_retrieval_quality(test_queries)
    
    # 6. 保存结果
    logger.info("保存分析结果...")
    
    logger.info("=== 系统构建完成 ===")
    
    return search_system, data_stats, token_stats, eval_results

if __name__ == "__main__":
    main()
```

## 7. 演示程序配置 (demo.py)

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from main import main

def interactive_demo():
    """交互式演示"""
    print("正在初始化系统...")
    search_system, data_stats, token_stats, eval_results = main()
    
    print("\n=== 中文新闻检索系统演示 ===")
    print("输入查询关键词，输入 'quit' 退出")
    print("输入 'stats' 查看系统统计信息")
    print("输入 'help' 查看帮助信息")
    
    while True:
        query = input("\n请输入查询: ").strip()
        
        if query.lower() == 'quit':
            print("感谢使用，再见！")
            break
        elif query.lower() == 'stats':
            print("\n=== 系统统计信息 ===")
            stats = search_system.get_system_stats()
            for key, value in stats.items():
                print(f"{key}: {value}")
        elif query.lower() == 'help':
            print("\n=== 帮助信息 ===")
            print("1. 直接输入关键词进行检索")
            print("2. 支持多词查询，如：人工智能 机器学习")
            print("3. 支持短语查询，如：北京冬奥会")
            print("4. 输入 'stats' 查看系统统计")
            print("5. 输入 'quit' 退出系统")
        elif not query:
            continue
        else:
            # 执行检索
            try:
                results, search_time = search_system.search(query, top_k=5)
                
                print(f"\n查询: '{query}'")
                print(f"检索时间: {search_time:.4f}秒")
                print(f"找到 {len(results)} 条相关结果:\n")
                
                for i, result in enumerate(results, 1):
                    doc = result['document']
                    score = result['similarity_score']
                    
                    print(f"{i}. 相似度: {score:.4f}")
                    print(f"   标题: {doc['title']}")
                    print(f"   摘要: {doc['content'][:100]}...")
                    if 'category' in doc:
                        print(f"   分类: {doc['category']}")
                    print("-" * 60)
                
                if not results:
                    print("未找到相关结果，请尝试其他关键词。")
                    
            except Exception as e:
                print(f"检索出错: {e}")

if __name__ == "__main__":
    interactive_demo()
```

## 8. 测试配置

### 单元测试示例 (tests/test_tokenizer.py)

```python
import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.tokenizer import OptimizedChineseTokenizer

class TestChineseTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = OptimizedChineseTokenizer()
    
    def test_basic_tokenization(self):
        """测试基础分词功能"""
        text = "人工智能技术在新闻领域的应用越来越广泛"
        tokens = self.tokenizer.tokenize_document(text, keep_pos=False)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertIn("人工智能", tokens)
        self.assertIn("技术", tokens)
    
    def test_pos_tagging(self):
        """测试词性标注"""
        text = "北京是中国的首都"
        tokens = self.tokenizer.tokenize_document(text, keep_pos=True)
        
        self.assertIsInstance(tokens, list)
        self.assertIsInstance(tokens[0], tuple)
        self.assertEqual(len(tokens[0]), 2)  # (word, pos)
    
    def test_stopword_filtering(self):
        """测试停用词过滤"""
        text = "这是一个测试文本"
        tokens = self.tokenizer.tokenize_document(text, keep_pos=False)
        
        # 停用词应该被过滤掉
        self.assertNotIn("这", tokens)
        self.assertNotIn("是", tokens)
        self.assertNotIn("一个", tokens)

if __name__ == '__main__':
    unittest.main()
```

## 9. Jupyter Notebook配置

### 数据分析笔记本示例 (notebooks/02_data_analysis.ipynb)

```python
# Cell 1: 导入库和配置
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent / "src"))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.analyzer import RawDataAnalyzer, DataVisualizer

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Cell 2: 加载数据
data_path = Path.cwd().parent / "data" / "raw"
# 加载爬取的新闻数据
# news_data = load_news_data(data_path)

# Cell 3: 数据统计分析
# analyzer = RawDataAnalyzer(news_data)
# stats = analyzer.comprehensive_analysis()

# Cell 4: 可视化
# visualizer = DataVisualizer(stats)
# visualizer.create_all_visualizations()
```

## 10. 部署配置

### Docker配置 (可选)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# 设置环境变量
ENV PYTHONPATH=/app/src

# 暴露端口（如果需要Web界面）
EXPOSE 8000

# 启动命令
CMD ["python", "demo.py"]
```

这个配置文档提供了完整的项目设置，包括依赖管理、目录结构、配置文件、主程序和测试框架。所有配置都针对中文新闻稀疏检索系统的需求进行了优化。