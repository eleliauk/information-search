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
    "max_articles": 500  # 最大爬取文章数
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