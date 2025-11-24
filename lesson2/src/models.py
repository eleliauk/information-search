"""
核心数据模型定义
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class Document:
    """文档数据模型"""
    doc_id: int
    content: str
    metadata: Dict = field(default_factory=dict)
    token_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Posting:
    """倒排列表项数据模型"""
    doc_id: int
    positions: List[int] = field(default_factory=list)
    term_freq: int = 0


@dataclass
class IndexStatistics:
    """索引统计信息数据模型"""
    document_count: int = 0
    vocabulary_size: int = 0
    total_tokens: int = 0
    avg_doc_length: float = 0.0
    term_frequencies: Dict[str, int] = field(default_factory=dict)


@dataclass
class BooleanQuery:
    """布尔查询数据模型"""
    operator: str  # 'AND', 'OR', 'NOT'
    terms: List[str] = field(default_factory=list)
    exclude_terms: List[str] = field(default_factory=list)
