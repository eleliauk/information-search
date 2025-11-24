"""
测试核心数据模型
"""
import pytest
from datetime import datetime
from src.models import Document, Posting, IndexStatistics, BooleanQuery


def test_document_creation():
    """测试文档创建"""
    doc = Document(
        doc_id=1,
        content="测试文档内容",
        metadata={"title": "测试"},
        token_count=4
    )
    assert doc.doc_id == 1
    assert doc.content == "测试文档内容"
    assert doc.metadata["title"] == "测试"
    assert doc.token_count == 4
    assert isinstance(doc.created_at, datetime)


def test_posting_creation():
    """测试倒排列表项创建"""
    posting = Posting(
        doc_id=1,
        positions=[0, 5, 10],
        term_freq=3
    )
    assert posting.doc_id == 1
    assert posting.positions == [0, 5, 10]
    assert posting.term_freq == 3


def test_index_statistics_creation():
    """测试索引统计信息创建"""
    stats = IndexStatistics(
        document_count=10,
        vocabulary_size=100,
        total_tokens=500,
        avg_doc_length=50.0,
        term_frequencies={"测试": 5}
    )
    assert stats.document_count == 10
    assert stats.vocabulary_size == 100
    assert stats.total_tokens == 500
    assert stats.avg_doc_length == 50.0
    assert stats.term_frequencies["测试"] == 5


def test_boolean_query_creation():
    """测试布尔查询创建"""
    query = BooleanQuery(
        operator="AND",
        terms=["测试", "文档"],
        exclude_terms=[]
    )
    assert query.operator == "AND"
    assert query.terms == ["测试", "文档"]
    assert query.exclude_terms == []
