"""
索引统计信息功能的单元测试
"""
import pytest
from src.inverted_index import InvertedIndex
from src.models import IndexStatistics


class TestStatistics:
    """统计信息测试类"""
    
    def test_get_statistics_empty_index(self):
        """测试空索引的统计信息"""
        index = InvertedIndex()
        stats = index.get_statistics()
        
        assert isinstance(stats, IndexStatistics)
        assert stats.document_count == 0
        assert stats.vocabulary_size == 0
        assert stats.total_tokens == 0
        assert stats.avg_doc_length == 0.0
        assert stats.term_frequencies == {}
    
    def test_get_statistics_single_document(self):
        """测试单个文档的统计信息"""
        index = InvertedIndex()
        tokens = ["中国", "北京", "中国", "首都"]
        index.build_index(doc_id=1, tokens=tokens)
        
        stats = index.get_statistics()
        
        # 验证文档总数
        assert stats.document_count == 1
        
        # 验证唯一词项数
        assert stats.vocabulary_size == 3  # 中国、北京、首都
        
        # 验证总词项数
        assert stats.total_tokens == 4
        
        # 验证平均文档长度
        assert stats.avg_doc_length == 4.0
        
        # 验证词项文档频率
        assert stats.term_frequencies["中国"] == 1
        assert stats.term_frequencies["北京"] == 1
        assert stats.term_frequencies["首都"] == 1
    
    def test_get_statistics_multiple_documents(self):
        """测试多个文档的统计信息"""
        index = InvertedIndex()
        
        # 文档1: 4个词项
        index.build_index(doc_id=1, tokens=["中国", "北京", "中国", "首都"])
        
        # 文档2: 2个词项
        index.build_index(doc_id=2, tokens=["中国", "上海"])
        
        # 文档3: 3个词项
        index.build_index(doc_id=3, tokens=["日本", "东京", "首都"])
        
        stats = index.get_statistics()
        
        # 验证文档总数
        assert stats.document_count == 3
        
        # 验证唯一词项数
        assert stats.vocabulary_size == 6  # 中国、北京、首都、上海、日本、东京
        
        # 验证总词项数
        assert stats.total_tokens == 9  # 4 + 2 + 3
        
        # 验证平均文档长度
        assert stats.avg_doc_length == 3.0  # 9 / 3
        
        # 验证词项文档频率
        assert stats.term_frequencies["中国"] == 2  # 出现在文档1和2
        assert stats.term_frequencies["北京"] == 1  # 只在文档1
        assert stats.term_frequencies["首都"] == 2  # 出现在文档1和3
        assert stats.term_frequencies["上海"] == 1  # 只在文档2
        assert stats.term_frequencies["日本"] == 1  # 只在文档3
        assert stats.term_frequencies["东京"] == 1  # 只在文档3
    
    def test_get_statistics_term_frequency_accuracy(self):
        """测试词项文档频率的准确性"""
        index = InvertedIndex()
        
        # 创建5个文档，"中国"出现在3个文档中
        index.build_index(doc_id=1, tokens=["中国", "北京"])
        index.build_index(doc_id=2, tokens=["中国", "上海"])
        index.build_index(doc_id=3, tokens=["日本", "东京"])
        index.build_index(doc_id=4, tokens=["中国", "广州"])
        index.build_index(doc_id=5, tokens=["韩国", "首尔"])
        
        stats = index.get_statistics()
        
        # "中国"应该出现在3个文档中
        assert stats.term_frequencies["中国"] == 3
        
        # "日本"应该只出现在1个文档中
        assert stats.term_frequencies["日本"] == 1
        
        # 验证所有词项都被统计
        assert len(stats.term_frequencies) == stats.vocabulary_size
