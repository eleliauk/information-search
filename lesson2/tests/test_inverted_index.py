"""
倒排索引组件的单元测试
"""
import pytest
import tempfile
import os
from src.inverted_index import InvertedIndex
from src.models import Posting


class TestInvertedIndex:
    """倒排索引测试类"""
    
    def test_initialization(self):
        """测试初始化"""
        index = InvertedIndex()
        assert index.get_vocabulary_size() == 0
        assert index.get_all_terms() == []
    
    def test_build_index_single_document(self):
        """测试单个文档的索引构建"""
        index = InvertedIndex()
        tokens = ["中国", "北京", "中国", "首都"]
        
        index.build_index(doc_id=1, tokens=tokens)
        
        # 验证词汇表大小
        assert index.get_vocabulary_size() == 3
        
        # 验证词项存在
        assert "中国" in index.get_all_terms()
        assert "北京" in index.get_all_terms()
        assert "首都" in index.get_all_terms()
    
    def test_build_index_positions(self):
        """测试位置信息记录"""
        index = InvertedIndex()
        tokens = ["中国", "北京", "中国", "首都"]
        
        index.build_index(doc_id=1, tokens=tokens)
        
        # 获取"中国"的倒排列表
        posting_list = index.get_posting_list("中国")
        assert len(posting_list) == 1
        
        posting = posting_list[0]
        assert posting.doc_id == 1
        assert posting.positions == [0, 2]  # "中国"出现在位置0和2
        assert posting.term_freq == 2
    
    def test_build_index_multiple_documents(self):
        """测试多个文档的索引构建"""
        index = InvertedIndex()
        
        # 文档1
        tokens1 = ["中国", "北京"]
        index.build_index(doc_id=1, tokens=tokens1)
        
        # 文档2
        tokens2 = ["中国", "上海"]
        index.build_index(doc_id=2, tokens=tokens2)
        
        # 验证"中国"在两个文档中都出现
        posting_list = index.get_posting_list("中国")
        assert len(posting_list) == 2
        
        doc_ids = [p.doc_id for p in posting_list]
        assert 1 in doc_ids
        assert 2 in doc_ids
    
    def test_get_posting_list_nonexistent_term(self):
        """测试获取不存在词项的倒排列表"""
        index = InvertedIndex()
        tokens = ["中国", "北京"]
        index.build_index(doc_id=1, tokens=tokens)
        
        # 查询不存在的词项
        posting_list = index.get_posting_list("不存在")
        assert posting_list == []
    
    def test_get_term_frequency(self):
        """测试词项文档频率"""
        index = InvertedIndex()
        
        # 文档1
        index.build_index(doc_id=1, tokens=["中国", "北京"])
        # 文档2
        index.build_index(doc_id=2, tokens=["中国", "上海"])
        # 文档3
        index.build_index(doc_id=3, tokens=["日本", "东京"])
        
        # "中国"出现在2个文档中
        assert index.get_term_frequency("中国") == 2
        
        # "北京"出现在1个文档中
        assert index.get_term_frequency("北京") == 1
        
        # 不存在的词项
        assert index.get_term_frequency("不存在") == 0
    
    def test_get_vocabulary_size(self):
        """测试词汇表大小"""
        index = InvertedIndex()
        
        index.build_index(doc_id=1, tokens=["中国", "北京", "中国"])
        index.build_index(doc_id=2, tokens=["中国", "上海"])
        
        # 唯一词项：中国、北京、上海
        assert index.get_vocabulary_size() == 3
    
    def test_get_document_length(self):
        """测试文档长度记录"""
        index = InvertedIndex()
        
        tokens1 = ["中国", "北京", "中国", "首都"]
        index.build_index(doc_id=1, tokens=tokens1)
        
        tokens2 = ["日本", "东京"]
        index.build_index(doc_id=2, tokens=tokens2)
        
        assert index.get_document_length(1) == 4
        assert index.get_document_length(2) == 2
        assert index.get_document_length(999) == 0  # 不存在的文档
    
    def test_save_and_load(self):
        """测试索引的保存和加载"""
        index = InvertedIndex()
        
        # 构建索引
        index.build_index(doc_id=1, tokens=["中国", "北京"])
        index.build_index(doc_id=2, tokens=["中国", "上海"])
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp_path = tmp.name
        
        try:
            index.save(tmp_path)
            
            # 创建新索引并加载
            new_index = InvertedIndex()
            new_index.load(tmp_path)
            
            # 验证加载后的索引
            assert new_index.get_vocabulary_size() == 3
            assert new_index.get_term_frequency("中国") == 2
            assert new_index.get_document_length(1) == 2
            
            posting_list = new_index.get_posting_list("中国")
            assert len(posting_list) == 2
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        index = InvertedIndex()
        
        with pytest.raises(FileNotFoundError):
            index.load("nonexistent_file.pkl")
    
    def test_empty_tokens(self):
        """测试空词项列表"""
        index = InvertedIndex()
        
        index.build_index(doc_id=1, tokens=[])
        
        assert index.get_vocabulary_size() == 0
        assert index.get_document_length(1) == 0
