"""
IndexSystem 集成测试
测试索引系统的主入口功能
"""
import pytest
import tempfile
import os
from pathlib import Path

from src.index_system import IndexSystem


class TestIndexSystem:
    """测试 IndexSystem 类"""
    
    def test_initialization(self):
        """测试索引系统初始化"""
        system = IndexSystem()
        assert system is not None
    
    def test_add_document(self):
        """测试添加文档"""
        system = IndexSystem()
        
        # 添加文档
        doc_id = system.add_document("这是一个测试文档")
        assert doc_id == 1
        
        # 添加第二个文档
        doc_id2 = system.add_document("这是另一个测试文档")
        assert doc_id2 == 2
    
    def test_add_empty_document(self):
        """测试添加空文档应该失败"""
        system = IndexSystem()
        
        with pytest.raises(ValueError):
            system.add_document("")
        
        with pytest.raises(ValueError):
            system.add_document("   ")
    
    def test_search(self):
        """测试单词项查询"""
        system = IndexSystem()
        
        # 添加文档
        doc_id1 = system.add_document("Python 是一门编程语言")
        doc_id2 = system.add_document("Java 也是一门编程语言")
        doc_id3 = system.add_document("Python 很流行")
        
        # 查询 "python"（应该找到 doc1 和 doc3）
        results = system.search("Python")
        assert sorted(results) == [doc_id1, doc_id3]
        
        # 查询 "java"（应该找到 doc2）
        results = system.search("Java")
        assert results == [doc_id2]
        
        # 查询不存在的词
        results = system.search("不存在的词")
        assert results == []
    
    def test_boolean_search_and(self):
        """测试 AND 布尔查询"""
        system = IndexSystem()
        
        doc_id1 = system.add_document("Python 是一门编程语言")
        doc_id2 = system.add_document("Java 也是一门编程语言")
        doc_id3 = system.add_document("Python 很流行")
        
        # 查询包含 "Python" AND "一门"（两个文档都有"一门"）
        results = system.boolean_search("Python AND 一门")
        assert results == [doc_id1]
    
    def test_boolean_search_or(self):
        """测试 OR 布尔查询"""
        system = IndexSystem()
        
        doc_id1 = system.add_document("Python 是一门编程语言")
        doc_id2 = system.add_document("Java 也是一门编程语言")
        doc_id3 = system.add_document("Python 很流行")
        
        # 查询包含 "Python" OR "Java"
        results = system.boolean_search("Python OR Java")
        assert sorted(results) == sorted([doc_id1, doc_id2, doc_id3])
    
    def test_boolean_search_not(self):
        """测试 NOT 布尔查询"""
        system = IndexSystem()
        
        doc_id1 = system.add_document("Python 是一门编程语言")
        doc_id2 = system.add_document("Java 也是一门编程语言")
        doc_id3 = system.add_document("Python 很流行")
        
        # 查询包含 "一门" 但不包含 "Java"
        results = system.boolean_search("一门 AND NOT Java")
        assert results == [doc_id1]
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        system = IndexSystem()
        
        # 添加文档
        system.add_document("这是第一个文档")
        system.add_document("这是第二个文档")
        
        # 获取统计信息
        stats = system.get_statistics()
        
        assert stats['document_count'] == 2
        assert stats['vocabulary_size'] > 0
        assert stats['total_tokens'] > 0
        assert stats['avg_doc_length'] > 0
    
    def test_save_and_load_index(self):
        """测试索引的保存和加载"""
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test_index")
            
            # 创建索引并添加文档
            system1 = IndexSystem()
            doc_id1 = system1.add_document("Python 是一门编程语言")
            doc_id2 = system1.add_document("Java 也是一门编程语言")
            
            # 保存索引
            success = system1.save_index(index_path)
            assert success is True
            
            # 创建新的系统实例并加载索引
            system2 = IndexSystem()
            success = system2.load_index(index_path)
            assert success is True
            
            # 验证加载后的索引可以正常查询
            results = system2.search("Python")
            assert doc_id1 in results
            
            results = system2.search("Java")
            assert doc_id2 in results
            
            # 验证统计信息一致
            stats = system2.get_statistics()
            assert stats['document_count'] == 2
    
    def test_load_nonexistent_index(self):
        """测试加载不存在的索引文件"""
        system = IndexSystem()
        
        success = system.load_index("/nonexistent/path/index")
        assert success is False
    
    def test_get_document(self):
        """测试获取文档内容"""
        system = IndexSystem()
        
        # 添加文档
        content = "这是一个测试文档"
        metadata = {"title": "测试"}
        doc_id = system.add_document(content, metadata)
        
        # 获取文档
        doc = system.get_document(doc_id)
        assert doc is not None
        assert doc['doc_id'] == doc_id
        assert doc['content'] == content
        assert doc['metadata'] == metadata
        assert doc['token_count'] > 0
        
        # 获取不存在的文档
        doc = system.get_document(9999)
        assert doc is None
    
    def test_with_stopwords(self):
        """测试使用停用词文件"""
        stopwords_path = "lesson2/config/stopwords.txt"
        
        # 检查停用词文件是否存在
        if not Path(stopwords_path).exists():
            pytest.skip("停用词文件不存在")
        
        system = IndexSystem(stopwords_path=stopwords_path)
        
        # 添加包含停用词的文档
        doc_id = system.add_document("这是一个测试文档")
        
        # 验证系统正常工作
        assert doc_id == 1
        
        # 获取统计信息
        stats = system.get_statistics()
        assert stats['document_count'] == 1
