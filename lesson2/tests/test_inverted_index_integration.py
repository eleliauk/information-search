"""
倒排索引与其他组件的集成测试
"""
import pytest
from src.inverted_index import InvertedIndex
from src.text_preprocessor import TextPreprocessor
from src.document_store import DocumentStore


class TestInvertedIndexIntegration:
    """测试倒排索引与其他组件的集成"""
    
    def test_integration_with_text_preprocessor(self):
        """测试与文本预处理器的集成"""
        # 创建组件
        preprocessor = TextPreprocessor()
        index = InvertedIndex()
        
        # 预处理文本
        text = "中国是一个伟大的国家，北京是中国的首都。"
        tokens = preprocessor.tokenize(text)
        
        # 构建索引
        index.build_index(doc_id=1, tokens=tokens)
        
        # 验证索引构建成功
        assert index.get_vocabulary_size() > 0
        assert index.get_term_frequency("中国") > 0
    
    def test_integration_with_document_store(self):
        """测试与文档存储的集成"""
        # 创建组件
        doc_store = DocumentStore()
        preprocessor = TextPreprocessor()
        index = InvertedIndex()
        
        # 添加文档
        content1 = "中国北京"
        doc_id1 = doc_store.add_document(content1)
        tokens1 = preprocessor.tokenize(content1)
        index.build_index(doc_id1, tokens1)
        
        content2 = "中国上海"
        doc_id2 = doc_store.add_document(content2)
        tokens2 = preprocessor.tokenize(content2)
        index.build_index(doc_id2, tokens2)
        
        # 验证索引和文档存储一致
        assert doc_store.document_count() == 2
        
        # 查询"中国"应该返回两个文档
        posting_list = index.get_posting_list("中国")
        assert len(posting_list) == 2
        
        # 验证文档ID匹配
        doc_ids = [p.doc_id for p in posting_list]
        assert doc_id1 in doc_ids
        assert doc_id2 in doc_ids
    
    def test_full_workflow(self):
        """测试完整的工作流程"""
        # 创建所有组件
        doc_store = DocumentStore()
        preprocessor = TextPreprocessor()
        index = InvertedIndex()
        
        # 添加多个文档
        documents = [
            "中国是一个伟大的国家",
            "北京是中国的首都",
            "上海是中国的经济中心",
            "日本是一个岛国"
        ]
        
        for content in documents:
            doc_id = doc_store.add_document(content)
            tokens = preprocessor.tokenize(content)
            index.build_index(doc_id, tokens)
        
        # 验证索引统计
        assert doc_store.document_count() == 4
        assert index.get_vocabulary_size() > 0
        
        # 查询"中国"应该在3个文档中
        china_postings = index.get_posting_list("中国")
        assert len(china_postings) == 3
        
        # 查询"日本"应该在1个文档中
        japan_postings = index.get_posting_list("日本")
        assert len(japan_postings) == 1
        
        # 查询不存在的词项
        nonexistent_postings = index.get_posting_list("不存在的词")
        assert len(nonexistent_postings) == 0
