"""
查询处理器测试
"""
import pytest
from src.query_processor import QueryProcessor
from src.inverted_index import InvertedIndex
from src.text_preprocessor import TextPreprocessor
from src.document_store import DocumentStore


class TestQueryProcessor:
    """QueryProcessor 基本功能测试"""
    
    @pytest.fixture
    def setup_system(self):
        """设置测试系统"""
        # 创建组件
        doc_store = DocumentStore()
        preprocessor = TextPreprocessor()
        index = InvertedIndex()
        query_processor = QueryProcessor(index, preprocessor, doc_store)
        
        # 添加测试文档
        doc1_content = "Python是一种编程语言"
        doc2_content = "Java也是编程语言"
        doc3_content = "Python很流行"
        
        doc1_id = doc_store.add_document(doc1_content)
        doc2_id = doc_store.add_document(doc2_content)
        doc3_id = doc_store.add_document(doc3_content)
        
        # 构建索引
        tokens1 = preprocessor.tokenize(doc1_content)
        tokens2 = preprocessor.tokenize(doc2_content)
        tokens3 = preprocessor.tokenize(doc3_content)
        
        index.build_index(doc1_id, tokens1)
        index.build_index(doc2_id, tokens2)
        index.build_index(doc3_id, tokens3)
        
        return {
            'query_processor': query_processor,
            'doc_store': doc_store,
            'index': index,
            'preprocessor': preprocessor,
            'doc_ids': [doc1_id, doc2_id, doc3_id]
        }
    
    def test_single_term_query(self, setup_system):
        """测试单词项查询"""
        qp = setup_system['query_processor']
        
        # 查询 "Python"
        results = qp.search("Python")
        
        # 应该返回文档1和文档3
        assert len(results) == 2
        assert 1 in results
        assert 3 in results
    
    def test_query_with_preprocessing(self, setup_system):
        """测试查询词预处理"""
        qp = setup_system['query_processor']
        
        # 查询大写的 "PYTHON"，应该与小写 "python" 返回相同结果
        results_upper = qp.search("PYTHON")
        results_lower = qp.search("python")
        
        assert results_upper == results_lower
        assert len(results_upper) == 2
    
    def test_nonexistent_term_query(self, setup_system):
        """测试不存在的词项查询"""
        qp = setup_system['query_processor']
        
        # 查询不存在的词项
        results = qp.search("不存在的词")
        
        # 应该返回空列表
        assert results == []
    
    def test_empty_query(self, setup_system):
        """测试空查询"""
        qp = setup_system['query_processor']
        
        # 空查询
        results = qp.search("")
        
        # 应该返回空列表
        assert results == []
    
    def test_query_result_sorted(self, setup_system):
        """测试查询结果排序"""
        qp = setup_system['query_processor']
        
        # 查询 "编程语言"（出现在文档1和2中）
        results = qp.search("编程语言")
        
        # 结果应该按文档ID升序排列
        assert results == sorted(results)
        assert len(results) == 2
        assert results[0] < results[1]
    
    def test_common_term_query(self, setup_system):
        """测试常见词项查询"""
        qp = setup_system['query_processor']
        
        # 查询 "编程语言"（出现在文档1和2中）
        results = qp.search("编程语言")
        
        assert len(results) == 2
        assert 1 in results
        assert 2 in results


class TestBooleanQuery:
    """布尔查询功能测试"""
    
    @pytest.fixture
    def setup_system(self):
        """设置测试系统"""
        # 创建组件
        doc_store = DocumentStore()
        preprocessor = TextPreprocessor()
        index = InvertedIndex()
        query_processor = QueryProcessor(index, preprocessor, doc_store)
        
        # 添加测试文档
        doc1_content = "Python是一种编程语言"
        doc2_content = "Java也是编程语言"
        doc3_content = "Python很流行"
        doc4_content = "机器学习使用Python"
        
        doc1_id = doc_store.add_document(doc1_content)
        doc2_id = doc_store.add_document(doc2_content)
        doc3_id = doc_store.add_document(doc3_content)
        doc4_id = doc_store.add_document(doc4_content)
        
        # 构建索引
        tokens1 = preprocessor.tokenize(doc1_content)
        tokens2 = preprocessor.tokenize(doc2_content)
        tokens3 = preprocessor.tokenize(doc3_content)
        tokens4 = preprocessor.tokenize(doc4_content)
        
        index.build_index(doc1_id, tokens1)
        index.build_index(doc2_id, tokens2)
        index.build_index(doc3_id, tokens3)
        index.build_index(doc4_id, tokens4)
        
        return {
            'query_processor': query_processor,
            'doc_store': doc_store,
            'index': index,
            'preprocessor': preprocessor,
            'doc_ids': [doc1_id, doc2_id, doc3_id, doc4_id]
        }
    
    def test_and_query(self, setup_system):
        """测试 AND 查询"""
        qp = setup_system['query_processor']
        
        # 查询同时包含 "Python" 和 "编程语言" 的文档
        results = qp.and_query(["Python", "编程语言"])
        
        # 只有文档1同时包含这两个词
        assert len(results) == 1
        assert 1 in results
    
    def test_and_query_multiple_terms(self, setup_system):
        """测试多个词项的 AND 查询"""
        qp = setup_system['query_processor']
        
        # 查询同时包含 "Python" 和 "流行" 的文档
        results = qp.and_query(["Python", "流行"])
        
        # 只有文档3同时包含这两个词
        assert len(results) == 1
        assert 3 in results
    
    def test_and_query_no_match(self, setup_system):
        """测试 AND 查询无匹配结果"""
        qp = setup_system['query_processor']
        
        # 查询同时包含 "Java" 和 "流行" 的文档（不存在）
        results = qp.and_query(["Java", "流行"])
        
        # 应该返回空列表
        assert results == []
    
    def test_or_query(self, setup_system):
        """测试 OR 查询"""
        qp = setup_system['query_processor']
        
        # 查询包含 "Java" 或 "流行" 的文档
        results = qp.or_query(["Java", "流行"])
        
        # 文档2包含Java，文档3包含流行
        assert len(results) == 2
        assert 2 in results
        assert 3 in results
    
    def test_or_query_multiple_terms(self, setup_system):
        """测试多个词项的 OR 查询"""
        qp = setup_system['query_processor']
        
        # 查询包含 "Python" 或 "Java" 的文档
        results = qp.or_query(["Python", "Java"])
        
        # 文档1,3,4包含Python，文档2包含Java
        assert len(results) == 4
        assert set(results) == {1, 2, 3, 4}
    
    def test_not_query(self, setup_system):
        """测试 NOT 查询"""
        qp = setup_system['query_processor']
        
        # 查询包含 "编程语言" 但不包含 "Java" 的文档
        results = qp.not_query(["编程语言"], ["Java"])
        
        # 文档1包含编程语言但不包含Java
        assert len(results) == 1
        assert 1 in results
    
    def test_not_query_exclude_only(self, setup_system):
        """测试只有排除词的 NOT 查询"""
        qp = setup_system['query_processor']
        
        # 查询不包含 "Python" 的所有文档
        results = qp.not_query([], ["Python"])
        
        # 只有文档2不包含Python
        assert len(results) == 1
        assert 2 in results
    
    def test_boolean_search_and(self, setup_system):
        """测试布尔查询字符串 - AND"""
        qp = setup_system['query_processor']
        
        # 使用字符串查询
        results = qp.boolean_search("Python AND 编程语言")
        
        # 只有文档1同时包含这两个词
        assert len(results) == 1
        assert 1 in results
    
    def test_boolean_search_or(self, setup_system):
        """测试布尔查询字符串 - OR"""
        qp = setup_system['query_processor']
        
        # 使用字符串查询
        results = qp.boolean_search("Java OR 流行")
        
        # 文档2包含Java，文档3包含流行
        assert len(results) == 2
        assert 2 in results
        assert 3 in results
    
    def test_boolean_search_not(self, setup_system):
        """测试布尔查询字符串 - NOT"""
        qp = setup_system['query_processor']
        
        # 使用字符串查询
        results = qp.boolean_search("编程语言 AND NOT Java")
        
        # 文档1包含编程语言但不包含Java
        assert len(results) == 1
        assert 1 in results
    
    def test_boolean_search_complex(self, setup_system):
        """测试复杂布尔查询"""
        qp = setup_system['query_processor']
        
        # 测试运算符优先级：NOT > AND > OR
        # "Python AND 编程语言 OR Java" 应该解析为 "(Python AND 编程语言) OR Java"
        results = qp.boolean_search("Python AND 编程语言 OR Java")
        
        # 文档1包含Python和编程语言，文档2包含Java
        assert len(results) == 2
        assert 1 in results
        assert 2 in results
    
    def test_boolean_search_single_term(self, setup_system):
        """测试布尔查询退化为单词项查询"""
        qp = setup_system['query_processor']
        
        # 没有运算符的查询应该等同于单词项查询
        results = qp.boolean_search("Python")
        
        # 应该返回所有包含Python的文档
        assert len(results) == 3
        assert set(results) == {1, 3, 4}
    
    def test_boolean_search_empty(self, setup_system):
        """测试空布尔查询"""
        qp = setup_system['query_processor']
        
        # 空查询
        results = qp.boolean_search("")
        
        # 应该返回空列表
        assert results == []
