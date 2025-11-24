"""
查询处理组件
负责处理用户查询并返回结果
"""
from typing import List, Set
from .inverted_index import InvertedIndex
from .text_preprocessor import TextPreprocessor
from .document_store import DocumentStore


class QueryProcessor:
    """
    查询处理器类
    
    职责：
    - 处理单词项查询
    - 处理布尔查询（AND/OR/NOT）
    - 对查询词进行预处理
    - 合并和排序查询结果
    """
    
    def __init__(
        self,
        inverted_index: InvertedIndex,
        text_preprocessor: TextPreprocessor,
        document_store: DocumentStore
    ):
        """
        初始化查询处理器
        
        Args:
            inverted_index: 倒排索引实例
            text_preprocessor: 文本预处理器实例
            document_store: 文档存储实例
        """
        self._index = inverted_index
        self._preprocessor = text_preprocessor
        self._doc_store = document_store
    
    def search(self, query: str) -> List[int]:
        """
        执行单词项查询
        
        处理流程：
        1. 对查询词进行预处理（与文档相同的预处理）
        2. 从倒排索引中获取倒排列表
        3. 提取文档ID
        4. 排序并返回结果
        
        Args:
            query: 查询字符串
        
        Returns:
            包含查询词的文档ID列表（按升序排列）
            如果查询词不存在，返回空列表
        """
        # 1. 对查询词进行预处理
        # 使用与文档相同的预处理流程
        tokens = self._preprocessor.tokenize(query)
        
        # 如果预处理后没有有效词项，返回空列表
        if not tokens:
            return []
        
        # 对于单词项查询，取第一个词项
        term = tokens[0]
        
        # 2. 从倒排索引中获取倒排列表
        posting_list = self._index.get_posting_list(term)
        
        # 3. 提取文档ID
        doc_ids = [posting.doc_id for posting in posting_list]
        
        # 4. 排序并返回结果（按文档ID升序）
        doc_ids.sort()
        
        return doc_ids
    
    def and_query(self, terms: List[str]) -> List[int]:
        """
        执行 AND 查询（交集运算）
        
        返回同时包含所有查询词的文档
        
        Args:
            terms: 查询词列表
        
        Returns:
            同时包含所有查询词的文档ID列表（按升序排列）
        """
        if not terms:
            return []
        
        # 对每个词项进行预处理并获取文档集合
        doc_sets = []
        for term in terms:
            # 预处理查询词
            processed_tokens = self._preprocessor.tokenize(term)
            if not processed_tokens:
                # 如果某个词项预处理后为空，返回空结果
                return []
            
            processed_term = processed_tokens[0]
            
            # 获取包含该词项的文档ID集合
            posting_list = self._index.get_posting_list(processed_term)
            doc_ids = {posting.doc_id for posting in posting_list}
            doc_sets.append(doc_ids)
        
        # 计算交集
        result_set = doc_sets[0]
        for doc_set in doc_sets[1:]:
            result_set = result_set.intersection(doc_set)
        
        # 转换为排序列表
        result = sorted(list(result_set))
        return result
    
    def or_query(self, terms: List[str]) -> List[int]:
        """
        执行 OR 查询（并集运算）
        
        返回包含任意查询词的文档
        
        Args:
            terms: 查询词列表
        
        Returns:
            包含任意查询词的文档ID列表（按升序排列）
        """
        if not terms:
            return []
        
        # 收集所有文档ID
        result_set = set()
        
        for term in terms:
            # 预处理查询词
            processed_tokens = self._preprocessor.tokenize(term)
            if not processed_tokens:
                continue
            
            processed_term = processed_tokens[0]
            
            # 获取包含该词项的文档ID
            posting_list = self._index.get_posting_list(processed_term)
            doc_ids = {posting.doc_id for posting in posting_list}
            
            # 添加到结果集（并集）
            result_set = result_set.union(doc_ids)
        
        # 转换为排序列表
        result = sorted(list(result_set))
        return result
    
    def not_query(self, include_terms: List[str], exclude_terms: List[str]) -> List[int]:
        """
        执行 NOT 查询（差集运算）
        
        返回包含 include_terms 但不包含 exclude_terms 的文档
        
        Args:
            include_terms: 必须包含的词项列表
            exclude_terms: 必须排除的词项列表
        
        Returns:
            符合条件的文档ID列表（按升序排列）
        """
        # 如果没有包含词项，从所有文档开始
        if not include_terms:
            # 获取所有文档ID
            all_docs = self._doc_store.get_all_documents()
            result_set = {doc.doc_id for doc in all_docs}
        else:
            # 先执行 AND 查询获取包含所有 include_terms 的文档
            result_set = set(self.and_query(include_terms))
        
        # 如果没有排除词项，直接返回
        if not exclude_terms:
            return sorted(list(result_set))
        
        # 获取需要排除的文档ID
        exclude_set = set()
        for term in exclude_terms:
            # 预处理查询词
            processed_tokens = self._preprocessor.tokenize(term)
            if not processed_tokens:
                continue
            
            processed_term = processed_tokens[0]
            
            # 获取包含该词项的文档ID
            posting_list = self._index.get_posting_list(processed_term)
            doc_ids = {posting.doc_id for posting in posting_list}
            exclude_set = exclude_set.union(doc_ids)
        
        # 计算差集
        result_set = result_set.difference(exclude_set)
        
        # 转换为排序列表
        result = sorted(list(result_set))
        return result
    
    def boolean_search(self, query: str) -> List[int]:
        """
        执行布尔查询
        
        解析布尔查询字符串并执行相应的查询操作
        支持的运算符：AND, OR, NOT
        运算符优先级：NOT > AND > OR
        
        查询格式示例：
        - "term1 AND term2"
        - "term1 OR term2"
        - "term1 AND NOT term2"
        - "term1 OR term2 AND term3"
        
        Args:
            query: 布尔查询字符串
        
        Returns:
            符合查询条件的文档ID列表（按升序排列）
        """
        if not query or not query.strip():
            return []
        
        # 简单的查询解析
        # 按优先级处理：NOT > AND > OR
        
        # 首先按 OR 分割（最低优先级）
        or_parts = query.split(' OR ')
        
        if len(or_parts) > 1:
            # 有 OR 运算符，递归处理每个部分并合并结果
            all_results = []
            for part in or_parts:
                part_results = self.boolean_search(part.strip())
                all_results.extend(part_results)
            # 去重并排序
            return sorted(list(set(all_results)))
        
        # 没有 OR，检查 AND
        and_parts = query.split(' AND ')
        
        if len(and_parts) > 1:
            # 有 AND 运算符
            # 分离出 NOT 部分
            include_terms = []
            exclude_terms = []
            
            for part in and_parts:
                part = part.strip()
                if part.startswith('NOT '):
                    # 这是一个 NOT 词项
                    exclude_term = part[4:].strip()
                    exclude_terms.append(exclude_term)
                else:
                    # 这是一个普通词项
                    include_terms.append(part)
            
            # 执行 NOT 查询
            if exclude_terms:
                return self.not_query(include_terms, exclude_terms)
            else:
                # 没有 NOT，只是 AND 查询
                return self.and_query(include_terms)
        
        # 没有 AND 和 OR，检查是否是单独的 NOT
        if query.strip().startswith('NOT '):
            exclude_term = query.strip()[4:].strip()
            return self.not_query([], [exclude_term])
        
        # 单个词项查询
        return self.search(query.strip())
