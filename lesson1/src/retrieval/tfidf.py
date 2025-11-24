#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple, Any, Union, Optional
import logging
import pickle
from pathlib import Path


class TFIDFVectorizer:
    """TF-IDF向量化器"""
    
    def __init__(self, min_df: int = 2, max_df: float = 0.95, use_log_tf: bool = True, 
                 normalize: bool = True):
        """
        初始化TF-IDF向量化器
        
        Args:
            min_df: 最小文档频率，低于此频率的词将被过滤
            max_df: 最大文档频率比例，高于此比例的词将被过滤
            use_log_tf: 是否使用对数词频
            normalize: 是否进行L2归一化
        """
        self.min_df = min_df
        self.max_df = max_df
        self.use_log_tf = use_log_tf
        self.normalize = normalize
        
        self.vocabulary = {}  # 词汇表 {word: index}
        self.idf_values = {}  # IDF值 {word: idf}
        self.feature_names = []  # 特征名称列表
        self.n_documents = 0  # 文档总数
        
        self.logger = logging.getLogger(__name__)
        
    def _extract_tokens(self, tokenized_doc: List[Union[Tuple[str, str], str]]) -> List[str]:
        """从分词结果中提取词汇"""
        tokens = []
        for token in tokenized_doc:
            if isinstance(token, tuple):
                tokens.append(token[0])  # (word, pos)
            else:
                tokens.append(token)  # word
        return tokens
    
    def _build_vocabulary(self, tokenized_documents: List[List[Union[Tuple[str, str], str]]]) -> None:
        """构建词汇表"""
        self.logger.info("构建词汇表...")
        
        # 统计词汇在多少个文档中出现
        doc_frequency = Counter()
        all_tokens = set()
        
        for doc_tokens in tokenized_documents:
            tokens = self._extract_tokens(doc_tokens)
            unique_tokens = set(tokens)
            
            for token in unique_tokens:
                doc_frequency[token] += 1
            
            all_tokens.update(tokens)
        
        # 根据文档频率过滤词汇
        n_docs = len(tokenized_documents)
        filtered_tokens = []
        
        for token, df in doc_frequency.items():
            # 最小文档频率过滤
            if df < self.min_df:
                continue
            
            # 最大文档频率过滤
            if df / n_docs > self.max_df:
                continue
            
            filtered_tokens.append(token)
        
        # 构建词汇表索引
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(filtered_tokens))}
        self.feature_names = sorted(filtered_tokens)
        
        self.logger.info(f"词汇表构建完成，词汇数量: {len(self.vocabulary)}")
        self.logger.info(f"原始词汇数: {len(all_tokens)}, 过滤后词汇数: {len(self.vocabulary)}")
    
    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """计算词频 (Term Frequency)"""
        tf_dict = Counter(tokens)
        doc_length = len(tokens)
        
        if doc_length == 0:
            return {}
        
        # 计算TF值
        tf_normalized = {}
        for token, count in tf_dict.items():
            if token in self.vocabulary:
                if self.use_log_tf:
                    # 对数归一化: 1 + log(tf)
                    tf_normalized[token] = 1 + math.log(count) if count > 0 else 0
                else:
                    # 简单归一化: tf / doc_length
                    tf_normalized[token] = count / doc_length
        
        return tf_normalized
    
    def _calculate_idf(self, tokenized_documents: List[List[Union[Tuple[str, str], str]]]) -> None:
        """计算逆文档频率 (Inverse Document Frequency)"""
        self.logger.info("计算IDF值...")
        
        N = len(tokenized_documents)  # 文档总数
        self.n_documents = N
        
        # 计算每个词出现在多少个文档中
        document_frequencies = Counter()
        
        for doc_tokens in tokenized_documents:
            tokens = self._extract_tokens(doc_tokens)
            unique_tokens = set(tokens)
            
            for token in unique_tokens:
                if token in self.vocabulary:
                    document_frequencies[token] += 1
        
        # 计算IDF: log(N / df)
        for token in self.vocabulary:
            df = document_frequencies.get(token, 0)
            if df > 0:
                self.idf_values[token] = math.log(N / df)
            else:
                self.idf_values[token] = 0
        
        self.logger.info("IDF计算完成")
    
    def _calculate_tfidf_vector(self, tokens: List[str]) -> np.ndarray:
        """计算单个文档的TF-IDF向量"""
        tf_dict = self._calculate_tf(tokens)
        
        # 创建TF-IDF向量
        tfidf_vector = np.zeros(len(self.vocabulary))
        
        for token, tf_value in tf_dict.items():
            if token in self.vocabulary:
                word_index = self.vocabulary[token]
                idf_value = self.idf_values.get(token, 0)
                tfidf_vector[word_index] = tf_value * idf_value
        
        # L2归一化
        if self.normalize:
            norm = np.linalg.norm(tfidf_vector)
            if norm > 0:
                tfidf_vector = tfidf_vector / norm
        
        return tfidf_vector
    
    def fit_transform(self, tokenized_documents: List[List[Union[Tuple[str, str], str]]]) -> csr_matrix:
        """
        训练并转换文档集合为TF-IDF矩阵
        
        Args:
            tokenized_documents: 分词后的文档列表
            
        Returns:
            TF-IDF稀疏矩阵
        """
        self.logger.info(f"开始TF-IDF向量化，文档数量: {len(tokenized_documents)}")
        
        # 1. 构建词汇表
        self._build_vocabulary(tokenized_documents)
        
        # 2. 计算IDF值
        self._calculate_idf(tokenized_documents)
        
        # 3. 计算每个文档的TF-IDF向量
        doc_vectors = []
        
        for i, doc_tokens in enumerate(tokenized_documents):
            if i % 100 == 0:
                self.logger.info(f"向量化进度: {i}/{len(tokenized_documents)}")
            
            tokens = self._extract_tokens(doc_tokens)
            tfidf_vector = self._calculate_tfidf_vector(tokens)
            doc_vectors.append(tfidf_vector)
        
        # 转换为稀疏矩阵以节省内存
        doc_matrix = csr_matrix(np.array(doc_vectors))
        
        self.logger.info(f"TF-IDF矩阵构建完成，形状: {doc_matrix.shape}")
        self.logger.info(f"矩阵稀疏度: {1 - doc_matrix.nnz / (doc_matrix.shape[0] * doc_matrix.shape[1]):.4f}")
        
        return doc_matrix
    
    def transform(self, tokenized_documents: List[List[Union[Tuple[str, str], str]]]) -> csr_matrix:
        """
        使用已训练的模型转换新文档
        
        Args:
            tokenized_documents: 分词后的文档列表
            
        Returns:
            TF-IDF稀疏矩阵
        """
        if not self.vocabulary:
            raise ValueError("模型未训练，请先调用 fit_transform()")
        
        doc_vectors = []
        
        for doc_tokens in tokenized_documents:
            tokens = self._extract_tokens(doc_tokens)
            tfidf_vector = self._calculate_tfidf_vector(tokens)
            doc_vectors.append(tfidf_vector)
        
        return csr_matrix(np.array(doc_vectors))
    
    def transform_query(self, query_tokens: List[str]) -> np.ndarray:
        """
        转换查询为TF-IDF向量
        
        Args:
            query_tokens: 查询的分词结果
            
        Returns:
            查询的TF-IDF向量
        """
        if not self.vocabulary:
            raise ValueError("模型未训练，请先调用 fit_transform()")
        
        query_vector = self._calculate_tfidf_vector(query_tokens)
        return query_vector.reshape(1, -1)  # 转换为行向量
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_names.copy()
    
    def get_vocabulary_size(self) -> int:
        """获取词汇表大小"""
        return len(self.vocabulary)
    
    def get_word_index(self, word: str) -> Optional[int]:
        """获取词汇在词汇表中的索引"""
        return self.vocabulary.get(word)
    
    def get_word_idf(self, word: str) -> Optional[float]:
        """获取词汇的IDF值"""
        return self.idf_values.get(word)
    
    def get_top_features_by_idf(self, top_k: int = 50) -> List[Tuple[str, float]]:
        """获取IDF值最高的前k个特征"""
        if not self.idf_values:
            return []
        
        sorted_features = sorted(self.idf_values.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_k]
    
    def analyze_document_vector(self, doc_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        分析文档向量，返回权重最高的词汇
        
        Args:
            doc_vector: 文档的TF-IDF向量
            top_k: 返回前k个重要词汇
            
        Returns:
            (词汇, TF-IDF权重) 的列表
        """
        if len(doc_vector.shape) > 1:
            doc_vector = doc_vector.flatten()
        
        # 获取非零元素的索引和值
        nonzero_indices = np.nonzero(doc_vector)[0]
        nonzero_values = doc_vector[nonzero_indices]
        
        # 排序并获取前k个
        sorted_indices = np.argsort(nonzero_values)[::-1][:top_k]
        
        top_features = []
        for idx in sorted_indices:
            word_idx = nonzero_indices[idx]
            word = self.feature_names[word_idx]
            weight = nonzero_values[idx]
            top_features.append((word, weight))
        
        return top_features
    
    def save_model(self, filepath: str) -> None:
        """保存模型到文件"""
        model_data = {
            'vocabulary': self.vocabulary,
            'idf_values': self.idf_values,
            'feature_names': self.feature_names,
            'n_documents': self.n_documents,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'use_log_tf': self.use_log_tf,
            'normalize': self.normalize
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"TF-IDF模型已保存到: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocabulary = model_data['vocabulary']
        self.idf_values = model_data['idf_values']
        self.feature_names = model_data['feature_names']
        self.n_documents = model_data['n_documents']
        self.min_df = model_data['min_df']
        self.max_df = model_data['max_df']
        self.use_log_tf = model_data['use_log_tf']
        self.normalize = model_data['normalize']
        
        self.logger.info(f"TF-IDF模型已从 {filepath} 加载")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "词汇表大小": len(self.vocabulary),
            "文档数量": self.n_documents,
            "最小文档频率": self.min_df,
            "最大文档频率比例": self.max_df,
            "使用对数TF": self.use_log_tf,
            "L2归一化": self.normalize,
            "平均IDF": np.mean(list(self.idf_values.values())) if self.idf_values else 0,
            "IDF标准差": np.std(list(self.idf_values.values())) if self.idf_values else 0
        }


class TFIDFRetrieval:
    """基于TF-IDF的检索器"""
    
    def __init__(self, tokenizer, min_df: int = 2, max_df: float = 0.95):
        """
        初始化检索器
        
        Args:
            tokenizer: 分词器实例
            min_df: 最小文档频率
            max_df: 最大文档频率比例
        """
        self.tokenizer = tokenizer
        self.vectorizer = TFIDFVectorizer(min_df=min_df, max_df=max_df)
        self.doc_vectors = None  # 文档向量矩阵
        self.documents = []  # 原始文档
        self.is_fitted = False
        
        self.logger = logging.getLogger(__name__)
    
    def fit(self, documents: List[Dict[str, Any]]) -> None:
        """
        训练检索模型
        
        Args:
            documents: 文档列表
        """
        self.logger.info(f"开始训练TF-IDF检索模型，文档数量: {len(documents)}")
        
        self.documents = documents
        
        # 1. 对所有文档进行分词
        self.logger.info("文档分词中...")
        texts = [doc.get('content', '') for doc in documents]
        tokenized_docs = self.tokenizer.batch_tokenize(texts, keep_pos=True, show_progress=True)
        
        # 2. 构建TF-IDF矩阵
        self.doc_vectors = self.vectorizer.fit_transform(tokenized_docs)
        
        self.is_fitted = True
        self.logger.info("TF-IDF检索模型训练完成")
    
    def search(self, query: str, top_k: int = 10, similarity_threshold: float = 0.01) -> List[Dict[str, Any]]:
        """
        执行检索
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            
        Returns:
            检索结果列表
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用 fit()")
        
        # 1. 查询分词
        query_tokens = self.tokenizer.tokenize_document(query, keep_pos=False)
        
        if not query_tokens:
            self.logger.warning("查询分词结果为空")
            return []
        
        # 2. 查询向量化
        query_vector = self.vectorizer.transform_query(query_tokens)
        
        # 3. 计算相似度
        similarities = self._calculate_similarities(query_vector)
        
        # 4. 过滤和排序
        valid_indices = np.where(similarities > similarity_threshold)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # 按相似度排序
        valid_similarities = similarities[valid_indices]
        sorted_indices = valid_indices[np.argsort(valid_similarities)[::-1]]
        
        # 获取前top_k个结果
        top_indices = sorted_indices[:top_k]
        
        # 构造结果
        results = []
        for idx in top_indices:
            result = {
                'document_id': idx,
                'similarity_score': float(similarities[idx]),
                'document': self.documents[idx],
                'title': self.documents[idx].get('title', ''),
                'content_preview': self.documents[idx].get('content', '')[:200] + '...',
                'category': self.documents[idx].get('category', '未分类')
            }
            results.append(result)
        
        return results
    
    def _calculate_similarities(self, query_vector: np.ndarray) -> np.ndarray:
        """计算查询向量与所有文档向量的相似度（这里用点积作为相似度）"""
        # 由于向量已经归一化，点积等于余弦相似度
        similarities = self.doc_vectors.dot(query_vector.T).toarray().flatten()
        return similarities
    
    def explain_search(self, query: str, document_id: int) -> Dict[str, Any]:
        """
        解释检索结果
        
        Args:
            query: 查询字符串
            document_id: 文档ID
            
        Returns:
            解释信息
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用 fit()")
        
        if document_id >= len(self.documents):
            raise ValueError("文档ID超出范围")
        
        # 查询分词和向量化
        query_tokens = self.tokenizer.tokenize_document(query, keep_pos=False)
        query_vector = self.vectorizer.transform_query(query_tokens)
        
        # 获取文档向量
        doc_vector = self.doc_vectors[document_id].toarray().flatten()
        
        # 分析查询向量
        query_features = self.vectorizer.analyze_document_vector(query_vector.flatten(), top_k=10)
        
        # 分析文档向量
        doc_features = self.vectorizer.analyze_document_vector(doc_vector, top_k=10)
        
        # 找出共同特征
        query_words = {word for word, _ in query_features}
        doc_words = {word for word, _ in doc_features}
        common_words = query_words & doc_words
        
        # 计算相似度
        similarity = np.dot(query_vector.flatten(), doc_vector)
        
        return {
            "查询": query,
            "文档ID": document_id,
            "文档标题": self.documents[document_id].get('title', ''),
            "相似度": float(similarity),
            "查询关键词": query_features,
            "文档关键词": doc_features,
            "共同关键词": list(common_words),
            "查询词汇数": len(query_tokens),
            "匹配词汇数": len(common_words)
        }
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        if not self.is_fitted:
            return {"错误": "模型未训练"}
        
        stats = self.vectorizer.get_model_info()
        stats.update({
            "文档矩阵形状": self.doc_vectors.shape,
            "矩阵非零元素": self.doc_vectors.nnz,
            "矩阵稀疏度": 1 - self.doc_vectors.nnz / (self.doc_vectors.shape[0] * self.doc_vectors.shape[1]),
            "内存使用(MB)": self.doc_vectors.data.nbytes / (1024 * 1024)
        })
        
        return stats
    
    def save_model(self, filepath: str) -> None:
        """保存整个检索模型"""
        if not self.is_fitted:
            raise ValueError("模型未训练，无法保存")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'doc_vectors': self.doc_vectors,
            'documents': self.documents,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"检索模型已保存到: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """加载检索模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.doc_vectors = model_data['doc_vectors']
        self.documents = model_data['documents']
        self.is_fitted = model_data['is_fitted']
        
        self.logger.info(f"检索模型已从 {filepath} 加载")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    test_docs = [
        ["人工智能", "技术", "发展", "迅速"],
        ["机器学习", "算法", "应用", "广泛"],
        ["深度学习", "神经网络", "技术", "突破"],
        ["自然语言", "处理", "技术", "进步"]
    ]
    
    # 测试TF-IDF向量化器
    vectorizer = TFIDFVectorizer(min_df=1, max_df=1.0)
    
    print("=== TF-IDF向量化测试 ===")
    doc_matrix = vectorizer.fit_transform(test_docs)
    
    print(f"文档矩阵形状: {doc_matrix.shape}")
    print(f"词汇表大小: {vectorizer.get_vocabulary_size()}")
    print(f"特征名称: {vectorizer.get_feature_names()}")
    
    # 测试查询向量化
    query_tokens = ["人工智能", "技术"]
    query_vector = vectorizer.transform_query(query_tokens)
    print(f"查询向量形状: {query_vector.shape}")
    
    # 分析文档向量
    print("\n=== 文档向量分析 ===")
    for i in range(doc_matrix.shape[0]):
        doc_vector = doc_matrix[i].toarray().flatten()
        top_features = vectorizer.analyze_document_vector(doc_vector, top_k=3)
        print(f"文档 {i}: {top_features}")
    
    print("\nTF-IDF测试完成！")