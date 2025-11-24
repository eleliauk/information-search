#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple, Any, Optional
import logging
import time


class CosineRetrieval:
    """基于余弦相似度的检索器"""
    
    def __init__(self, tfidf_retrieval, documents: List[Dict[str, Any]]):
        """
        初始化余弦相似度检索器
        
        Args:
            tfidf_retrieval: TF-IDF检索器实例
            documents: 文档列表
        """
        self.tfidf_retrieval = tfidf_retrieval
        self.documents = documents
        self.logger = logging.getLogger(__name__)
        
    def cosine_similarity_manual(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        手动实现余弦相似度计算
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            余弦相似度值
        """
        # 确保向量是一维的
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        # 计算点积
        dot_product = np.dot(vec1, vec2)
        
        # 计算向量的L2范数
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # 避免除零错误
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        # 计算余弦相似度
        cosine_sim = dot_product / (norm_vec1 * norm_vec2)
        return float(cosine_sim)
    
    def batch_cosine_similarity(self, query_vector: np.ndarray, doc_matrix: csr_matrix) -> np.ndarray:
        """
        批量计算余弦相似度
        
        Args:
            query_vector: 查询向量
            doc_matrix: 文档矩阵
            
        Returns:
            相似度数组
        """
        # 使用sklearn的实现（更高效）
        similarities = cosine_similarity(query_vector, doc_matrix).flatten()
        return similarities
    
    def search(self, query_text: str, top_k: int = 10, similarity_threshold: float = 0.01, 
               use_manual: bool = False) -> List[Dict[str, Any]]:
        """
        执行余弦相似度检索
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            use_manual: 是否使用手动实现的余弦相似度
            
        Returns:
            检索结果列表
        """
        start_time = time.time()
        
        self.logger.info(f"开始检索: '{query_text}'")
        
        # 1. 查询分词
        query_tokens = self.tfidf_retrieval.tokenizer.tokenize_document(query_text, keep_pos=False)
        
        if not query_tokens:
            self.logger.warning("查询分词结果为空")
            return []
        
        # 2. 查询向量化
        query_vector = self.tfidf_retrieval.vectorizer.transform_query(query_tokens)
        
        # 3. 计算与所有文档的相似度
        if use_manual:
            # 使用手动实现（较慢但可解释）
            similarities = self._manual_batch_similarity(query_vector)
        else:
            # 使用优化实现
            similarities = self.batch_cosine_similarity(
                query_vector, 
                self.tfidf_retrieval.doc_vectors
            )
        
        # 4. 过滤低相似度结果
        valid_indices = np.where(similarities > similarity_threshold)[0]
        
        if len(valid_indices) == 0:
            self.logger.info("未找到满足阈值的相似文档")
            return []
        
        # 5. 排序并获取top-k结果
        valid_similarities = similarities[valid_indices]
        sorted_indices = valid_indices[np.argsort(valid_similarities)[::-1]]
        
        top_indices = sorted_indices[:top_k]
        
        # 6. 构造结果
        results = []
        for idx in top_indices:
            results.append({
                'document_id': int(idx),
                'similarity_score': float(similarities[idx]),
                'document': self.documents[idx],
                'title': self.documents[idx].get('title', ''),
                'content_preview': self.documents[idx].get('content', '')[:200] + '...',
                'category': self.documents[idx].get('category', '未分类'),
                'word_count': self.documents[idx].get('word_count', 0)
            })
        
        search_time = time.time() - start_time
        self.logger.info(f"检索完成，找到 {len(results)} 条结果，耗时 {search_time:.4f} 秒")
        
        return results
    
    def _manual_batch_similarity(self, query_vector: np.ndarray) -> np.ndarray:
        """手动批量计算相似度（用于演示和调试）"""
        n_docs = self.tfidf_retrieval.doc_vectors.shape[0]
        similarities = np.zeros(n_docs)
        
        query_vec_flat = query_vector.flatten()
        
        for i in range(n_docs):
            doc_vec = self.tfidf_retrieval.doc_vectors[i].toarray().flatten()
            similarities[i] = self.cosine_similarity_manual(query_vec_flat, doc_vec)
        
        return similarities
    
    def explain_similarity(self, query_text: str, document_id: int) -> Dict[str, Any]:
        """
        详细解释相似度计算过程
        
        Args:
            query_text: 查询文本
            document_id: 文档ID
            
        Returns:
            详细的相似度解释
        """
        if document_id >= len(self.documents):
            raise ValueError("文档ID超出范围")
        
        # 获取查询向量
        query_tokens = self.tfidf_retrieval.tokenizer.tokenize_document(query_text, keep_pos=False)
        query_vector = self.tfidf_retrieval.vectorizer.transform_query(query_tokens)
        
        # 获取文档向量
        doc_vector = self.tfidf_retrieval.doc_vectors[document_id].toarray()
        
        # 分析查询中的词汇
        query_terms = []
        for token in query_tokens:
            word_idx = self.tfidf_retrieval.vectorizer.get_word_index(token)
            if word_idx is not None:
                weight = query_vector[0, word_idx]
                idf = self.tfidf_retrieval.vectorizer.get_word_idf(token)
                query_terms.append({
                    'word': token,
                    'index': word_idx,
                    'tfidf_weight': float(weight),
                    'idf': float(idf) if idf else 0
                })
        
        # 分析文档中的词汇
        doc_terms = []
        doc_vec_flat = doc_vector.flatten()
        nonzero_indices = np.nonzero(doc_vec_flat)[0]
        
        for idx in nonzero_indices:
            word = self.tfidf_retrieval.vectorizer.feature_names[idx]
            weight = doc_vec_flat[idx]
            idf = self.tfidf_retrieval.vectorizer.get_word_idf(word)
            doc_terms.append({
                'word': word,
                'index': idx,
                'tfidf_weight': float(weight),
                'idf': float(idf) if idf else 0
            })
        
        # 找出共同词汇及其贡献
        query_words = {term['word']: term for term in query_terms}
        doc_words = {term['word']: term for term in doc_terms}
        
        common_terms = []
        total_contribution = 0
        
        for word in query_words:
            if word in doc_words:
                q_weight = query_words[word]['tfidf_weight']
                d_weight = doc_words[word]['tfidf_weight']
                contribution = q_weight * d_weight
                total_contribution += contribution
                
                common_terms.append({
                    'word': word,
                    'query_weight': q_weight,
                    'doc_weight': d_weight,
                    'contribution': contribution,
                    'idf': query_words[word]['idf']
                })
        
        # 按贡献度排序
        common_terms.sort(key=lambda x: x['contribution'], reverse=True)
        
        # 计算最终相似度
        similarity = self.cosine_similarity_manual(query_vector, doc_vector)
        
        # 计算向量范数
        query_norm = np.linalg.norm(query_vector)
        doc_norm = np.linalg.norm(doc_vector)
        
        return {
            "查询": query_text,
            "查询分词": query_tokens,
            "文档ID": document_id,
            "文档标题": self.documents[document_id].get('title', ''),
            "相似度计算": {
                "余弦相似度": float(similarity),
                "点积": float(total_contribution),
                "查询向量范数": float(query_norm),
                "文档向量范数": float(doc_norm)
            },
            "查询词汇": query_terms,
            "文档关键词": sorted(doc_terms, key=lambda x: x['tfidf_weight'], reverse=True)[:10],
            "共同词汇": common_terms,
            "匹配统计": {
                "查询词汇数": len(query_tokens),
                "文档非零特征数": len(doc_terms),
                "共同词汇数": len(common_terms),
                "匹配率": f"{len(common_terms)/len(query_tokens)*100:.1f}%" if query_tokens else "0%"
            }
        }
    
    def compare_similarity_methods(self, query_text: str, document_id: int) -> Dict[str, Any]:
        """
        比较不同相似度计算方法的结果
        
        Args:
            query_text: 查询文本
            document_id: 文档ID
            
        Returns:
            不同方法的比较结果
        """
        if document_id >= len(self.documents):
            raise ValueError("文档ID超出范围")
        
        # 获取向量
        query_tokens = self.tfidf_retrieval.tokenizer.tokenize_document(query_text, keep_pos=False)
        query_vector = self.tfidf_retrieval.vectorizer.transform_query(query_tokens)
        doc_vector = self.tfidf_retrieval.doc_vectors[document_id].toarray()
        
        # 不同相似度计算方法
        results = {}
        
        # 1. 手动实现的余弦相似度
        start_time = time.time()
        manual_cosine = self.cosine_similarity_manual(query_vector, doc_vector)
        manual_time = time.time() - start_time
        
        # 2. sklearn的余弦相似度
        start_time = time.time()
        sklearn_cosine = cosine_similarity(query_vector, doc_vector)[0, 0]
        sklearn_time = time.time() - start_time
        
        # 3. 点积相似度（向量已归一化）
        start_time = time.time()
        dot_product = np.dot(query_vector.flatten(), doc_vector.flatten())
        dot_time = time.time() - start_time
        
        # 4. 欧几里得距离（转换为相似度）
        start_time = time.time()
        euclidean_dist = np.linalg.norm(query_vector.flatten() - doc_vector.flatten())
        euclidean_sim = 1 / (1 + euclidean_dist)  # 转换为相似度
        euclidean_time = time.time() - start_time
        
        return {
            "查询": query_text,
            "文档ID": document_id,
            "文档标题": self.documents[document_id].get('title', ''),
            "相似度方法比较": {
                "手动余弦相似度": {
                    "值": float(manual_cosine),
                    "计算时间": f"{manual_time*1000:.4f} ms"
                },
                "sklearn余弦相似度": {
                    "值": float(sklearn_cosine),
                    "计算时间": f"{sklearn_time*1000:.4f} ms"
                },
                "点积相似度": {
                    "值": float(dot_product),
                    "计算时间": f"{dot_time*1000:.4f} ms"
                },
                "欧几里得相似度": {
                    "值": float(euclidean_sim),
                    "计算时间": f"{euclidean_time*1000:.4f} ms"
                }
            },
            "一致性检查": {
                "手动vs sklearn差异": abs(manual_cosine - sklearn_cosine),
                "归一化向量点积等于余弦": abs(dot_product - sklearn_cosine) < 1e-10
            }
        }
    
    def get_similarity_distribution(self, query_text: str, num_bins: int = 10) -> Dict[str, Any]:
        """
        分析查询结果的相似度分布
        
        Args:
            query_text: 查询文本
            num_bins: 分布区间数
            
        Returns:
            相似度分布统计
        """
        # 计算所有文档的相似度
        query_tokens = self.tfidf_retrieval.tokenizer.tokenize_document(query_text, keep_pos=False)
        query_vector = self.tfidf_retrieval.vectorizer.transform_query(query_tokens)
        
        similarities = self.batch_cosine_similarity(
            query_vector, 
            self.tfidf_retrieval.doc_vectors
        )
        
        # 计算统计信息
        stats = {
            "查询": query_text,
            "总文档数": len(similarities),
            "相似度统计": {
                "最大值": float(np.max(similarities)),
                "最小值": float(np.min(similarities)),
                "平均值": float(np.mean(similarities)),
                "中位数": float(np.median(similarities)),
                "标准差": float(np.std(similarities))
            },
            "分布区间": {}
        }
        
        # 创建分布区间
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        bin_edges = np.linspace(min_sim, max_sim, num_bins + 1)
        
        for i in range(num_bins):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            count = np.sum((similarities >= bin_start) & (similarities < bin_end))
            
            if i == num_bins - 1:  # 最后一个区间包含最大值
                count = np.sum((similarities >= bin_start) & (similarities <= bin_end))
            
            stats["分布区间"][f"{bin_start:.3f}-{bin_end:.3f}"] = int(count)
        
        # 高相似度文档统计
        high_sim_docs = np.sum(similarities > 0.1)
        medium_sim_docs = np.sum((similarities > 0.01) & (similarities <= 0.1))
        low_sim_docs = np.sum(similarities <= 0.01)
        
        stats["相似度级别"] = {
            "高相似度(>0.1)": int(high_sim_docs),
            "中等相似度(0.01-0.1)": int(medium_sim_docs),
            "低相似度(<=0.01)": int(low_sim_docs)
        }
        
        return stats


class SimilarityAnalyzer:
    """相似度分析工具"""
    
    def __init__(self, cosine_retrieval: CosineRetrieval):
        self.cosine_retrieval = cosine_retrieval
        self.logger = logging.getLogger(__name__)
    
    def analyze_query_performance(self, queries: List[str], sample_size: int = 100) -> Dict[str, Any]:
        """
        分析查询性能
        
        Args:
            queries: 查询列表
            sample_size: 抽样大小
            
        Returns:
            性能分析结果
        """
        results = {
            "查询数量": len(queries),
            "抽样大小": sample_size,
            "性能统计": {
                "平均检索时间": [],
                "平均结果数量": [],
                "平均最高相似度": []
            },
            "查询分析": []
        }
        
        for query in queries:
            start_time = time.time()
            
            # 执行检索
            search_results = self.cosine_retrieval.search(query, top_k=10)
            
            search_time = time.time() - start_time
            
            # 收集统计信息
            results["性能统计"]["平均检索时间"].append(search_time)
            results["性能统计"]["平均结果数量"].append(len(search_results))
            
            if search_results:
                max_similarity = max(r['similarity_score'] for r in search_results)
                results["性能统计"]["平均最高相似度"].append(max_similarity)
            else:
                results["性能统计"]["平均最高相似度"].append(0)
            
            # 详细分析
            query_analysis = {
                "查询": query,
                "检索时间": search_time,
                "结果数量": len(search_results),
                "最高相似度": max([r['similarity_score'] for r in search_results]) if search_results else 0,
                "平均相似度": np.mean([r['similarity_score'] for r in search_results]) if search_results else 0
            }
            
            results["查询分析"].append(query_analysis)
        
        # 计算总体统计
        perf_stats = results["性能统计"]
        results["总体性能"] = {
            "平均检索时间": f"{np.mean(perf_stats['平均检索时间']):.4f} 秒",
            "检索时间标准差": f"{np.std(perf_stats['平均检索时间']):.4f} 秒",
            "平均结果数量": f"{np.mean(perf_stats['平均结果数量']):.2f}",
            "平均最高相似度": f"{np.mean(perf_stats['平均最高相似度']):.4f}",
            "无结果查询比例": f"{len([q for q in results['查询分析'] if q['结果数量'] == 0])/len(queries)*100:.1f}%"
        }
        
        return results
    
    def compare_retrieval_methods(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        比较不同检索方法的结果
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            方法比较结果
        """
        results = {}
        
        # 1. 使用sklearn的余弦相似度
        start_time = time.time()
        sklearn_results = self.cosine_retrieval.search(query, top_k=top_k, use_manual=False)
        sklearn_time = time.time() - start_time
        
        # 2. 使用手动实现的余弦相似度
        start_time = time.time()
        manual_results = self.cosine_retrieval.search(query, top_k=top_k, use_manual=True)
        manual_time = time.time() - start_time
        
        return {
            "查询": query,
            "sklearn方法": {
                "检索时间": f"{sklearn_time:.4f} 秒",
                "结果数量": len(sklearn_results),
                "结果": [{"文档ID": r['document_id'], "相似度": r['similarity_score'], "标题": r['title']} 
                        for r in sklearn_results]
            },
            "手动实现方法": {
                "检索时间": f"{manual_time:.4f} 秒",
                "结果数量": len(manual_results),
                "结果": [{"文档ID": r['document_id'], "相似度": r['similarity_score'], "标题": r['title']} 
                        for r in manual_results]
            },
            "性能比较": {
                "速度提升": f"{manual_time/sklearn_time:.2f}x (sklearn更快)" if sklearn_time < manual_time else f"{sklearn_time/manual_time:.2f}x (手动更快)",
                "结果一致性": len(sklearn_results) == len(manual_results) and 
                             all(s['document_id'] == m['document_id'] for s, m in zip(sklearn_results, manual_results))
            }
        }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=== 余弦相似度模块测试 ===")
    
    # 创建测试向量
    vec1 = np.array([1, 2, 3, 4])
    vec2 = np.array([2, 4, 6, 8])
    vec3 = np.array([1, 0, 0, 0])
    
    # 测试相似度计算
    retrieval = CosineRetrieval(None, [])
    
    sim1 = retrieval.cosine_similarity_manual(vec1, vec2)
    sim2 = retrieval.cosine_similarity_manual(vec1, vec3)
    
    print(f"向量1和向量2的余弦相似度: {sim1:.4f}")
    print(f"向量1和向量3的余弦相似度: {sim2:.4f}")
    
    # 与sklearn结果对比
    sklearn_sim1 = cosine_similarity([vec1], [vec2])[0, 0]
    sklearn_sim2 = cosine_similarity([vec1], [vec3])[0, 0]
    
    print(f"sklearn结果1: {sklearn_sim1:.4f}, 差异: {abs(sim1 - sklearn_sim1):.6f}")
    print(f"sklearn结果2: {sklearn_sim2:.4f}, 差异: {abs(sim2 - sklearn_sim2):.6f}")
    
    print("\n余弦相似度测试完成！")