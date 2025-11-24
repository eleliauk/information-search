"""
倒排索引组件
负责索引的构建和维护
"""
from typing import Dict, List, Optional
from collections import defaultdict
from .models import Posting, IndexStatistics
import pickle


class InvertedIndex:
    """
    倒排索引类
    
    职责：
    - 构建和更新倒排索引
    - 维护词项到文档的映射
    - 记录词项位置信息
    - 提供索引查询接口
    
    数据结构：
    - index: Dict[str, List[Posting]] - 词项到倒排列表的映射
    """
    
    def __init__(self):
        """初始化倒排索引"""
        # 倒排索引：词项 -> 倒排列表
        self._index: Dict[str, List[Posting]] = {}
        
        # 文档长度记录：doc_id -> token_count
        self._doc_lengths: Dict[int, int] = {}
    
    def build_index(self, doc_id: int, tokens: List[str]) -> None:
        """
        为文档构建倒排索引
        
        处理流程：
        1. 遍历文档中的所有词项
        2. 记录每个词项的位置信息
        3. 更新或创建倒排列表
        4. 记录文档长度
        
        Args:
            doc_id: 文档ID
            tokens: 文档的词项列表（已经过预处理）
        """
        # 记录文档长度
        self._doc_lengths[doc_id] = len(tokens)
        
        # 用于记录每个词项在当前文档中的位置
        term_positions: Dict[str, List[int]] = defaultdict(list)
        
        # 遍历词项，记录位置
        for position, term in enumerate(tokens):
            term_positions[term].append(position)
        
        # 更新倒排索引
        for term, positions in term_positions.items():
            # 如果词项不存在，创建新的倒排列表
            if term not in self._index:
                self._index[term] = []
            
            # 创建 Posting 对象
            posting = Posting(
                doc_id=doc_id,
                positions=positions,
                term_freq=len(positions)
            )
            
            # 添加到倒排列表
            self._index[term].append(posting)
    
    def get_posting_list(self, term: str) -> List[Posting]:
        """
        获取词项的倒排列表
        
        Args:
            term: 查询词项
        
        Returns:
            倒排列表，如果词项不存在则返回空列表
        """
        return self._index.get(term, [])
    
    def get_term_frequency(self, term: str) -> int:
        """
        获取词项的文档频率（包含该词项的文档数量）
        
        Args:
            term: 查询词项
        
        Returns:
            文档频率（DF）
        """
        posting_list = self.get_posting_list(term)
        return len(posting_list)
    
    def get_vocabulary_size(self) -> int:
        """
        获取词汇表大小（唯一词项数量）
        
        Returns:
            唯一词项的数量
        """
        return len(self._index)
    
    def get_document_length(self, doc_id: int) -> int:
        """
        获取文档长度（词项数量）
        
        Args:
            doc_id: 文档ID
        
        Returns:
            文档长度，如果文档不存在则返回0
        """
        return self._doc_lengths.get(doc_id, 0)
    
    def get_all_terms(self) -> List[str]:
        """
        获取所有词项
        
        Returns:
            所有词项的列表
        """
        return list(self._index.keys())
    
    def save(self, filepath: str) -> None:
        """
        将索引保存到磁盘
        
        Args:
            filepath: 保存路径
        
        Raises:
            IOError: 文件写入错误
        """
        try:
            with open(filepath, 'wb') as f:
                # 保存索引和文档长度信息
                data = {
                    'index': self._index,
                    'doc_lengths': self._doc_lengths
                }
                pickle.dump(data, f)
        except Exception as e:
            raise IOError(f"保存索引失败: {str(e)}")
    
    def load(self, filepath: str) -> None:
        """
        从磁盘加载索引
        
        Args:
            filepath: 索引文件路径
        
        Raises:
            FileNotFoundError: 文件不存在
            IOError: 文件读取或反序列化错误
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self._index = data.get('index', {})
                self._doc_lengths = data.get('doc_lengths', {})
        except FileNotFoundError:
            raise FileNotFoundError(f"索引文件不存在: {filepath}")
        except Exception as e:
            raise IOError(f"加载索引失败: {str(e)}")
    
    def get_statistics(self) -> IndexStatistics:
        """
        获取索引统计信息
        
        返回索引的各项统计数据，包括：
        - 文档总数
        - 唯一词项数（词汇表大小）
        - 总词项数
        - 平均文档长度
        - 词项文档频率
        
        Returns:
            IndexStatistics 对象，包含所有统计信息
        """
        # 1. 文档总数：从文档长度字典中获取
        document_count = len(self._doc_lengths)
        
        # 2. 唯一词项数：索引中的词项数量
        vocabulary_size = len(self._index)
        
        # 3. 总词项数：所有文档的词项总和
        total_tokens = sum(self._doc_lengths.values())
        
        # 4. 平均文档长度
        avg_doc_length = total_tokens / document_count if document_count > 0 else 0.0
        
        # 5. 词项文档频率：每个词项出现在多少个文档中
        term_frequencies = {}
        for term, posting_list in self._index.items():
            term_frequencies[term] = len(posting_list)
        
        return IndexStatistics(
            document_count=document_count,
            vocabulary_size=vocabulary_size,
            total_tokens=total_tokens,
            avg_doc_length=avg_doc_length,
            term_frequencies=term_frequencies
        )
