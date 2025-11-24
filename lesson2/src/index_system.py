"""
索引系统主入口
集成所有组件，提供统一的接口
"""
import logging
from typing import Optional, List, Dict
from pathlib import Path

from .document_store import DocumentStore
from .text_preprocessor import TextPreprocessor
from .inverted_index import InvertedIndex
from .query_processor import QueryProcessor
from .models import IndexStatistics


class IndexSystem:
    """
    索引系统主入口类
    
    职责：
    - 集成所有组件（文档存储、文本预处理、倒排索引、查询处理）
    - 提供统一的对外接口
    - 协调各组件之间的交互
    - 提供错误处理和日志记录
    """
    
    def __init__(self, stopwords_path: Optional[str] = None, log_level: int = logging.INFO):
        """
        初始化索引系统
        
        Args:
            stopwords_path: 停用词文件路径（可选）
            log_level: 日志级别（默认为 INFO）
        """
        # 设置日志
        self._setup_logging(log_level)
        self._logger = logging.getLogger(__name__)
        
        # 初始化各个组件
        self._logger.info("初始化索引系统...")
        
        try:
            # 1. 文档存储
            self._doc_store = DocumentStore()
            self._logger.debug("文档存储组件初始化完成")
            
            # 2. 文本预处理器
            self._preprocessor = TextPreprocessor(stopwords_path)
            self._logger.debug("文本预处理器初始化完成")
            
            # 3. 倒排索引
            self._index = InvertedIndex()
            self._logger.debug("倒排索引组件初始化完成")
            
            # 4. 查询处理器
            self._query_processor = QueryProcessor(
                self._index,
                self._preprocessor,
                self._doc_store
            )
            self._logger.debug("查询处理器初始化完成")
            
            self._logger.info("索引系统初始化成功")
            
        except Exception as e:
            self._logger.error(f"索引系统初始化失败: {str(e)}")
            raise
    
    def _setup_logging(self, log_level: int) -> None:
        """
        设置日志配置
        
        Args:
            log_level: 日志级别
        """
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def add_document(self, content: str, metadata: Optional[dict] = None) -> int:
        """
        添加文档到索引系统
        
        协调文档存储和索引构建：
        1. 验证文档内容
        2. 添加文档到文档存储
        3. 对文档进行分词预处理
        4. 构建倒排索引
        5. 更新文档的词项计数
        
        Args:
            content: 文档内容
            metadata: 文档元数据（可选）
        
        Returns:
            分配的文档ID
        
        Raises:
            ValueError: 如果文档内容为空
            Exception: 其他处理错误
        """
        try:
            self._logger.info(f"添加文档: 内容长度={len(content)}")
            
            # 1. 添加文档到文档存储（会自动验证文档有效性）
            doc_id = self._doc_store.add_document(content, metadata)
            self._logger.debug(f"文档已存储，分配ID: {doc_id}")
            
            # 2. 对文档进行分词预处理
            tokens = self._preprocessor.tokenize(content)
            self._logger.debug(f"文档分词完成，词项数: {len(tokens)}")
            
            # 3. 构建倒排索引
            self._index.build_index(doc_id, tokens)
            self._logger.debug(f"文档 {doc_id} 的倒排索引构建完成")
            
            # 4. 更新文档的词项计数
            document = self._doc_store.get_document(doc_id)
            if document:
                document.token_count = len(tokens)
            
            self._logger.info(f"文档添加成功，ID: {doc_id}")
            return doc_id
            
        except ValueError as e:
            # 文档验证错误（空文档等）
            self._logger.warning(f"文档验证失败: {str(e)}")
            raise
        except Exception as e:
            # 其他错误
            self._logger.error(f"添加文档时发生错误: {str(e)}")
            raise Exception(f"添加文档失败: {str(e)}")
    
    def search(self, query: str) -> List[int]:
        """
        执行单词项查询
        
        Args:
            query: 查询字符串
        
        Returns:
            包含查询词的文档ID列表（按升序排列）
        
        Raises:
            Exception: 查询处理错误
        """
        try:
            self._logger.info(f"执行查询: {query}")
            
            if not query or not query.strip():
                self._logger.warning("查询字符串为空")
                return []
            
            results = self._query_processor.search(query)
            self._logger.info(f"查询完成，找到 {len(results)} 个文档")
            
            return results
            
        except Exception as e:
            self._logger.error(f"查询处理错误: {str(e)}")
            raise Exception(f"查询失败: {str(e)}")
    
    def boolean_search(self, query: str) -> List[int]:
        """
        执行布尔查询
        
        支持的运算符：AND, OR, NOT
        运算符优先级：NOT > AND > OR
        
        Args:
            query: 布尔查询字符串
        
        Returns:
            符合查询条件的文档ID列表（按升序排列）
        
        Raises:
            Exception: 查询处理错误
        """
        try:
            self._logger.info(f"执行布尔查询: {query}")
            
            if not query or not query.strip():
                self._logger.warning("查询字符串为空")
                return []
            
            results = self._query_processor.boolean_search(query)
            self._logger.info(f"布尔查询完成，找到 {len(results)} 个文档")
            
            return results
            
        except Exception as e:
            self._logger.error(f"布尔查询处理错误: {str(e)}")
            raise Exception(f"布尔查询失败: {str(e)}")
    
    def get_statistics(self) -> Dict:
        """
        获取索引统计信息
        
        Returns:
            包含统计信息的字典：
            - document_count: 文档总数
            - vocabulary_size: 唯一词项数
            - total_tokens: 总词项数
            - avg_doc_length: 平均文档长度
            - term_frequencies: 词项文档频率（可选，数据量大时可能省略）
        
        Raises:
            Exception: 统计信息获取错误
        """
        try:
            self._logger.info("获取索引统计信息")
            
            stats: IndexStatistics = self._index.get_statistics()
            
            # 转换为字典格式
            stats_dict = {
                'document_count': stats.document_count,
                'vocabulary_size': stats.vocabulary_size,
                'total_tokens': stats.total_tokens,
                'avg_doc_length': stats.avg_doc_length,
                'term_frequencies': stats.term_frequencies
            }
            
            self._logger.info(f"统计信息: 文档数={stats.document_count}, "
                            f"词汇表大小={stats.vocabulary_size}")
            
            return stats_dict
            
        except Exception as e:
            self._logger.error(f"获取统计信息错误: {str(e)}")
            raise Exception(f"获取统计信息失败: {str(e)}")
    
    def save_index(self, filepath: str) -> bool:
        """
        保存索引到磁盘
        
        保存内容包括：
        - 倒排索引数据
        - 文档存储数据
        
        Args:
            filepath: 保存路径
        
        Returns:
            True 表示保存成功，False 表示保存失败
        """
        try:
            self._logger.info(f"保存索引到: {filepath}")
            
            # 确保目录存在
            file_path = Path(filepath)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存倒排索引
            index_file = str(file_path.with_suffix('.index'))
            self._index.save(index_file)
            self._logger.debug(f"倒排索引已保存到: {index_file}")
            
            # 保存文档存储
            import pickle
            doc_store_file = str(file_path.with_suffix('.docs'))
            with open(doc_store_file, 'wb') as f:
                pickle.dump(self._doc_store, f)
            self._logger.debug(f"文档存储已保存到: {doc_store_file}")
            
            self._logger.info("索引保存成功")
            return True
            
        except Exception as e:
            self._logger.error(f"保存索引失败: {str(e)}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """
        从磁盘加载索引
        
        加载内容包括：
        - 倒排索引数据
        - 文档存储数据
        
        Args:
            filepath: 索引文件路径
        
        Returns:
            True 表示加载成功，False 表示加载失败
        """
        try:
            self._logger.info(f"从文件加载索引: {filepath}")
            
            file_path = Path(filepath)
            
            # 加载倒排索引
            index_file = str(file_path.with_suffix('.index'))
            if not Path(index_file).exists():
                self._logger.error(f"索引文件不存在: {index_file}")
                return False
            
            self._index.load(index_file)
            self._logger.debug(f"倒排索引已加载: {index_file}")
            
            # 加载文档存储
            import pickle
            doc_store_file = str(file_path.with_suffix('.docs'))
            if not Path(doc_store_file).exists():
                self._logger.error(f"文档存储文件不存在: {doc_store_file}")
                return False
            
            with open(doc_store_file, 'rb') as f:
                self._doc_store = pickle.load(f)
            self._logger.debug(f"文档存储已加载: {doc_store_file}")
            
            # 重新初始化查询处理器（因为文档存储已更新）
            self._query_processor = QueryProcessor(
                self._index,
                self._preprocessor,
                self._doc_store
            )
            
            self._logger.info("索引加载成功")
            return True
            
        except FileNotFoundError as e:
            self._logger.error(f"文件不存在: {str(e)}")
            return False
        except Exception as e:
            self._logger.error(f"加载索引失败: {str(e)}")
            return False
    
    def get_document(self, doc_id: int) -> Optional[Dict]:
        """
        获取文档内容
        
        Args:
            doc_id: 文档ID
        
        Returns:
            文档信息字典，如果文档不存在则返回 None
        """
        try:
            document = self._doc_store.get_document(doc_id)
            
            if document is None:
                return None
            
            return {
                'doc_id': document.doc_id,
                'content': document.content,
                'metadata': document.metadata,
                'token_count': document.token_count,
                'created_at': document.created_at.isoformat()
            }
            
        except Exception as e:
            self._logger.error(f"获取文档错误: {str(e)}")
            return None
