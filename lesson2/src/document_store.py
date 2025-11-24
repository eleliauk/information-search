"""
文档存储组件
负责文档的存储和管理
"""
from typing import Optional, List, Dict
from .models import Document


class DocumentStore:
    """
    文档存储类
    
    职责：
    - 为每个文档分配唯一标识符
    - 存储文档内容和元数据
    - 提供文档检索功能
    - 验证文档有效性
    """
    
    def __init__(self):
        """初始化文档存储"""
        self._documents: Dict[int, Document] = {}
        self._next_doc_id: int = 1
    
    def add_document(self, content: str, metadata: dict = None) -> int:
        """
        添加文档到存储
        
        Args:
            content: 文档内容
            metadata: 文档元数据（可选）
        
        Returns:
            分配的文档ID
        
        Raises:
            ValueError: 如果文档内容为空或只包含空白字符
        """
        # 验证文档有效性：拒绝空文档
        if not content or content.strip() == "":
            raise ValueError("文档内容不能为空")
        
        # 分配唯一的文档ID
        doc_id = self._next_doc_id
        self._next_doc_id += 1
        
        # 创建文档对象
        if metadata is None:
            metadata = {}
        
        document = Document(
            doc_id=doc_id,
            content=content,
            metadata=metadata
        )
        
        # 存储文档
        self._documents[doc_id] = document
        
        return doc_id
    
    def get_document(self, doc_id: int) -> Optional[Document]:
        """
        根据文档ID获取文档
        
        Args:
            doc_id: 文档ID
        
        Returns:
            文档对象，如果不存在则返回 None
        """
        return self._documents.get(doc_id)
    
    def get_all_documents(self) -> List[Document]:
        """
        获取所有文档
        
        Returns:
            所有文档的列表
        """
        return list(self._documents.values())
    
    def document_count(self) -> int:
        """
        获取文档总数
        
        Returns:
            文档数量
        """
        return len(self._documents)
