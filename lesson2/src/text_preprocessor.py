"""
文本预处理组件
负责文本的分词和预处理
"""
import jieba
import re
import string
from typing import List, Set, Optional
from pathlib import Path


class TextPreprocessor:
    """
    文本预处理类
    
    职责：
    - 中文分词（使用 jieba）
    - 停用词过滤
    - 文本规范化（小写转换、标点处理）
    - 提供一致的预处理流程
    """
    
    def __init__(self, stopwords_path: Optional[str] = None):
        """
        初始化文本预处理器
        
        Args:
            stopwords_path: 停用词文件路径（可选）
        """
        self._stopwords: Set[str] = set()
        
        # 如果提供了停用词路径，加载停用词
        if stopwords_path:
            self.load_stopwords(stopwords_path)
    
    def load_stopwords(self, path: str) -> None:
        """
        从文件加载停用词
        
        Args:
            path: 停用词文件路径，每行一个停用词
        """
        stopwords_file = Path(path)
        
        if not stopwords_file.exists():
            raise FileNotFoundError(f"停用词文件不存在: {path}")
        
        self._stopwords.clear()
        
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:  # 忽略空行
                    self._stopwords.add(word)
    
    def preprocess(self, text: str) -> str:
        """
        预处理文本（规范化）
        
        Args:
            text: 原始文本
        
        Returns:
            规范化后的文本
        """
        if not text:
            return ""
        
        # 转换为小写（对英文字母）
        text = text.lower()
        
        # 移除标点符号（保留中文字符、英文字母、数字和空格）
        # 使用正则表达式移除标点符号
        # 保留中文字符 (\u4e00-\u9fff)、英文字母、数字和空格
        text = re.sub(r'[^\u4e00-\u9fffa-z0-9\s]', '', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词和预处理
        
        处理流程：
        1. 文本规范化（小写转换、标点处理）
        2. 使用 jieba 进行中文分词
        3. 过滤停用词
        4. 过滤空白词项
        
        Args:
            text: 原始文本
        
        Returns:
            处理后的词项列表
        """
        if not text:
            return []
        
        # 1. 文本规范化
        normalized_text = self.preprocess(text)
        
        if not normalized_text:
            return []
        
        # 2. 使用 jieba 进行分词
        tokens = jieba.lcut(normalized_text)
        
        # 3. 过滤停用词和空白词项
        filtered_tokens = [
            token.strip() 
            for token in tokens 
            if token.strip() and token.strip() not in self._stopwords
        ]
        
        return filtered_tokens
