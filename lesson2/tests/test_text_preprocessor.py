"""
文本预处理器测试
"""
import pytest
from pathlib import Path
from src.text_preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """TextPreprocessor 单元测试"""
    
    def test_initialization_without_stopwords(self):
        """测试不带停用词的初始化"""
        preprocessor = TextPreprocessor()
        assert preprocessor._stopwords == set()
    
    def test_initialization_with_stopwords(self):
        """测试带停用词的初始化"""
        stopwords_path = "lesson2/config/stopwords.txt"
        preprocessor = TextPreprocessor(stopwords_path)
        assert len(preprocessor._stopwords) > 0
        assert "的" in preprocessor._stopwords
    
    def test_load_stopwords(self):
        """测试加载停用词"""
        preprocessor = TextPreprocessor()
        stopwords_path = "lesson2/config/stopwords.txt"
        preprocessor.load_stopwords(stopwords_path)
        assert len(preprocessor._stopwords) > 0
        assert "的" in preprocessor._stopwords
    
    def test_load_stopwords_file_not_found(self):
        """测试加载不存在的停用词文件"""
        preprocessor = TextPreprocessor()
        with pytest.raises(FileNotFoundError):
            preprocessor.load_stopwords("nonexistent_file.txt")
    
    def test_preprocess_lowercase(self):
        """测试小写转换"""
        preprocessor = TextPreprocessor()
        text = "Hello World"
        result = preprocessor.preprocess(text)
        assert result == "hello world"
    
    def test_preprocess_punctuation_removal(self):
        """测试标点符号移除"""
        preprocessor = TextPreprocessor()
        text = "你好，世界！这是一个测试。"
        result = preprocessor.preprocess(text)
        # 标点符号应该被移除
        assert "，" not in result
        assert "！" not in result
        assert "。" not in result
        # 中文字符应该保留
        assert "你好" in result
        assert "世界" in result
    
    def test_preprocess_empty_text(self):
        """测试空文本处理"""
        preprocessor = TextPreprocessor()
        assert preprocessor.preprocess("") == ""
        assert preprocessor.preprocess(None) == ""
    
    def test_tokenize_basic(self):
        """测试基本分词功能"""
        preprocessor = TextPreprocessor()
        text = "我爱北京天安门"
        tokens = preprocessor.tokenize(text)
        assert len(tokens) > 0
        assert isinstance(tokens, list)
    
    def test_tokenize_with_stopwords(self):
        """测试带停用词过滤的分词"""
        stopwords_path = "lesson2/config/stopwords.txt"
        preprocessor = TextPreprocessor(stopwords_path)
        text = "我爱北京天安门"
        tokens = preprocessor.tokenize(text)
        # "的" 是停用词，不应该出现在结果中
        assert "的" not in tokens
    
    def test_tokenize_empty_text(self):
        """测试空文本分词"""
        preprocessor = TextPreprocessor()
        assert preprocessor.tokenize("") == []
        assert preprocessor.tokenize(None) == []
    
    def test_tokenize_with_punctuation(self):
        """测试包含标点的文本分词"""
        preprocessor = TextPreprocessor()
        text = "你好，世界！"
        tokens = preprocessor.tokenize(text)
        # 标点符号不应该出现在词项中
        for token in tokens:
            assert "，" not in token
            assert "！" not in token
    
    def test_tokenize_mixed_chinese_english(self):
        """测试中英文混合文本分词"""
        preprocessor = TextPreprocessor()
        text = "我喜欢Python编程"
        tokens = preprocessor.tokenize(text)
        assert len(tokens) > 0
        # 英文应该被转换为小写
        assert any("python" in token.lower() for token in tokens)
    
    def test_tokenize_filters_whitespace(self):
        """测试过滤空白词项"""
        preprocessor = TextPreprocessor()
        text = "测试   文本"
        tokens = preprocessor.tokenize(text)
        # 不应该有空白词项
        assert "" not in tokens
        assert all(token.strip() for token in tokens)
