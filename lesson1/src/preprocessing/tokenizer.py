#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jieba
import jieba.posseg as pseg
import re
import string
import time
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
from pathlib import Path
import numpy as np

class OptimizedChineseTokenizer:
    """优化的中文分词器"""
    
    def __init__(self, custom_dict_path: Optional[str] = None, stopwords_path: Optional[str] = None):
        """
        初始化分词器
        
        Args:
            custom_dict_path: 自定义词典路径
            stopwords_path: 停用词文件路径
        """
        self.logger = logging.getLogger(__name__)
        
        # 设置结巴分词模式
        jieba.enable_parallel(4)  # 并行分词
        
        # 加载自定义词典
        if custom_dict_path and Path(custom_dict_path).exists():
            jieba.load_userdict(custom_dict_path)
            self.logger.info(f"已加载自定义词典: {custom_dict_path}")
        else:
            self._load_default_custom_dict()
        
        # 加载停用词
        self.stopwords = self._load_stopwords(stopwords_path)
        
        # 词性过滤规则
        self.keep_pos = {
            'n', 'nr', 'ns', 'nt', 'nw', 'nz',  # 名词类
            'v', 'vd', 'vn',                      # 动词类
            'a', 'ad', 'an',                      # 形容词类
            'i', 'j', 'l'                         # 成语、简称、习用语
        }
        
        # 统计信息
        self.tokenization_stats = {
            "total_documents": 0,
            "total_tokens": 0,
            "unique_tokens": set(),
            "token_frequencies": Counter(),
            "pos_distribution": Counter(),
            "token_length_distribution": Counter(),
            "oov_tokens": set()  # 未登录词
        }
        
        self.logger.info("中文分词器初始化完成")
    
    def _load_default_custom_dict(self):
        """加载默认自定义词典"""
        # 新闻领域专用词典
        news_dict = [
            "人工智能", "机器学习", "深度学习", "神经网络", "大数据", "云计算",
            "物联网", "区块链", "5G网络", "智能制造", "新冠肺炎", "新冠病毒",
            "疫情防控", "核酸检测", "疫苗接种", "隔离管控", "健康码", "方舱医院",
            "碳达峰", "碳中和", "碳排放", "新能源", "可再生能源", "节能减排",
            "绿色发展", "生态环境", "数字经济", "共享经济", "平台经济",
            "供给侧改革", "高质量发展", "双循环", "自贸区", "营商环境",
            "全面小康", "脱贫攻坚", "乡村振兴", "共同富裕", "治理体系",
            "治理能力", "法治建设", "一带一路", "人类命运共同体", "多边主义",
            "全球化", "贸易保护主义", "地缘政治"
        ]
        
        # 动态添加词汇到jieba词典
        for word in news_dict:
            jieba.add_word(word, freq=10, tag='n')
        
        self.logger.info("已加载默认新闻词典")
    
    def _load_stopwords(self, stopwords_path: Optional[str] = None) -> set:
        """加载停用词表"""
        stopwords = set()
        
        # 基础停用词
        basic_stopwords = [
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", 
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "那", 
            "什么", "时候", "如果", "但是", "因为", "所以", "而且", "然后",
            "这样", "那样", "可以", "应该", "需要", "能够", "已经", "还是",
            "或者", "以及", "对于", "关于", "通过", "根据", "按照", "由于",
            "为了", "虽然", "尽管", "除了", "另外", "同时", "首先", "其次",
            "最后", "总之", "因此", "然而", "不过", "此外", "并且"
        ]
        
        # 时间词
        time_words = [
            "今天", "昨天", "明天", "现在", "以前", "以后", "最近", "目前",
            "当前", "过去", "未来", "今年", "去年", "明年", "本月", "上月", "下月"
        ]
        
        # 数量词
        quantity_words = [
            "一些", "很多", "少数", "大量", "全部", "部分", "大部分", "小部分",
            "许多", "几个", "一点"
        ]
        
        # 标点符号
        punctuation = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〱〲〳〴〵〶〷〸〹〺〻〼〽〾〿"
        
        stopwords.update(basic_stopwords)
        stopwords.update(time_words)
        stopwords.update(quantity_words)
        stopwords.update(punctuation)
        
        # 从文件加载停用词
        if stopwords_path and Path(stopwords_path).exists():
            try:
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    file_stopwords = [line.strip() for line in f if line.strip()]
                    stopwords.update(file_stopwords)
                self.logger.info(f"从文件加载停用词: {len(file_stopwords)} 个")
            except Exception as e:
                self.logger.warning(f"加载停用词文件失败: {e}")
        
        self.logger.info(f"停用词加载完成，共 {len(stopwords)} 个")
        return stopwords
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not text:
            return ""
        
        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        
        # 移除过多的重复字符
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # 移除特殊符号但保留中文标点
        text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\w\s\u3000-\u303f\uff00-\uffef]', '', text)
        
        return text.strip()
    
    def is_pure_punctuation(self, word: str) -> bool:
        """判断是否为纯标点符号"""
        chinese_punctuation = "！？。，；：""''（）【】《》、…—·"
        english_punctuation = string.punctuation
        all_punctuation = chinese_punctuation + english_punctuation
        
        return all(char in all_punctuation for char in word)
    
    def is_valid_token(self, word: str, pos: str = None) -> bool:
        """判断词汇是否有效"""
        # 长度检查
        if len(word) < 2:
            return False
        
        # 停用词检查
        if word in self.stopwords:
            return False
        
        # 纯数字检查
        if word.isdigit():
            return False
        
        # 纯标点检查
        if self.is_pure_punctuation(word):
            return False
        
        # 词性检查
        if pos and self.keep_pos and pos not in self.keep_pos:
            return False
        
        # 过滤单个字符（除非是重要的单字词）
        important_single_chars = {'国', '民', '党', '军', '法', '理', '工', '农', '商', '学'}
        if len(word) == 1 and word not in important_single_chars:
            return False
        
        return True
    
    def tokenize_document(self, text: str, keep_pos: bool = True) -> List[Union[Tuple[str, str], str]]:
        """
        对单个文档进行分词
        
        Args:
            text: 输入文本
            keep_pos: 是否保留词性标注
            
        Returns:
            分词结果列表
        """
        if not text:
            return []
        
        # 预处理文本
        text = self.preprocess_text(text)
        if not text:
            return []
        
        try:
            if keep_pos:
                # 带词性标注的分词
                words_with_pos = list(pseg.cut(text))
                
                filtered_tokens = []
                for word, pos in words_with_pos:
                    if self.is_valid_token(word, pos):
                        filtered_tokens.append((word, pos))
                        # 更新统计信息
                        self._update_stats(word, pos)
                
                return filtered_tokens
            else:
                # 只分词，不标注词性
                words = jieba.cut(text, cut_all=False)
                filtered_words = []
                
                for word in words:
                    if self.is_valid_token(word):
                        filtered_words.append(word)
                        # 更新统计信息
                        self._update_stats(word)
                
                return filtered_words
                
        except Exception as e:
            self.logger.error(f"分词处理失败: {e}")
            return []
    
    def _update_stats(self, word: str, pos: str = None):
        """更新分词统计信息"""
        self.tokenization_stats["total_tokens"] += 1
        self.tokenization_stats["unique_tokens"].add(word)
        self.tokenization_stats["token_frequencies"][word] += 1
        self.tokenization_stats["token_length_distribution"][len(word)] += 1
        
        if pos:
            self.tokenization_stats["pos_distribution"][pos] += 1
    
    def batch_tokenize(self, texts: List[str], keep_pos: bool = True, 
                      show_progress: bool = True) -> List[List[Union[Tuple[str, str], str]]]:
        """
        批量分词处理
        
        Args:
            texts: 文本列表
            keep_pos: 是否保留词性标注
            show_progress: 是否显示进度
            
        Returns:
            分词结果列表
        """
        results = []
        total = len(texts)
        
        start_time = time.time()
        
        for i, text in enumerate(texts):
            if show_progress and i % 100 == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                self.logger.info(f"分词进度: {i+1}/{total} ({speed:.1f} docs/sec)")
            
            tokens = self.tokenize_document(text, keep_pos)
            results.append(tokens)
        
        # 更新文档统计
        self.tokenization_stats["total_documents"] = total
        
        elapsed = time.time() - start_time
        self.logger.info(f"批量分词完成，处理 {total} 个文档，耗时 {elapsed:.2f} 秒")
        
        return results
    
    def get_tokenization_stats(self) -> Dict[str, Any]:
        """获取分词统计信息"""
        stats = self.tokenization_stats.copy()
        
        # 转换集合为数量
        stats["unique_tokens_count"] = len(stats["unique_tokens"])
        stats["oov_tokens_count"] = len(stats["oov_tokens"])
        
        # 计算其他统计指标
        if stats["total_tokens"] > 0:
            stats["vocabulary_richness"] = len(stats["unique_tokens"]) / stats["total_tokens"]
        else:
            stats["vocabulary_richness"] = 0
        
        if stats["total_documents"] > 0:
            stats["avg_tokens_per_doc"] = stats["total_tokens"] / stats["total_documents"]
        else:
            stats["avg_tokens_per_doc"] = 0
        
        # 不返回原始集合对象（避免序列化问题）
        stats.pop("unique_tokens", None)
        stats.pop("oov_tokens", None)
        
        return stats
    
    def clear_stats(self):
        """清空统计信息"""
        self.tokenization_stats = {
            "total_documents": 0,
            "total_tokens": 0,
            "unique_tokens": set(),
            "token_frequencies": Counter(),
            "pos_distribution": Counter(),
            "token_length_distribution": Counter(),
            "oov_tokens": set()
        }


class TokenizationAnalyzer:
    """分词效果统计分析器"""
    
    def __init__(self, tokenizer: OptimizedChineseTokenizer):
        self.tokenizer = tokenizer
        self.analysis_results = {}
        self.logger = logging.getLogger(__name__)
    
    def analyze_tokenization_results(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析分词结果"""
        self.logger.info("开始分词分析...")
        
        # 对所有文档进行分词
        tokenized_docs = []
        texts = [doc.get('content', '') for doc in documents]
        
        # 批量分词
        tokenized_results = self.tokenizer.batch_tokenize(texts, keep_pos=True)
        
        for i, (doc, tokens) in enumerate(zip(documents, tokenized_results)):
            tokenized_docs.append({
                'doc_id': doc.get('article_id', i),
                'title': doc.get('title', ''),
                'tokens': tokens,
                'token_count': len(tokens)
            })
        
        # 执行各项统计分析
        self.analysis_results = {
            "基础统计": self._basic_tokenization_stats(tokenized_docs),
            "词频分析": self._frequency_analysis(),
            "词性分析": self._pos_analysis(),
            "词长分析": self._length_analysis(),
            "分词质量评估": self._quality_assessment(documents, tokenized_docs)
        }
        
        self.logger.info("分词分析完成")
        return self.analysis_results
    
    def _basic_tokenization_stats(self, tokenized_docs: List[Dict]) -> Dict[str, Any]:
        """基础分词统计"""
        total_docs = len(tokenized_docs)
        total_tokens = sum(doc['token_count'] for doc in tokenized_docs)
        
        stats = self.tokenizer.get_tokenization_stats()
        unique_tokens = stats.get("unique_tokens_count", 0)
        
        token_counts = [doc['token_count'] for doc in tokenized_docs]
        
        return {
            "文档总数": total_docs,
            "词汇总数": total_tokens,
            "唯一词汇数": unique_tokens,
            "词汇丰富度": round(unique_tokens / total_tokens, 4) if total_tokens > 0 else 0,
            "平均每文档词数": round(total_tokens / total_docs, 2) if total_docs > 0 else 0,
            "词数分布": {
                "最大值": max(token_counts) if token_counts else 0,
                "最小值": min(token_counts) if token_counts else 0,
                "中位数": round(np.median(token_counts), 2) if token_counts else 0,
                "标准差": round(np.std(token_counts), 2) if token_counts else 0
            }
        }
    
    def _frequency_analysis(self) -> Dict[str, Any]:
        """词频分析"""
        token_freq = self.tokenizer.tokenization_stats["token_frequencies"]
        
        if not token_freq:
            return {"错误": "无词频数据"}
        
        # 高频词分析
        top_50_words = token_freq.most_common(50)
        
        # 低频词分析（只出现1次的词）
        hapax_legomena = [word for word, freq in token_freq.items() if freq == 1]
        
        # 词频分布区间
        freq_distribution = {
            "1次": len([w for w, f in token_freq.items() if f == 1]),
            "2-5次": len([w for w, f in token_freq.items() if 2 <= f <= 5]),
            "6-10次": len([w for w, f in token_freq.items() if 6 <= f <= 10]),
            "11-50次": len([w for w, f in token_freq.items() if 11 <= f <= 50]),
            "50次以上": len([w for w, f in token_freq.items() if f > 50])
        }
        
        return {
            "高频词TOP50": top_50_words,
            "hapax_legomena数量": len(hapax_legomena),
            "hapax_legomena比例": f"{len(hapax_legomena)/len(token_freq)*100:.2f}%",
            "词频分布区间": freq_distribution,
            "词频统计": {
                "最高频词": token_freq.most_common(1)[0] if token_freq else None,
                "平均频次": round(np.mean(list(token_freq.values())), 2),
                "频次标准差": round(np.std(list(token_freq.values())), 2)
            }
        }
    
    def _pos_analysis(self) -> Dict[str, Any]:
        """词性分析"""
        pos_dist = self.tokenizer.tokenization_stats["pos_distribution"]
        
        if not pos_dist:
            return {"错误": "无词性数据"}
        
        total_pos = sum(pos_dist.values())
        
        # 词性映射（中文说明）
        pos_mapping = {
            'n': '名词', 'nr': '人名', 'ns': '地名', 'nt': '机构名', 'nw': '作品名', 'nz': '其他专名',
            'v': '动词', 'vd': '副动词', 'vn': '名动词',
            'a': '形容词', 'ad': '副形词', 'an': '名形词',
            'i': '成语', 'j': '简称略语', 'l': '习用语'
        }
        
        pos_analysis = {}
        for pos, count in pos_dist.most_common():
            pos_name = pos_mapping.get(pos, pos)
            pos_analysis[pos_name] = {
                "数量": count,
                "占比": f"{count/total_pos*100:.2f}%"
            }
        
        return {
            "词性分布": pos_analysis,
            "主要词性": pos_dist.most_common(5),
            "词性丰富度": len(pos_dist),
            "词性集中度": round(max(pos_dist.values()) / total_pos, 4) if total_pos > 0 else 0
        }
    
    def _length_analysis(self) -> Dict[str, Any]:
        """词长分析"""
        length_dist = self.tokenizer.tokenization_stats["token_length_distribution"]
        
        if not length_dist:
            return {"错误": "无词长数据"}
        
        # 计算平均词长
        total_chars = sum(length * count for length, count in length_dist.items())
        total_words = sum(length_dist.values())
        avg_length = total_chars / total_words if total_words > 0 else 0
        
        return {
            "词长分布": dict(sorted(length_dist.items())),
            "平均词长": round(avg_length, 2),
            "最常见词长": max(length_dist.items(), key=lambda x: x[1])[0] if length_dist else 0,
            "词长统计": {
                "1字词占比": f"{length_dist.get(1, 0)/total_words*100:.2f}%",
                "2字词占比": f"{length_dist.get(2, 0)/total_words*100:.2f}%",
                "3字词占比": f"{length_dist.get(3, 0)/total_words*100:.2f}%",
                "4字及以上占比": f"{sum(count for length, count in length_dist.items() if length >= 4)/total_words*100:.2f}%"
            }
        }
    
    def _quality_assessment(self, original_docs: List[Dict], tokenized_docs: List[Dict]) -> Dict[str, Any]:
        """分词质量评估"""
        # 分词一致性检查（抽样）
        consistency_scores = []
        sample_size = min(10, len(original_docs))
        
        for i in range(sample_size):
            orig = original_docs[i]
            tokenized = tokenized_docs[i]
            
            # 重新分词同一文档
            tokens1 = [t[0] for t in self.tokenizer.tokenize_document(orig.get('content', ''))]
            tokens2 = [t[0] if isinstance(t, tuple) else t for t in tokenized['tokens']]
            
            # 计算一致性
            if tokens1 and tokens2:
                common_tokens = len(set(tokens1) & set(tokens2))
                total_tokens = len(set(tokens1) | set(tokens2))
                consistency = common_tokens / total_tokens if total_tokens > 0 else 0
                consistency_scores.append(consistency)
        
        # OOV（Out of Vocabulary）分析
        stats = self.tokenizer.get_tokenization_stats()
        total_tokens = stats.get("total_tokens", 0)
        oov_count = stats.get("oov_tokens_count", 0)
        
        return {
            "分词一致性": {
                "平均一致性": round(np.mean(consistency_scores), 4) if consistency_scores else 0,
                "一致性标准差": round(np.std(consistency_scores), 4) if consistency_scores else 0,
                "样本数量": len(consistency_scores)
            },
            "OOV分析": {
                "OOV词数": oov_count,
                "OOV比例": f"{oov_count/total_tokens*100:.2f}%" if total_tokens > 0 else "0%"
            },
            "分词效率": {
                "总处理时间": "实时计算",
                "平均处理速度": f"{stats.get('avg_tokens_per_doc', 0):.1f} 词/文档"
            },
            "质量评分": self._calculate_tokenization_quality_score(consistency_scores, oov_count, total_tokens)
        }
    
    def _calculate_tokenization_quality_score(self, consistency_scores: List[float], 
                                            oov_count: int, total_tokens: int) -> float:
        """计算分词质量评分"""
        score = 100.0
        
        # 一致性评分
        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            score *= avg_consistency
        
        # OOV惩罚
        if total_tokens > 0:
            oov_rate = oov_count / total_tokens
            score *= (1 - oov_rate)
        
        return round(min(100.0, max(0.0, score)), 2)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建分词器
    tokenizer = OptimizedChineseTokenizer()
    
    # 测试文本
    test_texts = [
        "人工智能技术在医疗领域的应用越来越广泛，特别是在疾病诊断和治疗方案制定方面。",
        "新冠疫情防控工作取得重大进展，疫苗接种率持续提升，健康码系统运行良好。",
        "碳达峰碳中和目标的实现需要全社会的共同努力，新能源产业发展前景广阔。"
    ]
    
    # 测试分词
    for i, text in enumerate(test_texts):
        print(f"\n=== 测试文本 {i+1} ===")
        print(f"原文: {text}")
        
        # 带词性分词
        tokens_with_pos = tokenizer.tokenize_document(text, keep_pos=True)
        print(f"分词结果(带词性): {tokens_with_pos}")
        
        # 不带词性分词
        tokens_only = tokenizer.tokenize_document(text, keep_pos=False)
        print(f"分词结果(仅词汇): {tokens_only}")
    
    # 获取统计信息
    stats = tokenizer.get_tokenization_stats()
    print(f"\n=== 分词统计 ===")
    for key, value in stats.items():
        if not isinstance(value, (Counter, dict)):
            print(f"{key}: {value}")
    
    print("\n分词器测试完成！")