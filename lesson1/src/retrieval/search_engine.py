#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import json

from ..preprocessing.tokenizer import OptimizedChineseTokenizer
from ..preprocessing.analyzer import RawDataAnalyzer
from ..preprocessing.tokenizer import TokenizationAnalyzer
from .tfidf import TFIDFRetrieval
from .similarity import CosineRetrieval, SimilarityAnalyzer


class ChineseNewsSearchSystem:
    """中文新闻检索系统"""
    
    def __init__(self, custom_dict_path: Optional[str] = None, 
                 stopwords_path: Optional[str] = None):
        """
        初始化搜索系统
        
        Args:
            custom_dict_path: 自定义词典路径
            stopwords_path: 停用词文件路径
        """
        self.tokenizer = OptimizedChineseTokenizer(custom_dict_path, stopwords_path)
        self.tfidf_retrieval = TFIDFRetrieval(self.tokenizer)
        self.cosine_retrieval = None
        self.similarity_analyzer = None
        
        self.documents = []
        self.is_indexed = False
        
        # 统计信息
        self.system_stats = {
            "index_build_time": 0,
            "avg_search_time": 0,
            "total_searches": 0,
            "last_update_time": None
        }
        
        # 分析结果缓存
        self.data_analysis = None
        self.tokenization_analysis = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("中文新闻检索系统初始化完成")
    
    def index_documents(self, documents: List[Dict[str, Any]], save_analysis: bool = True) -> Dict[str, Any]:
        """
        建立文档索引
        
        Args:
            documents: 文档列表
            save_analysis: 是否保存分析结果
            
        Returns:
            索引构建结果
        """
        self.logger.info(f"开始建立文档索引，文档数量: {len(documents)}")
        start_time = time.time()
        
        try:
            self.documents = documents
            
            # 1. 原始数据分析
            self.logger.info("执行原始数据分析...")
            data_analyzer = RawDataAnalyzer(documents)
            self.data_analysis = data_analyzer.comprehensive_analysis()
            
            # 2. 分词分析
            self.logger.info("执行分词分析...")
            tokenization_analyzer = TokenizationAnalyzer(self.tokenizer)
            self.tokenization_analysis = tokenization_analyzer.analyze_tokenization_results(documents)
            
            # 3. 构建TF-IDF检索模型
            self.logger.info("构建TF-IDF检索模型...")
            self.tfidf_retrieval.fit(documents)
            
            # 4. 初始化余弦相似度检索器
            self.cosine_retrieval = CosineRetrieval(self.tfidf_retrieval, documents)
            self.similarity_analyzer = SimilarityAnalyzer(self.cosine_retrieval)
            
            # 5. 更新系统状态
            build_time = time.time() - start_time
            self.system_stats["index_build_time"] = build_time
            self.system_stats["last_update_time"] = time.strftime('%Y-%m-%d %H:%M:%S')
            self.is_indexed = True
            
            # 6. 保存分析结果
            if save_analysis:
                self._save_analysis_results()
            
            result = {
                "success": True,
                "message": "索引构建完成",
                "statistics": {
                    "文档数量": len(documents),
                    "构建时间": f"{build_time:.2f} 秒",
                    "词汇表大小": self.tfidf_retrieval.vectorizer.get_vocabulary_size(),
                    "TF-IDF矩阵形状": self.tfidf_retrieval.doc_vectors.shape,
                    "内存使用": f"{self.tfidf_retrieval.doc_vectors.data.nbytes / (1024 * 1024):.2f} MB"
                }
            }
            
            self.logger.info(f"索引构建完成，耗时: {build_time:.2f}秒")
            return result
            
        except Exception as e:
            self.logger.error(f"索引构建失败: {e}")
            return {
                "success": False,
                "message": f"索引构建失败: {str(e)}",
                "statistics": {}
            }
    
    def search(self, query: str, top_k: int = 10, similarity_threshold: float = 0.01,
               explain: bool = False) -> Tuple[List[Dict[str, Any]], float]:
        """
        执行检索
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            explain: 是否返回详细解释
            
        Returns:
            检索结果和检索时间
        """
        if not self.is_indexed:
            raise ValueError("系统未建立索引，请先调用 index_documents()")
        
        start_time = time.time()
        
        try:
            # 执行检索
            results = self.cosine_retrieval.search(
                query, 
                top_k=top_k, 
                similarity_threshold=similarity_threshold
            )
            
            # 如果需要详细解释
            if explain and results:
                for result in results:
                    doc_id = result['document_id']
                    explanation = self.cosine_retrieval.explain_similarity(query, doc_id)
                    result['explanation'] = explanation
            
            # 更新统计信息
            search_time = time.time() - start_time
            self._update_search_stats(search_time)
            
            return results, search_time
            
        except Exception as e:
            self.logger.error(f"检索失败: {e}")
            return [], 0.0
    
    def explain_search(self, query: str, document_id: int) -> Dict[str, Any]:
        """
        解释检索结果
        
        Args:
            query: 查询字符串
            document_id: 文档ID
            
        Returns:
            详细解释信息
        """
        if not self.is_indexed:
            raise ValueError("系统未建立索引，请先调用 index_documents()")
        
        return self.cosine_retrieval.explain_similarity(query, document_id)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        if not self.is_indexed:
            return {"错误": "系统未建立索引"}
        
        # 基础统计
        basic_stats = {
            "索引状态": "已建立" if self.is_indexed else "未建立",
            "文档数量": len(self.documents),
            "词汇表大小": self.tfidf_retrieval.vectorizer.get_vocabulary_size(),
            "索引构建时间": f"{self.system_stats['index_build_time']:.2f} 秒",
            "平均检索时间": f"{self.system_stats['avg_search_time']:.4f} 秒",
            "总检索次数": self.system_stats["total_searches"],
            "最后更新时间": self.system_stats["last_update_time"]
        }
        
        # TF-IDF模型统计
        tfidf_stats = self.tfidf_retrieval.get_model_statistics()
        
        # 分词统计
        tokenization_stats = self.tokenizer.get_tokenization_stats()
        
        return {
            "基础统计": basic_stats,
            "TF-IDF模型": tfidf_stats,
            "分词统计": {
                "文档总数": tokenization_stats.get("total_documents", 0),
                "词汇总数": tokenization_stats.get("total_tokens", 0),
                "唯一词汇数": tokenization_stats.get("unique_tokens_count", 0),
                "词汇丰富度": tokenization_stats.get("vocabulary_richness", 0),
                "平均每文档词数": tokenization_stats.get("avg_tokens_per_doc", 0)
            }
        }
    
    def get_data_analysis(self) -> Optional[Dict[str, Any]]:
        """获取数据分析结果"""
        return self.data_analysis
    
    def get_tokenization_analysis(self) -> Optional[Dict[str, Any]]:
        """获取分词分析结果"""
        return self.tokenization_analysis
    
    def analyze_query_performance(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        分析查询性能
        
        Args:
            test_queries: 测试查询列表
            
        Returns:
            性能分析结果
        """
        if not self.is_indexed:
            raise ValueError("系统未建立索引，请先调用 index_documents()")
        
        return self.similarity_analyzer.analyze_query_performance(test_queries)
    
    def benchmark_search_methods(self, query: str) -> Dict[str, Any]:
        """
        基准测试不同搜索方法
        
        Args:
            query: 测试查询
            
        Returns:
            基准测试结果
        """
        if not self.is_indexed:
            raise ValueError("系统未建立索引，请先调用 index_documents()")
        
        return self.similarity_analyzer.compare_retrieval_methods(query)
    
    def get_similarity_distribution(self, query: str) -> Dict[str, Any]:
        """
        获取查询的相似度分布
        
        Args:
            query: 查询字符串
            
        Returns:
            相似度分布统计
        """
        if not self.is_indexed:
            raise ValueError("系统未建立索引，请先调用 index_documents()")
        
        return self.cosine_retrieval.get_similarity_distribution(query)
    
    def save_index(self, filepath: str) -> bool:
        """
        保存索引到文件
        
        Args:
            filepath: 保存路径
            
        Returns:
            是否保存成功
        """
        if not self.is_indexed:
            self.logger.error("没有可保存的索引")
            return False
        
        try:
            index_data = {
                'tfidf_retrieval': self.tfidf_retrieval,
                'documents': self.documents,
                'system_stats': self.system_stats,
                'data_analysis': self.data_analysis,
                'tokenization_analysis': self.tokenization_analysis,
                'is_indexed': self.is_indexed
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(index_data, f)
            
            self.logger.info(f"索引已保存到: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存索引失败: {e}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """
        从文件加载索引
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否加载成功
        """
        try:
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            
            self.tfidf_retrieval = index_data['tfidf_retrieval']
            self.documents = index_data['documents']
            self.system_stats = index_data['system_stats']
            self.data_analysis = index_data.get('data_analysis')
            self.tokenization_analysis = index_data.get('tokenization_analysis')
            self.is_indexed = index_data['is_indexed']
            
            # 重新初始化检索器
            self.cosine_retrieval = CosineRetrieval(self.tfidf_retrieval, self.documents)
            self.similarity_analyzer = SimilarityAnalyzer(self.cosine_retrieval)
            
            self.logger.info(f"索引已从 {filepath} 加载")
            return True
            
        except Exception as e:
            self.logger.error(f"加载索引失败: {e}")
            return False
    
    def _update_search_stats(self, search_time: float):
        """更新搜索统计信息"""
        self.system_stats["total_searches"] += 1
        
        # 更新平均搜索时间
        total_time = (self.system_stats["avg_search_time"] * 
                     (self.system_stats["total_searches"] - 1) + search_time)
        self.system_stats["avg_search_time"] = total_time / self.system_stats["total_searches"]
    
    def _save_analysis_results(self):
        """保存分析结果到文件"""
        try:
            # 创建分析结果目录
            analysis_dir = Path("data/analysis")
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存数据分析结果
            if self.data_analysis:
                with open(analysis_dir / "data_analysis.json", 'w', encoding='utf-8') as f:
                    json.dump(self.data_analysis, f, ensure_ascii=False, indent=2, default=str)
            
            # 保存分词分析结果
            if self.tokenization_analysis:
                with open(analysis_dir / "tokenization_analysis.json", 'w', encoding='utf-8') as f:
                    json.dump(self.tokenization_analysis, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info("分析结果已保存")
            
        except Exception as e:
            self.logger.warning(f"保存分析结果失败: {e}")


class SearchInterface:
    """搜索系统交互接口"""
    
    def __init__(self, search_system: ChineseNewsSearchSystem):
        self.search_system = search_system
        self.logger = logging.getLogger(__name__)
    
    def interactive_search(self):
        """交互式检索界面"""
        print("\n" + "="*50)
        print("🔍 中文新闻检索系统")
        print("="*50)
        
        if not self.search_system.is_indexed:
            print("❌ 系统尚未建立索引，请先加载数据并建立索引")
            return
        
        stats = self.search_system.get_system_stats()
        print(f"📊 系统状态:")
        print(f"   - 文档数量: {stats['基础统计']['文档数量']}")
        print(f"   - 词汇表大小: {stats['基础统计']['词汇表大小']}")
        print(f"   - 平均检索时间: {stats['基础统计']['平均检索时间']}")
        
        print("\n💡 使用说明:")
        print("   - 直接输入关键词进行检索")
        print("   - 输入 'stats' 查看系统统计")
        print("   - 输入 'help' 查看帮助")
        print("   - 输入 'quit' 退出系统")
        print("-"*50)
        
        while True:
            try:
                query = input("\n🔎 请输入查询: ").strip()
                
                if query.lower() == 'quit':
                    print("👋 感谢使用，再见！")
                    break
                
                elif query.lower() == 'stats':
                    self._show_system_stats()
                
                elif query.lower() == 'help':
                    self._show_help()
                
                elif not query:
                    continue
                
                else:
                    self._execute_search(query)
                    
            except KeyboardInterrupt:
                print("\n\n👋 程序已中断，再见！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
    
    def _execute_search(self, query: str):
        """执行搜索并显示结果"""
        print(f"\n🔍 正在搜索: '{query}'...")
        
        try:
            results, search_time = self.search_system.search(query, top_k=5)
            
            print(f"⏱️  检索时间: {search_time:.4f} 秒")
            print(f"📄 找到 {len(results)} 条相关结果:\n")
            
            if not results:
                print("😔 未找到相关结果，请尝试其他关键词")
                return
            
            for i, result in enumerate(results, 1):
                doc = result['document']
                score = result['similarity_score']
                
                print(f"📰 {i}. 相似度: {score:.4f}")
                print(f"   📌 标题: {doc['title']}")
                print(f"   📂 分类: {doc.get('category', '未分类')}")
                print(f"   📝 摘要: {doc['content'][:150]}...")
                print(f"   📊 字数: {doc.get('word_count', 0)} 字")
                
                if i < len(results):
                    print("   " + "-"*40)
                    
        except Exception as e:
            print(f"❌ 检索出错: {e}")
    
    def _show_system_stats(self):
        """显示系统统计信息"""
        print("\n📊 系统统计信息:")
        print("-"*30)
        
        stats = self.search_system.get_system_stats()
        
        # 基础统计
        basic = stats.get("基础统计", {})
        print("🔧 基础信息:")
        for key, value in basic.items():
            print(f"   - {key}: {value}")
        
        # TF-IDF统计
        tfidf = stats.get("TF-IDF模型", {})
        if tfidf:
            print("\n🧮 TF-IDF模型:")
            for key, value in list(tfidf.items())[:5]:  # 只显示前几项
                print(f"   - {key}: {value}")
        
        # 分词统计
        token = stats.get("分词统计", {})
        if token:
            print("\n✂️  分词统计:")
            for key, value in token.items():
                print(f"   - {key}: {value}")
    
    def _show_help(self):
        """显示帮助信息"""
        print("\n❓ 帮助信息:")
        print("-"*20)
        print("🔍 检索技巧:")
        print("   1. 使用具体的关键词，如：'人工智能'、'疫情防控'")
        print("   2. 支持多词查询，如：'人工智能 医疗应用'")
        print("   3. 使用专业术语可以获得更准确的结果")
        print("   4. 避免使用过于常见的词汇")
        
        print("\n⚙️  系统命令:")
        print("   - stats: 查看系统统计信息")
        print("   - help:  显示此帮助信息")
        print("   - quit:  退出系统")
        
        print("\n💡 示例查询:")
        print("   - 新冠疫情")
        print("   - 人工智能技术")
        print("   - 碳达峰碳中和")
        print("   - 经济发展政策")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    test_documents = [
        {
            "article_id": "test1",
            "title": "人工智能技术在医疗领域的应用",
            "content": "人工智能技术在医疗领域的应用越来越广泛，包括疾病诊断、药物研发、医疗影像分析等方面。机器学习算法可以帮助医生更准确地识别疾病症状，提高诊断效率。深度学习技术在医疗影像分析中表现出色，能够识别X光片、CT扫描等医疗图像中的异常情况。",
            "category": "科技",
            "word_count": 150
        },
        {
            "article_id": "test2", 
            "title": "新冠疫情防控取得重要进展",
            "content": "新冠疫情防控工作取得重要进展，疫苗接种率持续提升。全国多地建立了完善的疫情防控体系，包括核酸检测、疫苗接种、隔离管控等措施。健康码系统的广泛应用，为疫情防控提供了有力支撑。各级政府积极响应，确保人民群众的生命健康安全。",
            "category": "社会",
            "word_count": 120
        },
        {
            "article_id": "test3",
            "title": "碳达峰碳中和目标推动绿色发展",
            "content": "碳达峰碳中和目标的提出，为我国绿色发展指明了方向。新能源产业快速发展，太阳能、风能等可再生能源装机容量大幅增长。节能减排技术不断创新，绿色制造、清洁生产成为企业发展的重要方向。生态环境保护工作取得显著成效。",
            "category": "环保",
            "word_count": 110
        }
    ]
    
    print("=== 搜索引擎测试 ===")
    
    # 创建搜索系统
    search_system = ChineseNewsSearchSystem()
    
    # 建立索引
    result = search_system.index_documents(test_documents)
    print(f"索引构建结果: {result}")
    
    # 执行搜索
    test_queries = ["人工智能", "疫情防控", "绿色发展"]
    
    for query in test_queries:
        print(f"\n--- 搜索: '{query}' ---")
        results, search_time = search_system.search(query, top_k=2)
        
        print(f"检索时间: {search_time:.4f} 秒")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']} (相似度: {result['similarity_score']:.4f})")
    
    # 显示系统统计
    print(f"\n--- 系统统计 ---")
    stats = search_system.get_system_stats()
    print(f"基础统计: {stats['基础统计']}")
    
    print("\n搜索引擎测试完成！")