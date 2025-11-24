#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from pathlib import Path


class RetrievalEvaluator:
    """检索系统评估器"""
    
    def __init__(self, search_system):
        """
        初始化评估器
        
        Args:
            search_system: 搜索系统实例
        """
        self.search_system = search_system
        self.logger = logging.getLogger(__name__)
    
    def create_test_queries(self) -> List[Dict[str, Any]]:
        """创建测试查询集"""
        test_queries = [
            {
                "query": "人工智能",
                "type": "单词查询",
                "description": "测试单个热门词汇的检索效果"
            },
            {
                "query": "新冠疫情防控",
                "type": "短语查询",
                "description": "测试多词短语的检索能力"
            },
            {
                "query": "经济发展 政策",
                "type": "多词查询",
                "description": "测试多个相关词汇的组合查询"
            },
            {
                "query": "北京冬奥会开幕式",
                "type": "实体查询",
                "description": "测试特定事件实体的检索"
            },
            {
                "query": "碳达峰碳中和目标",
                "type": "复合概念查询",
                "description": "测试复合专业概念的检索"
            },
            {
                "query": "机器学习算法",
                "type": "技术词汇查询",
                "description": "测试技术领域词汇的检索"
            },
            {
                "query": "疫苗接种",
                "type": "热点话题查询",
                "description": "测试社会热点话题的检索"
            },
            {
                "query": "绿色发展",
                "type": "政策概念查询",
                "description": "测试政策相关概念的检索"
            },
            {
                "query": "数字经济",
                "type": "经济概念查询",
                "description": "测试经济领域概念的检索"
            },
            {
                "query": "智能制造",
                "type": "产业概念查询",
                "description": "测试产业发展概念的检索"
            }
        ]
        return test_queries
    
    def evaluate_retrieval_quality(self, test_queries: Optional[List[Dict[str, Any]]] = None,
                                 human_judgments: Optional[Dict[str, List[int]]] = None) -> Dict[str, Any]:
        """
        评估检索质量
        
        Args:
            test_queries: 测试查询列表，如果为None则使用默认查询
            human_judgments: 人工相关性判断 {query: [relevant_doc_ids]}
            
        Returns:
            评估结果
        """
        if test_queries is None:
            test_queries = self.create_test_queries()
        
        self.logger.info(f"开始评估检索质量，测试查询数量: {len(test_queries)}")
        
        results = {
            "测试概述": {
                "测试查询数量": len(test_queries),
                "评估时间": time.strftime('%Y-%m-%d %H:%M:%S'),
                "是否有人工标注": human_judgments is not None
            },
            "详细结果": {},
            "总体统计": {}
        }
        
        # 收集所有查询的统计数据
        all_response_times = []
        all_result_counts = []
        all_avg_similarities = []
        all_max_similarities = []
        query_types = {}
        
        for query_data in test_queries:
            query = query_data["query"]
            query_type = query_data.get("type", "未知类型")
            
            self.logger.info(f"评估查询: '{query}' ({query_type})")
            
            # 执行检索
            start_time = time.time()
            search_results, search_time = self.search_system.search(query, top_k=10)
            response_time = time.time() - start_time
            
            # 基础指标
            similarities = [r['similarity_score'] for r in search_results]
            
            query_result = {
                "查询": query,
                "类型": query_type,
                "描述": query_data.get("description", ""),
                "检索时间": round(search_time, 4),
                "总响应时间": round(response_time, 4),
                "结果数量": len(search_results),
                "平均相似度": round(np.mean(similarities), 4) if similarities else 0,
                "最高相似度": round(max(similarities), 4) if similarities else 0,
                "相似度标准差": round(np.std(similarities), 4) if similarities else 0,
                "检索结果": [
                    {
                        "文档ID": r['document_id'],
                        "标题": r['title'],
                        "相似度": round(r['similarity_score'], 4),
                        "分类": r.get('category', '未分类')
                    }
                    for r in search_results[:5]  # 只保存前5个结果
                ]
            }
            
            # 如果有人工判断数据，计算precision、recall等指标
            if human_judgments and query in human_judgments:
                relevant_docs = human_judgments[query]
                retrieved_docs = [r['document_id'] for r in search_results]
                
                precision = self.calculate_precision(retrieved_docs, relevant_docs)
                recall = self.calculate_recall(retrieved_docs, relevant_docs)
                f1 = self.calculate_f1(precision, recall)
                
                query_result.update({
                    "Precision@10": round(precision, 4),
                    "Recall@10": round(recall, 4),
                    "F1@10": round(f1, 4),
                    "相关文档数": len(relevant_docs),
                    "检索到相关文档数": len(set(retrieved_docs) & set(relevant_docs))
                })
            
            results["详细结果"][query] = query_result
            
            # 收集统计数据
            all_response_times.append(response_time)
            all_result_counts.append(len(search_results))
            if similarities:
                all_avg_similarities.append(np.mean(similarities))
                all_max_similarities.append(max(similarities))
            
            # 按类型统计
            if query_type not in query_types:
                query_types[query_type] = []
            query_types[query_type].append(query_result)
        
        # 计算总体统计
        results["总体统计"] = {
            "性能指标": {
                "平均响应时间": f"{np.mean(all_response_times):.4f} 秒",
                "响应时间标准差": f"{np.std(all_response_times):.4f} 秒",
                "最快响应时间": f"{min(all_response_times):.4f} 秒",
                "最慢响应时间": f"{max(all_response_times):.4f} 秒"
            },
            "检索效果": {
                "平均结果数量": round(np.mean(all_result_counts), 2),
                "平均相似度": round(np.mean(all_avg_similarities), 4) if all_avg_similarities else 0,
                "最高平均相似度": round(max(all_avg_similarities), 4) if all_avg_similarities else 0,
                "无结果查询数": len([c for c in all_result_counts if c == 0]),
                "无结果查询比例": f"{len([c for c in all_result_counts if c == 0])/len(test_queries)*100:.1f}%"
            }
        }
        
        # 按查询类型统计
        type_stats = {}
        for query_type, type_queries in query_types.items():
            type_response_times = [q["总响应时间"] for q in type_queries]
            type_similarities = [q["平均相似度"] for q in type_queries if q["平均相似度"] > 0]
            
            type_stats[query_type] = {
                "查询数量": len(type_queries),
                "平均响应时间": f"{np.mean(type_response_times):.4f} 秒",
                "平均相似度": round(np.mean(type_similarities), 4) if type_similarities else 0,
                "成功率": f"{len([q for q in type_queries if q['结果数量'] > 0])/len(type_queries)*100:.1f}%"
            }
        
        results["按类型统计"] = type_stats
        
        self.logger.info("检索质量评估完成")
        return results
    
    def calculate_precision(self, retrieved: List[int], relevant: List[int]) -> float:
        """
        计算精确率
        
        Args:
            retrieved: 检索到的文档ID列表
            relevant: 相关文档ID列表
            
        Returns:
            精确率
        """
        if not retrieved:
            return 0.0
        
        relevant_retrieved = len(set(retrieved) & set(relevant))
        return relevant_retrieved / len(retrieved)
    
    def calculate_recall(self, retrieved: List[int], relevant: List[int]) -> float:
        """
        计算召回率
        
        Args:
            retrieved: 检索到的文档ID列表
            relevant: 相关文档ID列表
            
        Returns:
            召回率
        """
        if not relevant:
            return 0.0
        
        relevant_retrieved = len(set(retrieved) & set(relevant))
        return relevant_retrieved / len(relevant)
    
    def calculate_f1(self, precision: float, recall: float) -> float:
        """
        计算F1分数
        
        Args:
            precision: 精确率
            recall: 召回率
            
        Returns:
            F1分数
        """
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def evaluate_system_performance(self, iterations: int = 100) -> Dict[str, Any]:
        """
        评估系统性能
        
        Args:
            iterations: 测试迭代次数
            
        Returns:
            性能评估结果
        """
        self.logger.info(f"开始系统性能评估，迭代次数: {iterations}")
        
        test_queries = [q["query"] for q in self.create_test_queries()]
        
        # 性能测试
        response_times = []
        memory_usage = []
        
        import psutil
        import random
        
        for i in range(iterations):
            # 随机选择查询
            query = random.choice(test_queries)
            
            # 记录内存使用
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 执行检索
            start_time = time.time()
            results, _ = self.search_system.search(query, top_k=10)
            response_time = time.time() - start_time
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            response_times.append(response_time)
            memory_usage.append(memory_after - memory_before)
            
            if (i + 1) % 20 == 0:
                self.logger.info(f"性能测试进度: {i+1}/{iterations}")
        
        # 计算统计信息
        performance_stats = {
            "测试配置": {
                "迭代次数": iterations,
                "测试查询数": len(test_queries),
                "测试时间": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            "响应时间统计": {
                "平均值": f"{np.mean(response_times):.4f} 秒",
                "中位数": f"{np.median(response_times):.4f} 秒",
                "最小值": f"{min(response_times):.4f} 秒",
                "最大值": f"{max(response_times):.4f} 秒",
                "标准差": f"{np.std(response_times):.4f} 秒",
                "95百分位": f"{np.percentile(response_times, 95):.4f} 秒"
            },
            "吞吐量统计": {
                "平均QPS": f"{1/np.mean(response_times):.2f} 查询/秒",
                "峰值QPS": f"{1/min(response_times):.2f} 查询/秒"
            },
            "内存使用": {
                "平均内存变化": f"{np.mean(memory_usage):.2f} MB",
                "最大内存变化": f"{max(memory_usage):.2f} MB",
                "内存标准差": f"{np.std(memory_usage):.2f} MB"
            }
        }
        
        self.logger.info("系统性能评估完成")
        return performance_stats
    
    def benchmark_different_configurations(self) -> Dict[str, Any]:
        """基准测试不同配置的性能"""
        self.logger.info("开始不同配置的基准测试")
        
        test_query = "人工智能技术应用"
        configurations = [
            {"top_k": 5, "threshold": 0.01, "name": "默认配置"},
            {"top_k": 10, "threshold": 0.01, "name": "增加结果数"},
            {"top_k": 5, "threshold": 0.05, "name": "提高阈值"},
            {"top_k": 20, "threshold": 0.001, "name": "最大召回"}
        ]
        
        benchmark_results = {
            "测试查询": test_query,
            "配置对比": {}
        }
        
        for config in configurations:
            times = []
            result_counts = []
            similarities = []
            
            # 每个配置测试10次
            for _ in range(10):
                start_time = time.time()
                results, _ = self.search_system.search(
                    test_query, 
                    top_k=config["top_k"],
                    similarity_threshold=config["threshold"]
                )
                response_time = time.time() - start_time
                
                times.append(response_time)
                result_counts.append(len(results))
                if results:
                    similarities.append(np.mean([r['similarity_score'] for r in results]))
            
            benchmark_results["配置对比"][config["name"]] = {
                "参数": f"top_k={config['top_k']}, threshold={config['threshold']}",
                "平均响应时间": f"{np.mean(times):.4f} 秒",
                "平均结果数": round(np.mean(result_counts), 1),
                "平均相似度": round(np.mean(similarities), 4) if similarities else 0,
                "标准差": f"{np.std(times):.4f} 秒"
            }
        
        return benchmark_results
    
    def generate_evaluation_report(self, output_dir: str = "data/analysis") -> str:
        """
        生成完整的评估报告
        
        Args:
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        self.logger.info("开始生成评估报告")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 执行各项评估
        quality_results = self.evaluate_retrieval_quality()
        performance_results = self.evaluate_system_performance(iterations=50)
        benchmark_results = self.benchmark_different_configurations()
        
        # 获取系统统计
        system_stats = self.search_system.get_system_stats()
        
        # 生成完整报告
        full_report = {
            "报告生成时间": time.strftime('%Y-%m-%d %H:%M:%S'),
            "系统信息": system_stats,
            "检索质量评估": quality_results,
            "系统性能评估": performance_results,
            "配置基准测试": benchmark_results,
            "评估总结": self._generate_summary(quality_results, performance_results)
        }
        
        # 保存报告
        report_file = output_path / f"evaluation_report_{int(time.time())}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2, default=str)
        
        # 生成简化的markdown报告
        markdown_file = output_path / f"evaluation_summary_{int(time.time())}.md"
        self._generate_markdown_report(full_report, markdown_file)
        
        self.logger.info(f"评估报告已生成: {report_file}")
        return str(report_file)
    
    def _generate_summary(self, quality_results: Dict, performance_results: Dict) -> Dict[str, Any]:
        """生成评估总结"""
        # 计算整体评分
        quality_score = 0
        performance_score = 0
        
        # 质量评分（基于平均相似度和成功率）
        avg_similarity = quality_results["总体统计"]["检索效果"]["平均相似度"]
        success_rate = 100 - float(quality_results["总体统计"]["检索效果"]["无结果查询比例"].rstrip('%'))
        quality_score = (avg_similarity * 100 + success_rate) / 2
        
        # 性能评分（基于响应时间）
        avg_response_time = float(performance_results["响应时间统计"]["平均值"].split()[0])
        if avg_response_time < 0.01:
            performance_score = 100
        elif avg_response_time < 0.05:
            performance_score = 90
        elif avg_response_time < 0.1:
            performance_score = 80
        else:
            performance_score = max(50, 100 - avg_response_time * 100)
        
        overall_score = (quality_score + performance_score) / 2
        
        return {
            "整体评分": round(overall_score, 1),
            "质量评分": round(quality_score, 1),
            "性能评分": round(performance_score, 1),
            "评分说明": {
                "90-100": "优秀",
                "80-89": "良好", 
                "70-79": "一般",
                "60-69": "及格",
                "0-59": "需要改进"
            },
            "主要优势": self._identify_strengths(quality_results, performance_results),
            "改进建议": self._identify_improvements(quality_results, performance_results)
        }
    
    def _identify_strengths(self, quality_results: Dict, performance_results: Dict) -> List[str]:
        """识别系统优势"""
        strengths = []
        
        avg_response_time = float(performance_results["响应时间统计"]["平均值"].split()[0])
        if avg_response_time < 0.05:
            strengths.append("响应速度快，平均检索时间低于50毫秒")
        
        success_rate = 100 - float(quality_results["总体统计"]["检索效果"]["无结果查询比例"].rstrip('%'))
        if success_rate > 90:
            strengths.append("检索成功率高，大部分查询都能返回相关结果")
        
        avg_similarity = quality_results["总体统计"]["检索效果"]["平均相似度"]
        if avg_similarity > 0.1:
            strengths.append("检索结果相关性较高，平均相似度表现良好")
        
        return strengths if strengths else ["系统运行稳定"]
    
    def _identify_improvements(self, quality_results: Dict, performance_results: Dict) -> List[str]:
        """识别改进建议"""
        improvements = []
        
        avg_response_time = float(performance_results["响应时间统计"]["平均值"].split()[0])
        if avg_response_time > 0.1:
            improvements.append("优化检索算法以提高响应速度")
        
        success_rate = 100 - float(quality_results["总体统计"]["检索效果"]["无结果查询比例"].rstrip('%'))
        if success_rate < 80:
            improvements.append("扩充词典和改进分词算法以提高检索成功率")
        
        avg_similarity = quality_results["总体统计"]["检索效果"]["平均相似度"]
        if avg_similarity < 0.05:
            improvements.append("调整TF-IDF参数或引入语义相似度模型")
        
        return improvements if improvements else ["系统表现良好，可考虑增加更多高级功能"]
    
    def _generate_markdown_report(self, report_data: Dict, output_file: Path):
        """生成Markdown格式的评估报告"""
        markdown_content = f"""# 中文新闻检索系统评估报告

**生成时间**: {report_data['报告生成时间']}

## 1. 系统概览

- **文档数量**: {report_data['系统信息']['基础统计']['文档数量']}
- **词汇表大小**: {report_data['系统信息']['基础统计']['词汇表大小']}
- **索引构建时间**: {report_data['系统信息']['基础统计']['索引构建时间']}

## 2. 检索质量评估

### 2.1 总体表现
- **平均响应时间**: {report_data['检索质量评估']['总体统计']['性能指标']['平均响应时间']}
- **平均结果数量**: {report_data['检索质量评估']['总体统计']['检索效果']['平均结果数量']}
- **平均相似度**: {report_data['检索质量评估']['总体统计']['检索效果']['平均相似度']}
- **无结果查询比例**: {report_data['检索质量评估']['总体统计']['检索效果']['无结果查询比例']}

### 2.2 按查询类型统计
"""
        
        for query_type, stats in report_data['检索质量评估']['按类型统计'].items():
            markdown_content += f"""
**{query_type}**:
- 查询数量: {stats['查询数量']}
- 平均响应时间: {stats['平均响应时间']}
- 平均相似度: {stats['平均相似度']}
- 成功率: {stats['成功率']}
"""
        
        markdown_content += f"""
## 3. 系统性能评估

- **平均响应时间**: {report_data['系统性能评估']['响应时间统计']['平均值']}
- **95百分位响应时间**: {report_data['系统性能评估']['响应时间统计']['95百分位']}
- **平均QPS**: {report_data['系统性能评估']['吞吐量统计']['平均QPS']}
- **峰值QPS**: {report_data['系统性能评估']['吞吐量统计']['峰值QPS']}

## 4. 评估总结

- **整体评分**: {report_data['评估总结']['整体评分']}/100
- **质量评分**: {report_data['评估总结']['质量评分']}/100  
- **性能评分**: {report_data['评估总结']['性能评分']}/100

### 主要优势
"""
        
        for strength in report_data['评估总结']['主要优势']:
            markdown_content += f"- {strength}\n"
        
        markdown_content += "\n### 改进建议\n"
        for improvement in report_data['评估总结']['改进建议']:
            markdown_content += f"- {improvement}\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=== 评估模块测试 ===")
    
    # 这里需要一个已经构建好的搜索系统进行测试
    # 实际使用时需要传入真正的搜索系统实例
    
    print("评估模块实现完成！")
    print("使用方法:")
    print("1. 创建评估器: evaluator = RetrievalEvaluator(search_system)")
    print("2. 评估质量: results = evaluator.evaluate_retrieval_quality()")
    print("3. 评估性能: performance = evaluator.evaluate_system_performance()")
    print("4. 生成报告: report_path = evaluator.generate_evaluation_report()")