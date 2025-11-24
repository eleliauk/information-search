#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional
import logging
import json
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RawDataAnalyzer:
    """原始数据统计分析器"""
    
    def __init__(self, news_data: List[Dict]):
        self.news_data = news_data
        self.analysis_results = {}
        self.logger = logging.getLogger(__name__)
    
    def comprehensive_analysis(self) -> Dict[str, Any]:
        """综合数据分析"""
        self.logger.info("开始进行综合数据分析...")
        
        self.analysis_results = {
            "基本统计": self.basic_statistics(),
            "文本长度分析": self.text_length_analysis(),
            "分类分布": self.category_distribution(),
            "时间分布": self.temporal_distribution(),
            "内容质量分析": self.content_quality_analysis()
        }
        
        self.logger.info("数据分析完成")
        return self.analysis_results
    
    def basic_statistics(self) -> Dict[str, Any]:
        """基本统计信息"""
        if not self.news_data:
            return {"错误": "没有数据可分析"}
        
        total_articles = len(self.news_data)
        total_words = sum(article.get('word_count', 0) for article in self.news_data)
        word_counts = [article.get('word_count', 0) for article in self.news_data]
        
        return {
            "文章总数": total_articles,
            "总字符数": total_words,
            "平均字符数": round(total_words / total_articles, 2) if total_articles > 0 else 0,
            "最长文章字符数": max(word_counts) if word_counts else 0,
            "最短文章字符数": min(word_counts) if word_counts else 0,
            "字符数标准差": round(np.std(word_counts), 2) if word_counts else 0
        }
    
    def text_length_analysis(self) -> Dict[str, Any]:
        """文本长度分析"""
        word_counts = [article.get('word_count', 0) for article in self.news_data]
        title_lengths = [len(article.get('title', '')) for article in self.news_data]
        
        # 长度分布区间
        length_bins = [0, 200, 500, 1000, 2000, 5000]
        length_distribution = {}
        
        for i in range(len(length_bins) - 1):
            count = sum(1 for wc in word_counts 
                       if length_bins[i] <= wc < length_bins[i+1])
            length_distribution[f"{length_bins[i]}-{length_bins[i+1]}"] = count
        
        # 处理超出最大区间的情况
        over_max = sum(1 for wc in word_counts if wc >= length_bins[-1])
        if over_max > 0:
            length_distribution[f"{length_bins[-1]}+"] = over_max
        
        return {
            "正文长度分布": length_distribution,
            "标题长度统计": {
                "平均长度": round(np.mean(title_lengths), 2) if title_lengths else 0,
                "最长标题": max(title_lengths) if title_lengths else 0,
                "最短标题": min(title_lengths) if title_lengths else 0,
                "标准差": round(np.std(title_lengths), 2) if title_lengths else 0
            },
            "正文长度百分位数": {
                "25%": round(np.percentile(word_counts, 25), 2) if word_counts else 0,
                "50%": round(np.percentile(word_counts, 50), 2) if word_counts else 0,
                "75%": round(np.percentile(word_counts, 75), 2) if word_counts else 0,
                "90%": round(np.percentile(word_counts, 90), 2) if word_counts else 0
            }
        }
    
    def category_distribution(self) -> Dict[str, Any]:
        """分类分布分析"""
        categories = [article.get('category', '未分类') for article in self.news_data]
        category_counts = Counter(categories)
        
        total = len(categories)
        category_stats = {}
        
        for category, count in category_counts.items():
            # 计算该分类下文章的平均字符数
            category_articles = [article for article in self.news_data 
                               if article.get('category') == category]
            avg_word_count = np.mean([a.get('word_count', 0) for a in category_articles])
            
            category_stats[category] = {
                "文章数量": count,
                "占比": f"{(count/total)*100:.2f}%",
                "平均字符数": round(avg_word_count, 2)
            }
        
        return category_stats
    
    def temporal_distribution(self) -> Dict[str, Any]:
        """时间分布分析"""
        # 解析发布时间
        publish_times = []
        for article in self.news_data:
            pub_time_str = article.get('publish_time')
            if pub_time_str:
                try:
                    # 尝试多种时间格式
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
                        try:
                            pub_time = datetime.strptime(pub_time_str, fmt)
                            publish_times.append(pub_time)
                            break
                        except ValueError:
                            continue
                except:
                    continue
        
        if not publish_times:
            return {"错误": "无有效时间数据"}
        
        # 按日期分组
        dates = [pt.date() for pt in publish_times]
        date_counts = Counter(dates)
        
        # 按小时分组
        hour_counts = Counter(pt.hour for pt in publish_times)
        
        # 按星期分组
        weekday_counts = Counter(pt.weekday() for pt in publish_times)
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        weekday_dist = {weekday_names[i]: weekday_counts.get(i, 0) for i in range(7)}
        
        return {
            "时间跨度": {
                "最早": min(publish_times).strftime('%Y-%m-%d %H:%M:%S'),
                "最晚": max(publish_times).strftime('%Y-%m-%d %H:%M:%S'),
                "跨度天数": (max(publish_times) - min(publish_times)).days
            },
            "每日发布量": {str(date): count for date, count in sorted(date_counts.items())},
            "每小时发布量": dict(sorted(hour_counts.items())),
            "每周发布量": weekday_dist,
            "发布高峰时段": max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else None
        }
    
    def content_quality_analysis(self) -> Dict[str, Any]:
        """内容质量分析"""
        # 检查重复内容
        title_similarities = self.calculate_title_similarities()
        
        # 检查内容完整性
        incomplete_articles = [
            article for article in self.news_data
            if not article.get('content') or len(article.get('content', '')) < 50
        ]
        
        # 检查特殊字符
        special_char_articles = []
        for article in self.news_data:
            content = article.get('content', '')
            if any(char in content for char in ['□', '■', '???', '乱码']):
                special_char_articles.append(article)
        
        # 检查标题长度异常
        title_length_issues = []
        for article in self.news_data:
            title_len = len(article.get('title', ''))
            if title_len < 5 or title_len > 150:
                title_length_issues.append(article)
        
        return {
            "数据完整性": {
                "完整文章数": len(self.news_data) - len(incomplete_articles),
                "不完整文章数": len(incomplete_articles),
                "完整率": f"{((len(self.news_data) - len(incomplete_articles)) / len(self.news_data)) * 100:.2f}%"
            },
            "内容质量": {
                "疑似重复标题数": len([s for s in title_similarities if s > 0.8]),
                "包含特殊字符文章数": len(special_char_articles),
                "标题长度异常数": len(title_length_issues),
                "质量评分": self.calculate_quality_score()
            },
            "重复度分析": {
                "高相似度标题对": len([s for s in title_similarities if s > 0.8]),
                "中等相似度标题对": len([s for s in title_similarities if 0.5 < s <= 0.8]),
                "平均标题相似度": round(np.mean(title_similarities), 4) if title_similarities else 0
            }
        }
    
    def calculate_title_similarities(self) -> List[float]:
        """计算标题相似度"""
        titles = [article.get('title', '') for article in self.news_data]
        similarities = []
        
        for i in range(len(titles)):
            for j in range(i+1, len(titles)):
                if titles[i] and titles[j]:
                    similarity = SequenceMatcher(None, titles[i], titles[j]).ratio()
                    similarities.append(similarity)
        
        return similarities
    
    def calculate_quality_score(self) -> float:
        """计算数据质量评分"""
        score = 100.0
        
        if not self.news_data:
            return 0.0
        
        # 根据各种质量指标扣分
        incomplete_rate = len([a for a in self.news_data 
                             if len(a.get('content', '')) < 100]) / len(self.news_data)
        score -= incomplete_rate * 30
        
        # 长度异常扣分
        word_counts = [a.get('word_count', 0) for a in self.news_data]
        if word_counts:
            length_variance = np.var(word_counts)
            if length_variance > 1000000:  # 长度差异过大
                score -= 10
        
        # 重复内容扣分
        title_similarities = self.calculate_title_similarities()
        if title_similarities:
            high_similarity_rate = len([s for s in title_similarities if s > 0.8]) / len(title_similarities)
            score -= high_similarity_rate * 20
        
        return max(0.0, round(score, 2))


class DataVisualizer:
    """数据可视化类"""
    
    def __init__(self, analysis_results: Dict[str, Any], save_path: str = "data/analysis"):
        self.analysis_results = analysis_results
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_all_visualizations(self) -> None:
        """创建所有可视化图表"""
        self.logger.info("开始创建可视化图表...")
        
        self.plot_length_distribution()
        self.plot_category_distribution()
        self.plot_temporal_distribution()
        self.plot_quality_metrics()
        
        self.logger.info("所有图表创建完成")
    
    def plot_length_distribution(self) -> None:
        """绘制文章长度分布图"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 正文长度分布
            text_analysis = self.analysis_results.get("文本长度分析", {})
            length_dist = text_analysis.get("正文长度分布", {})
            
            if length_dist:
                categories = list(length_dist.keys())
                values = list(length_dist.values())
                
                bars = ax1.bar(categories, values, color='skyblue', alpha=0.7)
                ax1.set_title('文章正文长度分布', fontsize=14, fontweight='bold')
                ax1.set_xlabel('字符数区间')
                ax1.set_ylabel('文章数量')
                ax1.tick_params(axis='x', rotation=45)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')
            
            # 2. 长度百分位数箱线图
            percentiles = text_analysis.get("正文长度百分位数", {})
            if percentiles:
                percentile_values = [percentiles.get("25%", 0), percentiles.get("50%", 0),
                                   percentiles.get("75%", 0), percentiles.get("90%", 0)]
                
                ax2.boxplot([percentile_values], labels=['正文长度'])
                ax2.set_title('文章长度百分位数分布')
                ax2.set_ylabel('字符数')
            
            # 3. 标题长度统计
            title_stats = text_analysis.get("标题长度统计", {})
            if title_stats:
                metrics = ['平均长度', '最长标题', '最短标题']
                values = [title_stats.get(metric, 0) for metric in metrics]
                
                bars = ax3.bar(metrics, values, color='lightcoral', alpha=0.7)
                ax3.set_title('标题长度统计')
                ax3.set_ylabel('字符数')
                
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom')
            
            # 4. 基本统计概览
            basic_stats = self.analysis_results.get("基本统计", {})
            if basic_stats:
                stats_labels = ['文章总数', '平均字符数', '字符数标准差']
                stats_values = [basic_stats.get('文章总数', 0),
                              basic_stats.get('平均字符数', 0),
                              basic_stats.get('字符数标准差', 0)]
                
                ax4.bar(stats_labels, stats_values, color='lightgreen', alpha=0.7)
                ax4.set_title('基本统计信息')
                ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.save_path / 'length_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"绘制长度分布图失败: {e}")
    
    def plot_category_distribution(self) -> None:
        """绘制分类分布图"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            category_dist = self.analysis_results.get("分类分布", {})
            
            if category_dist:
                categories = list(category_dist.keys())
                counts = [category_dist[cat]["文章数量"] for cat in categories]
                avg_lengths = [category_dist[cat]["平均字符数"] for cat in categories]
                
                # 饼图
                wedges, texts, autotexts = ax1.pie(counts, labels=categories, autopct='%1.1f%%', 
                                                  startangle=90, colors=sns.color_palette("husl", len(categories)))
                ax1.set_title('新闻分类分布', fontsize=14, fontweight='bold')
                
                # 柱状图：各分类平均字符数
                bars = ax2.bar(categories, avg_lengths, color=sns.color_palette("husl", len(categories)), alpha=0.7)
                ax2.set_title('各分类平均字符数')
                ax2.set_xlabel('分类')
                ax2.set_ylabel('平均字符数')
                ax2.tick_params(axis='x', rotation=45)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.0f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.save_path / 'category_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"绘制分类分布图失败: {e}")
    
    def plot_temporal_distribution(self) -> None:
        """绘制时间分布图"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            temporal_data = self.analysis_results.get("时间分布", {})
            
            # 1. 每小时发布量
            hour_data = temporal_data.get("每小时发布量", {})
            if hour_data:
                hours = list(hour_data.keys())
                counts = list(hour_data.values())
                
                ax1.bar(hours, counts, color='lightblue', alpha=0.7)
                ax1.set_title('每小时新闻发布量分布')
                ax1.set_xlabel('小时')
                ax1.set_ylabel('文章数量')
                ax1.set_xticks(range(0, 24, 2))
                ax1.grid(axis='y', alpha=0.3)
            
            # 2. 每周发布量
            weekday_data = temporal_data.get("每周发布量", {})
            if weekday_data:
                weekdays = list(weekday_data.keys())
                counts = list(weekday_data.values())
                
                ax2.bar(weekdays, counts, color='lightcoral', alpha=0.7)
                ax2.set_title('每周发布量分布')
                ax2.set_xlabel('星期')
                ax2.set_ylabel('文章数量')
                ax2.tick_params(axis='x', rotation=45)
            
            # 3. 发布时间趋势（如果有足够的数据）
            daily_data = temporal_data.get("每日发布量", {})
            if daily_data and len(daily_data) > 1:
                dates = list(daily_data.keys())[:30]  # 只显示最近30天
                counts = [daily_data[date] for date in dates]
                
                ax3.plot(range(len(dates)), counts, marker='o', linewidth=2, markersize=4)
                ax3.set_title('每日发布量趋势（最近30天）')
                ax3.set_xlabel('日期')
                ax3.set_ylabel('文章数量')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)
            
            # 4. 时间跨度信息
            time_span = temporal_data.get("时间跨度", {})
            if time_span:
                info_text = f"数据时间跨度：{time_span.get('跨度天数', 0)}天\n"
                info_text += f"最早：{time_span.get('最早', 'N/A')}\n"
                info_text += f"最晚：{time_span.get('最晚', 'N/A')}\n"
                if temporal_data.get("发布高峰时段"):
                    info_text += f"发布高峰：{temporal_data['发布高峰时段']}时"
                
                ax4.text(0.1, 0.5, info_text, fontsize=12, transform=ax4.transAxes,
                        verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                ax4.set_title('时间统计信息')
                ax4.axis('off')
            
            plt.tight_layout()
            plt.savefig(self.save_path / 'temporal_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"绘制时间分布图失败: {e}")
    
    def plot_quality_metrics(self) -> None:
        """绘制质量评估图"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            quality_data = self.analysis_results.get("内容质量分析", {})
            
            # 1. 数据完整性
            completeness = quality_data.get("数据完整性", {})
            if completeness:
                labels = ['完整文章', '不完整文章']
                sizes = [completeness.get("完整文章数", 0), completeness.get("不完整文章数", 0)]
                colors = ['lightgreen', 'lightcoral']
                
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                ax1.set_title('数据完整性分布')
            
            # 2. 内容质量指标
            content_quality = quality_data.get("内容质量", {})
            if content_quality:
                metrics = ['疑似重复标题数', '包含特殊字符文章数', '标题长度异常数']
                values = [content_quality.get(metric, 0) for metric in metrics]
                
                bars = ax2.bar(metrics, values, color=['orange', 'red', 'purple'], alpha=0.7)
                ax2.set_title('内容质量问题统计')
                ax2.set_ylabel('文章数量')
                ax2.tick_params(axis='x', rotation=45)
                
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')
            
            # 3. 质量评分
            quality_score = content_quality.get("质量评分", 0) if content_quality else 0
            
            # 创建圆形进度条样式的质量评分
            theta = np.linspace(0, 2*np.pi, 100)
            r = 1
            ax3.plot(theta, [r]*100, 'lightgray', linewidth=10)
            
            score_theta = np.linspace(0, 2*np.pi * (quality_score/100), int(quality_score))
            if len(score_theta) > 0:
                ax3.plot(score_theta, [r]*len(score_theta), 'green', linewidth=10)
            
            ax3.text(0, 0, f'{quality_score:.1f}', ha='center', va='center', fontsize=20, fontweight='bold')
            ax3.set_xlim(-1.5, 1.5)
            ax3.set_ylim(-1.5, 1.5)
            ax3.set_aspect('equal')
            ax3.axis('off')
            ax3.set_title('数据质量评分')
            
            # 4. 重复度分析
            duplicate_analysis = quality_data.get("重复度分析", {})
            if duplicate_analysis:
                categories = ['高相似度', '中等相似度', '低相似度']
                high_sim = duplicate_analysis.get("高相似度标题对", 0)
                mid_sim = duplicate_analysis.get("中等相似度标题对", 0)
                # 估算低相似度对数
                total_pairs = len(self.analysis_results.get("基本统计", {}).get("文章总数", 0))
                if total_pairs > 0:
                    total_pairs = total_pairs * (total_pairs - 1) // 2
                    low_sim = max(0, total_pairs - high_sim - mid_sim)
                else:
                    low_sim = 0
                
                values = [high_sim, mid_sim, low_sim]
                colors = ['red', 'orange', 'green']
                
                ax4.bar(categories, values, color=colors, alpha=0.7)
                ax4.set_title('标题相似度分布')
                ax4.set_ylabel('标题对数量')
                ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.save_path / 'quality_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"绘制质量评估图失败: {e}")
    
    def save_analysis_report(self, filename: str = "analysis_report.json") -> str:
        """保存分析报告为JSON文件"""
        try:
            report_path = self.save_path / filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"分析报告已保存到: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"保存分析报告失败: {e}")
            return ""


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    test_data = [
        {
            "article_id": "test1",
            "title": "人工智能技术发展迅速",
            "content": "人工智能技术在各个领域都取得了显著进展，特别是在机器学习和深度学习方面。" * 10,
            "category": "科技",
            "publish_time": "2024-01-15 10:30:00",
            "word_count": 500
        },
        {
            "article_id": "test2",
            "title": "经济形势持续向好",
            "content": "今年以来，国民经济运行总体平稳，主要指标保持在合理区间内。" * 15,
            "category": "经济",
            "publish_time": "2024-01-15 14:20:00",
            "word_count": 750
        }
    ]
    
    # 执行分析
    analyzer = RawDataAnalyzer(test_data)
    results = analyzer.comprehensive_analysis()
    
    # 可视化
    visualizer = DataVisualizer(results)
    visualizer.create_all_visualizations()
    
    # 保存报告
    report_path = visualizer.save_analysis_report()
    
    print("数据分析完成！")
    print(f"分析结果: {json.dumps(results, ensure_ascii=False, indent=2)}")