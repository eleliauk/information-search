#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中文新闻稀疏检索系统演示程序
Quick Demo for Chinese News Sparse Retrieval System
"""

import sys
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.crawler.news_spider import NewsDataCollector
from src.retrieval.search_engine import ChineseNewsSearchSystem

def quick_demo():
    """快速演示"""
    print("🔍 中文新闻稀疏检索系统 - 快速演示")
    print("=" * 50)
    
    # 设置简单日志
    logging.basicConfig(level=logging.WARNING)
    
    print("📊 生成演示数据...")
    
    # 1. 生成少量演示数据
    collector = NewsDataCollector()
    news_data = collector.generate_mock_news_data(num_articles=50)
    
    print(f"✅ 生成了 {len(news_data)} 篇演示新闻")
    
    # 2. 构建搜索系统
    print("🔧 构建搜索系统...")
    search_system = ChineseNewsSearchSystem()
    
    result = search_system.index_documents(news_data, save_analysis=False)
    
    if not result["success"]:
        print(f"❌ 系统构建失败: {result['message']}")
        return
    
    print(f"✅ 系统构建完成!")
    print(f"   - 文档数量: {result['statistics']['文档数量']}")
    print(f"   - 词汇表大小: {result['statistics']['词汇表大小']}")
    
    # 3. 演示检索
    print("\n🔍 检索演示:")
    print("-" * 30)
    
    demo_queries = [
        "人工智能技术",
        "疫情防控",
        "绿色发展",
        "经济政策"
    ]
    
    for query in demo_queries:
        print(f"\n🔎 查询: '{query}'")
        
        try:
            results, search_time = search_system.search(query, top_k=3)
            print(f"⏱️  耗时: {search_time:.4f} 秒")
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"   {i}. {result['title'][:40]}...")
                    print(f"      相似度: {result['similarity_score']:.4f}")
                    print(f"      分类: {result['category']}")
            else:
                print("   未找到相关结果")
                
        except Exception as e:
            print(f"   ❌ 检索出错: {e}")
    
    # 4. 系统统计
    print(f"\n📊 系统统计:")
    print("-" * 20)
    stats = search_system.get_system_stats()
    basic_stats = stats.get("基础统计", {})
    
    print(f"文档数量: {basic_stats.get('文档数量', 0)}")
    print(f"词汇表大小: {basic_stats.get('词汇表大小', 0)}")
    print(f"平均检索时间: {basic_stats.get('平均检索时间', 'N/A')}")
    print(f"总检索次数: {basic_stats.get('总检索次数', 0)}")
    
    print("\n🎉 演示完成!")
    print("\n💡 如需完整功能，请运行: python main.py --mode full")

if __name__ == "__main__":
    try:
        quick_demo()
    except KeyboardInterrupt:
        print("\n\n👋 演示中断，再见!")
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        print("请检查系统环境和依赖是否正确安装")