#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中文新闻稀疏检索系统主程序
Chinese News Sparse Retrieval System Main Program
"""

import sys
import os
import logging
import argparse
import time
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.crawler.news_spider import NewsDataCollector, DataStorage
from src.preprocessing.tokenizer import OptimizedChineseTokenizer
from src.preprocessing.analyzer import RawDataAnalyzer, DataVisualizer
from src.preprocessing.tokenizer import TokenizationAnalyzer
from src.retrieval.search_engine import ChineseNewsSearchSystem, SearchInterface
from src.evaluation.metrics import RetrievalEvaluator

def setup_logging(log_level: str = "INFO"):
    """设置日志配置"""
    # 创建logs目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 配置日志
    log_file = log_dir / f"system_{int(time.time())}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("日志系统初始化完成")
    return logger

def collect_news_data(max_articles: int = 500, use_mock: bool = True) -> list:
    """收集新闻数据"""
    logger = logging.getLogger(__name__)
    logger.info(f"开始收集新闻数据，目标数量: {max_articles}")
    
    # 初始化数据收集器和存储器
    collector = NewsDataCollector()
    storage = DataStorage()
    
    # 检查是否有已存在的数据
    latest_data_file = storage.get_latest_raw_data()
    
    if latest_data_file:
        print(f"发现已存在的数据文件: {latest_data_file}")
        choice = input("是否使用已存在的数据？(y/n): ").lower().strip()
        
        if choice == 'y':
            news_data = storage.load_raw_data(latest_data_file)
            logger.info(f"加载已存在数据，数量: {len(news_data)}")
            return news_data
    
    # 收集新数据
    news_data = collector.collect_news(max_articles=max_articles, use_mock=use_mock)
    
    # 保存数据
    saved_path = storage.save_raw_data(news_data)
    logger.info(f"新闻数据已保存到: {saved_path}")
    
    return news_data

def analyze_data(news_data: list, save_visualizations: bool = True) -> tuple:
    """分析数据"""
    logger = logging.getLogger(__name__)
    logger.info("开始数据分析...")
    
    # 1. 原始数据分析
    print("\n📊 执行原始数据分析...")
    data_analyzer = RawDataAnalyzer(news_data)
    data_analysis = data_analyzer.comprehensive_analysis()
    
    # 2. 分词分析
    print("\n✂️  执行分词分析...")
    tokenizer = OptimizedChineseTokenizer(
        custom_dict_path="config/news_dict.txt",
        stopwords_path="config/stopwords.txt"
    )
    
    tokenization_analyzer = TokenizationAnalyzer(tokenizer)
    tokenization_analysis = tokenization_analyzer.analyze_tokenization_results(news_data)
    
    # 3. 生成可视化图表
    if save_visualizations:
        print("\n📈 生成可视化图表...")
        try:
            # 数据分析可视化
            data_visualizer = DataVisualizer(data_analysis)
            data_visualizer.create_all_visualizations()
            data_visualizer.save_analysis_report()
            
            logger.info("数据可视化图表已生成")
        except Exception as e:
            logger.warning(f"生成可视化图表失败: {e}")
    
    return data_analysis, tokenization_analysis, tokenizer

def build_search_system(news_data: list, tokenizer) -> ChineseNewsSearchSystem:
    """构建检索系统"""
    logger = logging.getLogger(__name__)
    logger.info("开始构建检索系统...")
    
    print("\n🔧 构建检索系统...")
    
    # 创建搜索系统
    search_system = ChineseNewsSearchSystem(
        custom_dict_path="config/news_dict.txt",
        stopwords_path="config/stopwords.txt"
    )
    
    # 构建索引
    result = search_system.index_documents(news_data)
    
    if result["success"]:
        print(f"✅ 索引构建成功!")
        print(f"   - 文档数量: {result['statistics']['文档数量']}")
        print(f"   - 构建时间: {result['statistics']['构建时间']}")
        print(f"   - 词汇表大小: {result['statistics']['词汇表大小']}")
        print(f"   - 内存使用: {result['statistics']['内存使用']}")
    else:
        print(f"❌ 索引构建失败: {result['message']}")
        return None
    
    return search_system

def evaluate_system(search_system: ChineseNewsSearchSystem) -> dict:
    """评估系统性能"""
    logger = logging.getLogger(__name__)
    logger.info("开始系统评估...")
    
    print("\n📋 执行系统评估...")
    
    # 创建评估器
    evaluator = RetrievalEvaluator(search_system)
    
    # 执行评估
    print("   - 检索质量评估...")
    quality_results = evaluator.evaluate_retrieval_quality()
    
    print("   - 系统性能评估...")
    performance_results = evaluator.evaluate_system_performance(iterations=20)
    
    print("   - 配置基准测试...")
    benchmark_results = evaluator.benchmark_different_configurations()
    
    # 生成完整报告
    print("   - 生成评估报告...")
    report_path = evaluator.generate_evaluation_report()
    
    print(f"✅ 评估完成，报告已保存到: {report_path}")
    
    return {
        "quality": quality_results,
        "performance": performance_results,
        "benchmark": benchmark_results,
        "report_path": report_path
    }

def run_interactive_demo(search_system: ChineseNewsSearchSystem):
    """运行交互式演示"""
    print("\n🚀 启动交互式检索演示...")
    
    # 创建交互接口
    interface = SearchInterface(search_system)
    
    # 运行交互式搜索
    interface.interactive_search()

def run_batch_test(search_system: ChineseNewsSearchSystem):
    """运行批量测试"""
    print("\n🧪 运行批量测试...")
    
    test_queries = [
        "人工智能",
        "新冠疫情防控", 
        "碳达峰碳中和",
        "经济发展政策",
        "机器学习算法",
        "疫苗接种",
        "绿色发展",
        "数字经济",
        "智能制造",
        "5G网络技术"
    ]
    
    print(f"测试查询数量: {len(test_queries)}")
    print("-" * 50)
    
    total_time = 0
    total_results = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i:2d}. 查询: '{query}'")
        
        try:
            results, search_time = search_system.search(query, top_k=3)
            total_time += search_time
            total_results += len(results)
            
            print(f"    检索时间: {search_time:.4f}秒, 结果数: {len(results)}")
            
            if results:
                for j, result in enumerate(results, 1):
                    print(f"    {j}. {result['title'][:50]}... (相似度: {result['similarity_score']:.4f})")
            else:
                print("    无相关结果")
                
        except Exception as e:
            print(f"    ❌ 检索失败: {e}")
        
        print()
    
    print("=" * 50)
    print(f"批量测试完成:")
    print(f"  - 总查询数: {len(test_queries)}")
    print(f"  - 总耗时: {total_time:.4f} 秒")
    print(f"  - 平均响应时间: {total_time/len(test_queries):.4f} 秒")
    print(f"  - 平均结果数: {total_results/len(test_queries):.1f}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="中文新闻稀疏检索系统")
    parser.add_argument("--mode", choices=["full", "demo", "test", "collect", "analyze"], 
                       default="full", help="运行模式")
    parser.add_argument("--articles", type=int, default=500, help="爬取文章数量")
    parser.add_argument("--mock", action="store_true", help="使用模拟数据")
    parser.add_argument("--no-viz", action="store_true", help="不生成可视化图表")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    parser.add_argument("--save-index", type=str, help="保存索引文件路径")
    parser.add_argument("--load-index", type=str, help="加载索引文件路径")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    print("🔍 中文新闻稀疏检索系统")
    print("=" * 50)
    print(f"运行模式: {args.mode}")
    print(f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    try:
        if args.mode == "collect":
            # 仅数据收集模式
            news_data = collect_news_data(args.articles, args.mock)
            print(f"✅ 数据收集完成，共收集 {len(news_data)} 篇文章")
            
        elif args.mode == "analyze":
            # 仅数据分析模式
            storage = DataStorage()
            latest_data_file = storage.get_latest_raw_data()
            
            if not latest_data_file:
                print("❌ 未找到数据文件，请先运行数据收集")
                return
            
            news_data = storage.load_raw_data(latest_data_file)
            data_analysis, tokenization_analysis, tokenizer = analyze_data(news_data, not args.no_viz)
            print("✅ 数据分析完成")
            
        else:
            # 完整流程或演示模式
            
            # 1. 数据收集
            if args.load_index:
                # 如果要加载索引，先尝试从索引文件获取数据
                print(f"🔄 加载索引文件: {args.load_index}")
                search_system = ChineseNewsSearchSystem(
                    custom_dict_path="config/news_dict.txt",
                    stopwords_path="config/stopwords.txt"
                )
                
                if search_system.load_index(args.load_index):
                    print("✅ 索引加载成功")
                    news_data = search_system.documents
                else:
                    print("❌ 索引加载失败，转为重新构建")
                    news_data = collect_news_data(args.articles, args.mock)
                    search_system = None
            else:
                news_data = collect_news_data(args.articles, args.mock)
                search_system = None
            
            # 2. 数据分析
            if args.mode == "full":
                data_analysis, tokenization_analysis, tokenizer = analyze_data(news_data, not args.no_viz)
            
            # 3. 构建搜索系统
            if search_system is None:
                search_system = build_search_system(news_data, tokenizer if 'tokenizer' in locals() else None)
            
            if search_system is None:
                print("❌ 搜索系统构建失败")
                return
            
            # 4. 保存索引
            if args.save_index:
                if search_system.save_index(args.save_index):
                    print(f"✅ 索引已保存到: {args.save_index}")
                else:
                    print("❌ 索引保存失败")
            
            # 5. 系统评估
            if args.mode == "full":
                evaluation_results = evaluate_system(search_system)
            
            # 6. 运行演示或测试
            if args.mode in ["demo", "full"]:
                print("\n选择运行模式:")
                print("1. 交互式检索演示")
                print("2. 批量测试")
                print("3. 跳过演示")
                
                choice = input("请选择 (1/2/3): ").strip()
                
                if choice == "1":
                    run_interactive_demo(search_system)
                elif choice == "2":
                    run_batch_test(search_system)
                elif choice == "3":
                    print("跳过演示")
                else:
                    print("无效选择，跳过演示")
            
            elif args.mode == "test":
                run_batch_test(search_system)
        
        print(f"\n🎉 程序执行完成! 总耗时: {time.time() - start_time:.2f} 秒")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        print(f"\n❌ 程序执行出错: {e}")
    finally:
        print("\n👋 感谢使用中文新闻稀疏检索系统!")

if __name__ == "__main__":
    start_time = time.time()
    main()