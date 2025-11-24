"""
倒排索引系统交互式演示程序
提供命令行界面，支持所有核心功能的交互式操作
"""
import sys
from pathlib import Path
from src.index_system import IndexSystem


class InteractiveDemo:
    """交互式演示程序"""
    
    def __init__(self):
        """初始化演示程序"""
        self.system = None
        self.running = True
        
    def print_header(self):
        """打印程序头部"""
        print("\n" + "=" * 70)
        print(" " * 20 + "倒排索引系统 - 交互式演示")
        print("=" * 70)
        print("\n欢迎使用倒排索引系统！")
        print("本系统支持中文文档的索引构建、查询和布尔检索功能。\n")
    
    def print_menu(self):
        """打印主菜单"""
        print("\n" + "-" * 70)
        print("主菜单：")
        print("-" * 70)
        print("  1. 初始化索引系统")
        print("  2. 添加文档")
        print("  3. 批量添加示例文档")
        print("  4. 单词项查询")
        print("  5. 布尔查询")
        print("  6. 查看文档内容")
        print("  7. 查看索引统计信息")
        print("  8. 保存索引")
        print("  9. 加载索引")
        print("  0. 退出程序")
        print("-" * 70)
    
    def initialize_system(self):
        """初始化索引系统"""
        print("\n>>> 初始化索引系统")
        
        # 询问是否使用停用词
        use_stopwords = input("是否使用停用词过滤？(y/n，默认y): ").strip().lower()
        
        stopwords_path = None
        if use_stopwords != 'n':
            default_path = "config/stopwords.txt"
            if Path(default_path).exists():
                stopwords_path = default_path
                print(f"使用停用词文件: {stopwords_path}")
            else:
                print(f"警告: 默认停用词文件 {default_path} 不存在，将不使用停用词过滤")
        
        try:
            self.system = IndexSystem(stopwords_path=stopwords_path)
            print("✓ 索引系统初始化成功！")
        except Exception as e:
            print(f"✗ 初始化失败: {str(e)}")
    
    def add_document(self):
        """添加单个文档"""
        if not self._check_system():
            return
        
        print("\n>>> 添加文档")
        print("请输入文档内容（输入空行结束）：")
        
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        
        if not lines:
            print("✗ 文档内容为空，取消添加")
            return
        
        content = " ".join(lines)
        
        # 询问是否添加元数据
        add_metadata = input("\n是否添加元数据？(y/n，默认n): ").strip().lower()
        metadata = None
        
        if add_metadata == 'y':
            title = input("  标题: ").strip()
            category = input("  分类: ").strip()
            metadata = {}
            if title:
                metadata['title'] = title
            if category:
                metadata['category'] = category
        
        try:
            doc_id = self.system.add_document(content, metadata)
            print(f"\n✓ 文档添加成功！")
            print(f"  文档ID: {doc_id}")
            print(f"  内容: {content[:50]}{'...' if len(content) > 50 else ''}")
            if metadata:
                print(f"  元数据: {metadata}")
        except Exception as e:
            print(f"✗ 添加文档失败: {str(e)}")
    
    def add_sample_documents(self):
        """批量添加示例文档"""
        if not self._check_system():
            return
        
        print("\n>>> 批量添加示例文档")
        
        # 准备示例文档
        sample_docs = [
            {
                "content": "Python是一种广泛使用的高级编程语言，以其简洁的语法和强大的功能而闻名",
                "metadata": {"title": "Python简介", "category": "编程语言"}
            },
            {
                "content": "Java是一种面向对象的编程语言，广泛应用于企业级应用开发",
                "metadata": {"title": "Java简介", "category": "编程语言"}
            },
            {
                "content": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习",
                "metadata": {"title": "机器学习概述", "category": "人工智能"}
            },
            {
                "content": "Python在数据科学和机器学习领域非常流行，拥有丰富的科学计算库",
                "metadata": {"title": "Python与数据科学", "category": "数据科学"}
            },
            {
                "content": "深度学习是机器学习的一个子领域，使用神经网络模型处理复杂问题",
                "metadata": {"title": "深度学习", "category": "人工智能"}
            },
            {
                "content": "自然语言处理技术使计算机能够理解和生成人类语言",
                "metadata": {"title": "自然语言处理", "category": "人工智能"}
            },
            {
                "content": "数据结构和算法是计算机科学的基础，对编程能力的提升至关重要",
                "metadata": {"title": "数据结构与算法", "category": "计算机科学"}
            },
            {
                "content": "倒排索引是信息检索系统的核心数据结构，用于快速查找文档",
                "metadata": {"title": "倒排索引", "category": "信息检索"}
            }
        ]
        
        print(f"准备添加 {len(sample_docs)} 个示例文档...")
        
        added_count = 0
        for i, doc_data in enumerate(sample_docs, 1):
            try:
                doc_id = self.system.add_document(
                    doc_data["content"],
                    doc_data["metadata"]
                )
                print(f"  ✓ [{i}/{len(sample_docs)}] 文档 {doc_id}: {doc_data['metadata']['title']}")
                added_count += 1
            except Exception as e:
                print(f"  ✗ [{i}/{len(sample_docs)}] 添加失败: {str(e)}")
        
        print(f"\n完成！成功添加 {added_count}/{len(sample_docs)} 个文档")
    
    def search_documents(self):
        """单词项查询"""
        if not self._check_system():
            return
        
        print("\n>>> 单词项查询")
        query = input("请输入查询词: ").strip()
        
        if not query:
            print("✗ 查询词不能为空")
            return
        
        try:
            results = self.system.search(query)
            
            print(f"\n查询: '{query}'")
            print(f"找到 {len(results)} 个文档")
            
            if results:
                print("\n结果列表:")
                for doc_id in results:
                    doc = self.system.get_document(doc_id)
                    if doc:
                        title = doc['metadata'].get('title', '无标题') if doc['metadata'] else '无标题'
                        content_preview = doc['content'][:60] + '...' if len(doc['content']) > 60 else doc['content']
                        print(f"  [{doc_id}] {title}")
                        print(f"      {content_preview}")
            else:
                print("  未找到包含该词项的文档")
        
        except Exception as e:
            print(f"✗ 查询失败: {str(e)}")
    
    def boolean_search(self):
        """布尔查询"""
        if not self._check_system():
            return
        
        print("\n>>> 布尔查询")
        print("支持的运算符: AND, OR, NOT")
        print("运算符优先级: NOT > AND > OR")
        print("示例: 'Python AND 机器学习', 'Java OR Python', 'Python AND NOT 数据'")
        
        query = input("\n请输入布尔查询: ").strip()
        
        if not query:
            print("✗ 查询不能为空")
            return
        
        try:
            results = self.system.boolean_search(query)
            
            print(f"\n查询: '{query}'")
            print(f"找到 {len(results)} 个文档")
            
            if results:
                print("\n结果列表:")
                for doc_id in results:
                    doc = self.system.get_document(doc_id)
                    if doc:
                        title = doc['metadata'].get('title', '无标题') if doc['metadata'] else '无标题'
                        content_preview = doc['content'][:60] + '...' if len(doc['content']) > 60 else doc['content']
                        print(f"  [{doc_id}] {title}")
                        print(f"      {content_preview}")
            else:
                print("  未找到符合条件的文档")
        
        except Exception as e:
            print(f"✗ 查询失败: {str(e)}")
    
    def view_document(self):
        """查看文档内容"""
        if not self._check_system():
            return
        
        print("\n>>> 查看文档内容")
        doc_id_str = input("请输入文档ID: ").strip()
        
        try:
            doc_id = int(doc_id_str)
        except ValueError:
            print("✗ 文档ID必须是数字")
            return
        
        try:
            doc = self.system.get_document(doc_id)
            
            if doc:
                print(f"\n文档详情:")
                print(f"  ID: {doc['doc_id']}")
                print(f"  内容: {doc['content']}")
                print(f"  词项数: {doc['token_count']}")
                print(f"  创建时间: {doc['created_at']}")
                if doc['metadata']:
                    print(f"  元数据:")
                    for key, value in doc['metadata'].items():
                        print(f"    {key}: {value}")
            else:
                print(f"✗ 文档 {doc_id} 不存在")
        
        except Exception as e:
            print(f"✗ 获取文档失败: {str(e)}")
    
    def view_statistics(self):
        """查看索引统计信息"""
        if not self._check_system():
            return
        
        print("\n>>> 索引统计信息")
        
        try:
            stats = self.system.get_statistics()
            
            print(f"\n基本统计:")
            print(f"  文档总数: {stats['document_count']}")
            print(f"  唯一词项数: {stats['vocabulary_size']}")
            print(f"  总词项数: {stats['total_tokens']}")
            print(f"  平均文档长度: {stats['avg_doc_length']:.2f} 个词项")
            
            # 显示词频信息
            if stats['term_frequencies']:
                print(f"\n词项文档频率（按频率降序，显示前20个）:")
                term_freqs = sorted(
                    stats['term_frequencies'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
                
                for i, (term, freq) in enumerate(term_freqs, 1):
                    print(f"  {i:2d}. '{term}': {freq} 个文档")
        
        except Exception as e:
            print(f"✗ 获取统计信息失败: {str(e)}")
    
    def save_index(self):
        """保存索引"""
        if not self._check_system():
            return
        
        print("\n>>> 保存索引")
        
        default_path = "data/saved_index"
        filepath = input(f"请输入保存路径（默认: {default_path}）: ").strip()
        
        if not filepath:
            filepath = default_path
        
        try:
            success = self.system.save_index(filepath)
            
            if success:
                print(f"✓ 索引已成功保存到: {filepath}")
                print(f"  生成文件:")
                print(f"    - {filepath}.index (倒排索引)")
                print(f"    - {filepath}.docs (文档存储)")
            else:
                print(f"✗ 索引保存失败")
        
        except Exception as e:
            print(f"✗ 保存索引时发生错误: {str(e)}")
    
    def load_index(self):
        """加载索引"""
        print("\n>>> 加载索引")
        
        default_path = "data/saved_index"
        filepath = input(f"请输入索引文件路径（默认: {default_path}）: ").strip()
        
        if not filepath:
            filepath = default_path
        
        # 检查文件是否存在
        index_file = Path(filepath).with_suffix('.index')
        docs_file = Path(filepath).with_suffix('.docs')
        
        if not index_file.exists() or not docs_file.exists():
            print(f"✗ 索引文件不存在:")
            if not index_file.exists():
                print(f"  - {index_file}")
            if not docs_file.exists():
                print(f"  - {docs_file}")
            return
        
        try:
            # 如果系统未初始化，先初始化
            if self.system is None:
                print("正在初始化索引系统...")
                self.system = IndexSystem()
            
            success = self.system.load_index(filepath)
            
            if success:
                print(f"✓ 索引已成功从 {filepath} 加载")
                
                # 显示加载的索引信息
                stats = self.system.get_statistics()
                print(f"\n加载的索引信息:")
                print(f"  文档总数: {stats['document_count']}")
                print(f"  唯一词项数: {stats['vocabulary_size']}")
            else:
                print(f"✗ 索引加载失败")
        
        except Exception as e:
            print(f"✗ 加载索引时发生错误: {str(e)}")
    
    def _check_system(self):
        """检查系统是否已初始化"""
        if self.system is None:
            print("\n✗ 索引系统未初始化，请先选择选项 1 初始化系统")
            return False
        return True
    
    def run(self):
        """运行交互式程序"""
        self.print_header()
        
        while self.running:
            self.print_menu()
            
            choice = input("\n请选择操作 (0-9): ").strip()
            
            if choice == '1':
                self.initialize_system()
            elif choice == '2':
                self.add_document()
            elif choice == '3':
                self.add_sample_documents()
            elif choice == '4':
                self.search_documents()
            elif choice == '5':
                self.boolean_search()
            elif choice == '6':
                self.view_document()
            elif choice == '7':
                self.view_statistics()
            elif choice == '8':
                self.save_index()
            elif choice == '9':
                self.load_index()
            elif choice == '0':
                print("\n感谢使用倒排索引系统，再见！")
                self.running = False
            else:
                print("\n✗ 无效的选项，请重新选择")
        
        print("\n" + "=" * 70)


def main():
    """主函数"""
    try:
        demo = InteractiveDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n程序发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
