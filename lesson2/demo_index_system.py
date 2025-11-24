"""
IndexSystem 演示程序
展示倒排索引系统的主要功能
"""
from src.index_system import IndexSystem


def main():
    print("=" * 60)
    print("倒排索引系统演示")
    print("=" * 60)
    
    # 1. 初始化索引系统
    print("\n1. 初始化索引系统...")
    system = IndexSystem()
    print("✓ 索引系统初始化成功")
    
    # 2. 添加文档
    print("\n2. 添加文档...")
    documents = [
        "Python 是一门流行的编程语言",
        "Java 也是一门广泛使用的编程语言",
        "Python 在数据科学领域很受欢迎",
        "机器学习通常使用 Python 实现",
        "Java 适合开发企业级应用"
    ]
    
    doc_ids = []
    for i, doc in enumerate(documents, 1):
        doc_id = system.add_document(doc, metadata={"title": f"文档{i}"})
        doc_ids.append(doc_id)
        print(f"  ✓ 文档 {doc_id}: {doc}")
    
    # 3. 单词项查询
    print("\n3. 单词项查询...")
    queries = ["Python", "Java", "编程语言"]
    
    for query in queries:
        results = system.search(query)
        print(f"  查询 '{query}': 找到 {len(results)} 个文档 -> {results}")
    
    # 4. 布尔查询
    print("\n4. 布尔查询...")
    boolean_queries = [
        "Python AND 数据",
        "Python OR Java",
        "编程语言 AND NOT Java"
    ]
    
    for query in boolean_queries:
        results = system.boolean_search(query)
        print(f"  查询 '{query}': 找到 {len(results)} 个文档 -> {results}")
    
    # 5. 获取统计信息
    print("\n5. 索引统计信息...")
    stats = system.get_statistics()
    print(f"  文档总数: {stats['document_count']}")
    print(f"  唯一词项数: {stats['vocabulary_size']}")
    print(f"  总词项数: {stats['total_tokens']}")
    print(f"  平均文档长度: {stats['avg_doc_length']:.2f}")
    
    # 显示部分词频信息
    print("\n  词项文档频率（前10个）:")
    term_freqs = sorted(stats['term_frequencies'].items(), 
                       key=lambda x: x[1], 
                       reverse=True)[:10]
    for term, freq in term_freqs:
        print(f"    '{term}': {freq} 个文档")
    
    # 6. 获取文档内容
    print("\n6. 获取文档内容...")
    doc = system.get_document(1)
    if doc:
        print(f"  文档 ID: {doc['doc_id']}")
        print(f"  内容: {doc['content']}")
        print(f"  元数据: {doc['metadata']}")
        print(f"  词项数: {doc['token_count']}")
    
    # 7. 保存和加载索引
    print("\n7. 索引持久化...")
    index_path = "data/demo_index"
    
    # 保存索引
    success = system.save_index(index_path)
    if success:
        print(f"  ✓ 索引已保存到: {index_path}")
    else:
        print(f"  ✗ 索引保存失败")
        return
    
    # 加载索引
    print("\n8. 加载索引...")
    new_system = IndexSystem()
    success = new_system.load_index(index_path)
    if success:
        print(f"  ✓ 索引已从 {index_path} 加载")
        
        # 验证加载的索引
        results = new_system.search("Python")
        print(f"  验证查询 'Python': 找到 {len(results)} 个文档 -> {results}")
        
        stats = new_system.get_statistics()
        print(f"  验证统计: 文档总数 = {stats['document_count']}")
    else:
        print(f"  ✗ 索引加载失败")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
