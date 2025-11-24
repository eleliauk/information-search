"""
布尔查询功能演示
"""
from src.document_store import DocumentStore
from src.text_preprocessor import TextPreprocessor
from src.inverted_index import InvertedIndex
from src.query_processor import QueryProcessor


def main():
    """演示布尔查询功能"""
    print("=" * 60)
    print("倒排索引系统 - 布尔查询功能演示")
    print("=" * 60)
    
    # 1. 初始化系统组件
    print("\n1. 初始化系统组件...")
    doc_store = DocumentStore()
    preprocessor = TextPreprocessor()
    index = InvertedIndex()
    query_processor = QueryProcessor(index, preprocessor, doc_store)
    
    # 2. 添加示例文档
    print("\n2. 添加示例文档...")
    documents = [
        "Python是一种流行的编程语言",
        "Java也是一种编程语言",
        "Python在机器学习领域很流行",
        "机器学习使用Python和数学",
        "数据科学需要统计学知识"
    ]
    
    doc_ids = []
    for i, content in enumerate(documents, 1):
        doc_id = doc_store.add_document(content)
        doc_ids.append(doc_id)
        print(f"   文档 {doc_id}: {content}")
        
        # 构建索引
        tokens = preprocessor.tokenize(content)
        index.build_index(doc_id, tokens)
    
    # 3. 演示单词项查询
    print("\n3. 单词项查询示例")
    print("-" * 60)
    query = "Python"
    results = query_processor.search(query)
    print(f"查询: '{query}'")
    print(f"结果: 文档 {results}")
    for doc_id in results:
        doc = doc_store.get_document(doc_id)
        print(f"   - 文档 {doc_id}: {doc.content}")
    
    # 4. 演示 AND 查询
    print("\n4. AND 查询示例")
    print("-" * 60)
    query = "Python AND 编程语言"
    results = query_processor.boolean_search(query)
    print(f"查询: '{query}'")
    print(f"结果: 文档 {results}")
    for doc_id in results:
        doc = doc_store.get_document(doc_id)
        print(f"   - 文档 {doc_id}: {doc.content}")
    
    # 5. 演示 OR 查询
    print("\n5. OR 查询示例")
    print("-" * 60)
    query = "Java OR 统计学"
    results = query_processor.boolean_search(query)
    print(f"查询: '{query}'")
    print(f"结果: 文档 {results}")
    for doc_id in results:
        doc = doc_store.get_document(doc_id)
        print(f"   - 文档 {doc_id}: {doc.content}")
    
    # 6. 演示 NOT 查询
    print("\n6. NOT 查询示例")
    print("-" * 60)
    query = "Python AND NOT 编程语言"
    results = query_processor.boolean_search(query)
    print(f"查询: '{query}'")
    print(f"结果: 文档 {results}")
    for doc_id in results:
        doc = doc_store.get_document(doc_id)
        print(f"   - 文档 {doc_id}: {doc.content}")
    
    # 7. 演示复杂布尔查询
    print("\n7. 复杂布尔查询示例")
    print("-" * 60)
    query = "Python AND 机器学习 OR Java"
    results = query_processor.boolean_search(query)
    print(f"查询: '{query}'")
    print(f"说明: 运算符优先级 NOT > AND > OR")
    print(f"解析为: (Python AND 机器学习) OR Java")
    print(f"结果: 文档 {results}")
    for doc_id in results:
        doc = doc_store.get_document(doc_id)
        print(f"   - 文档 {doc_id}: {doc.content}")
    
    # 8. 演示只有排除词的 NOT 查询
    print("\n8. 纯 NOT 查询示例")
    print("-" * 60)
    query = "NOT Python"
    results = query_processor.boolean_search(query)
    print(f"查询: '{query}'")
    print(f"结果: 文档 {results}")
    for doc_id in results:
        doc = doc_store.get_document(doc_id)
        print(f"   - 文档 {doc_id}: {doc.content}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
