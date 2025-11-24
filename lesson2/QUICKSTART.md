# 快速入门指南

本指南将帮助您快速上手倒排索引系统。

## 5分钟快速体验

### 1. 安装依赖

```bash
cd lesson2
pip install -r requirements.txt
```

### 2. 运行交互式演示

```bash
python demo.py
```

按照提示操作：
1. 选择 `1` 初始化系统
2. 选择 `3` 批量添加示例文档
3. 选择 `4` 尝试单词项查询（例如：输入 "Python"）
4. 选择 `5` 尝试布尔查询（例如：输入 "Python AND 机器学习"）
5. 选择 `7` 查看索引统计信息

### 3. 运行自动演示

如果您想快速查看系统功能，可以运行自动演示程序：

```bash
# 完整功能演示
python demo_index_system.py

# 布尔查询专项演示
python demo_boolean_query.py
```

## 基础使用示例

### 示例 1: 构建简单索引

```python
from src.index_system import IndexSystem

# 初始化系统
system = IndexSystem()

# 添加文档
system.add_document("Python是一种流行的编程语言")
system.add_document("Java也是一种编程语言")
system.add_document("Python在数据科学领域很受欢迎")

# 查询
results = system.search("Python")
print(f"包含'Python'的文档: {results}")
# 输出: 包含'Python'的文档: [1, 3]
```

### 示例 2: 使用停用词

```python
from src.index_system import IndexSystem

# 使用停用词文件初始化
system = IndexSystem(stopwords_path="config/stopwords.txt")

# 添加文档（停用词会被自动过滤）
system.add_document("这是一个关于Python的文档")
# "这"、"是"、"一个"、"的" 等停用词会被过滤
```

### 示例 3: 布尔查询

```python
from src.index_system import IndexSystem

system = IndexSystem()

# 添加文档
system.add_document("Python用于机器学习")
system.add_document("Java用于企业开发")
system.add_document("Python和Java都是编程语言")

# AND 查询 - 同时包含两个词
results = system.boolean_search("Python AND 机器学习")
print(results)  # [1]

# OR 查询 - 包含任一词
results = system.boolean_search("Python OR Java")
print(results)  # [1, 2, 3]

# NOT 查询 - 排除特定词
results = system.boolean_search("编程语言 AND NOT Java")
print(results)  # 包含"编程语言"但不包含"Java"的文档
```

### 示例 4: 保存和加载索引

```python
from src.index_system import IndexSystem

# 构建索引
system = IndexSystem()
system.add_document("文档内容1")
system.add_document("文档内容2")

# 保存索引
system.save_index("data/my_index")

# 稍后加载索引
new_system = IndexSystem()
new_system.load_index("data/my_index")

# 继续使用
results = new_system.search("文档")
```

### 示例 5: 查看统计信息

```python
from src.index_system import IndexSystem

system = IndexSystem()
system.add_document("Python是一种编程语言")
system.add_document("Java也是一种编程语言")

# 获取统计信息
stats = system.get_statistics()

print(f"文档总数: {stats['document_count']}")
print(f"唯一词项数: {stats['vocabulary_size']}")
print(f"平均文档长度: {stats['avg_doc_length']:.2f}")

# 查看词频
for term, freq in stats['term_frequencies'].items():
    print(f"'{term}': {freq} 个文档")
```

## 常见问题

### Q: 如何处理中文文档？

A: 系统默认支持中文，使用 jieba 进行分词。只需直接添加中文文档即可。

```python
system.add_document("这是一个中文文档")
```

### Q: 如何自定义停用词？

A: 编辑 `config/stopwords.txt` 文件，每行一个停用词。

### Q: 布尔查询的运算符优先级是什么？

A: 优先级为 `NOT > AND > OR`。例如：
- `A AND B OR C` 解析为 `(A AND B) OR C`
- `A OR B AND C` 解析为 `A OR (B AND C)`
- `NOT A AND B` 解析为 `(NOT A) AND B`

### Q: 如何查看某个文档的详细信息？

```python
doc = system.get_document(doc_id)
print(f"内容: {doc['content']}")
print(f"词项数: {doc['token_count']}")
print(f"元数据: {doc['metadata']}")
```

### Q: 索引文件保存在哪里？

A: 默认保存在 `data/` 目录下。保存时会生成两个文件：
- `.index` 文件：倒排索引数据
- `.docs` 文件：文档存储数据

## 下一步

- 查看 [README.md](README.md) 了解完整功能
- 查看 `.kiro/specs/inverted-index/` 了解系统设计
- 运行 `pytest tests/ -v` 查看测试用例
- 尝试修改和扩展系统功能

## 获取帮助

如果遇到问题：
1. 查看错误日志（系统会输出详细的日志信息）
2. 运行测试确保系统正常：`pytest tests/ -v`
3. 查看设计文档了解系统架构
