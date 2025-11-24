# 倒排索引系统

一个简单但功能完整的中文文档倒排索引和检索系统。

## 功能特性

- ✅ **文档管理**: 添加、存储和检索文档
- ✅ **中文分词**: 基于 jieba 的中文分词处理
- ✅ **文本预处理**: 停用词过滤、大小写规范化、标点处理
- ✅ **倒排索引**: 高效的词项到文档映射，支持位置信息记录
- ✅ **单词项查询**: 快速查找包含特定词项的文档
- ✅ **布尔查询**: 支持 AND、OR、NOT 运算符的复杂查询
- ✅ **索引持久化**: 保存和加载索引到磁盘
- ✅ **统计信息**: 查看索引规模、词频等统计数据

## 项目结构

```
lesson2/
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── models.py                 # 核心数据模型
│   ├── document_store.py         # 文档存储组件
│   ├── text_preprocessor.py      # 文本预处理组件
│   ├── inverted_index.py         # 倒排索引核心组件
│   ├── query_processor.py        # 查询处理组件
│   └── index_system.py           # 系统主入口
├── tests/                        # 测试目录
│   ├── __init__.py
│   ├── test_models.py            # 数据模型测试
│   ├── test_document_store.py    # 文档存储测试
│   ├── test_text_preprocessor.py # 文本预处理测试
│   ├── test_inverted_index.py    # 倒排索引测试
│   ├── test_query_processor.py   # 查询处理测试
│   ├── test_index_system.py      # 系统集成测试
│   └── test_statistics.py        # 统计功能测试
├── data/                         # 数据目录（索引文件）
├── config/                       # 配置目录
│   └── stopwords.txt             # 停用词列表
├── demo.py                       # 交互式演示程序
├── demo_index_system.py          # 系统功能演示
├── demo_boolean_query.py         # 布尔查询演示
├── requirements.txt              # 项目依赖
├── pytest.ini                    # pytest 配置
└── README.md                     # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行交互式演示

```bash
python demo.py
```

这将启动一个交互式命令行界面，您可以：
- 初始化索引系统
- 添加文档（单个或批量）
- 执行单词项查询
- 执行布尔查询
- 查看文档内容
- 查看索引统计信息
- 保存和加载索引

### 3. 运行功能演示

**系统功能演示**：
```bash
python demo_index_system.py
```

展示完整的索引构建、查询、统计和持久化流程。

**布尔查询演示**：
```bash
python demo_boolean_query.py
```

展示 AND、OR、NOT 运算符的使用和复杂布尔查询。

## 使用示例

### 基本使用

```python
from src.index_system import IndexSystem

# 1. 初始化系统
system = IndexSystem(stopwords_path="config/stopwords.txt")

# 2. 添加文档
doc_id1 = system.add_document(
    "Python是一种流行的编程语言",
    metadata={"title": "Python简介"}
)
doc_id2 = system.add_document(
    "Java也是一种编程语言",
    metadata={"title": "Java简介"}
)

# 3. 单词项查询
results = system.search("编程语言")
print(f"找到文档: {results}")  # [1, 2]

# 4. 布尔查询
results = system.boolean_search("Python AND 编程语言")
print(f"找到文档: {results}")  # [1]

# 5. 获取统计信息
stats = system.get_statistics()
print(f"文档总数: {stats['document_count']}")
print(f"词汇表大小: {stats['vocabulary_size']}")

# 6. 保存索引
system.save_index("data/my_index")

# 7. 加载索引
new_system = IndexSystem()
new_system.load_index("data/my_index")
```

### 布尔查询语法

- **AND 查询**: `Python AND 机器学习` - 返回同时包含两个词的文档
- **OR 查询**: `Python OR Java` - 返回包含任一词的文档
- **NOT 查询**: `Python AND NOT 数据` - 返回包含 Python 但不包含数据的文档
- **复杂查询**: `Python AND 机器学习 OR Java` - 支持运算符组合（优先级: NOT > AND > OR）

## 核心组件

### 1. DocumentStore（文档存储）
负责文档的添加、存储和检索，自动分配唯一文档ID。

### 2. TextPreprocessor（文本预处理）
- 中文分词（使用 jieba）
- 停用词过滤
- 大小写规范化
- 标点符号处理

### 3. InvertedIndex（倒排索引）
- 构建词项到文档的映射
- 记录词项位置信息
- 支持索引持久化
- 提供统计信息查询

### 4. QueryProcessor（查询处理）
- 单词项查询
- 布尔查询（AND/OR/NOT）
- 查询预处理
- 结果排序

### 5. IndexSystem（系统主入口）
集成所有组件，提供统一的对外接口。

## 运行测试

### 运行所有测试
```bash
pytest tests/ -v
```

### 运行特定测试
```bash
# 测试文档存储
pytest tests/test_document_store.py -v

# 测试倒排索引
pytest tests/test_inverted_index.py -v

# 测试查询处理
pytest tests/test_query_processor.py -v
```

### 查看测试覆盖率
```bash
pytest tests/ --cov=src --cov-report=html
```

## 性能特点

- **索引构建**: O(n*m)，n为文档数，m为平均文档长度
- **单词项查询**: O(1) 词项查找 + O(k) 结果返回，k为结果数量
- **布尔查询**: O(k1 + k2 + ...) 集合运算，ki为各词项的结果数量
- **内存占用**: 取决于文档数量和词汇表大小

## 扩展性

系统设计支持以下扩展：

1. **短语查询**: 利用位置信息实现短语匹配
2. **相关性排序**: 添加 TF-IDF 或 BM25 评分
3. **字段索引**: 支持对文档不同字段建立索引
4. **实时更新**: 支持文档的删除和更新操作
5. **分布式索引**: 将索引分片到多个节点

## 技术栈

- **Python 3.8+**: 核心语言
- **jieba**: 中文分词
- **pytest**: 测试框架
- **hypothesis**: 基于属性的测试（可选）

## 开发文档

详细的需求、设计和实现计划请参见：
- 需求文档: `.kiro/specs/inverted-index/requirements.md`
- 设计文档: `.kiro/specs/inverted-index/design.md`
- 任务列表: `.kiro/specs/inverted-index/tasks.md`

## 许可证

本项目仅用于学习和研究目的。
