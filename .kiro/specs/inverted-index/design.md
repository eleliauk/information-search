# 设计文档

## 概述

本系统实现一个简单但功能完整的倒排索引程序，用于支持中文文档的快速检索。系统采用模块化设计，将文档管理、文本预处理、索引构建和查询处理分离为独立组件，便于维护和扩展。

核心功能包括：
- 文档的添加和管理
- 中文文本的分词和预处理
- 倒排索引的构建和维护
- 单词项查询和布尔查询
- 索引的持久化存储
- 索引统计信息的查询

## 架构

系统采用分层架构设计：

```
┌─────────────────────────────────────┐
│      查询接口层 (Query Interface)    │
│  - 单词项查询                         │
│  - 布尔查询 (AND/OR/NOT)             │
│  - 统计信息查询                       │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│    索引管理层 (Index Manager)        │
│  - 倒排索引构建                       │
│  - 索引更新和维护                     │
│  - 索引持久化                         │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│   文本处理层 (Text Processor)        │
│  - 中文分词                           │
│  - 停用词过滤                         │
│  - 文本规范化                         │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│    文档存储层 (Document Store)       │
│  - 文档添加和存储                     │
│  - 文档检索                           │
│  - 文档元数据管理                     │
└─────────────────────────────────────┘
```

## 组件和接口

### 1. 文档存储 (DocumentStore)

负责文档的存储和管理。

**接口：**
```python
class DocumentStore:
    def add_document(self, content: str, metadata: dict = None) -> int
    def get_document(self, doc_id: int) -> Optional[Document]
    def get_all_documents(self) -> List[Document]
    def document_count(self) -> int
```

**职责：**
- 为每个文档分配唯一标识符
- 存储文档内容和元数据
- 提供文档检索功能
- 验证文档有效性

### 2. 文本预处理器 (TextPreprocessor)

负责文本的分词和预处理。

**接口：**
```python
class TextPreprocessor:
    def __init__(self, stopwords_path: str = None)
    def tokenize(self, text: str) -> List[str]
    def preprocess(self, text: str) -> str
    def load_stopwords(self, path: str) -> None
```

**职责：**
- 中文分词（使用 jieba）
- 停用词过滤
- 文本规范化（小写转换、标点处理）
- 提供一致的预处理流程

### 3. 倒排索引 (InvertedIndex)

核心组件，负责索引的构建和维护。

**接口：**
```python
class InvertedIndex:
    def build_index(self, doc_id: int, tokens: List[str]) -> None
    def get_posting_list(self, term: str) -> List[Posting]
    def get_term_frequency(self, term: str) -> int
    def get_vocabulary_size(self) -> int
    def save(self, filepath: str) -> None
    def load(self, filepath: str) -> None
```

**数据结构：**
```python
# 倒排列表项
class Posting:
    doc_id: int           # 文档ID
    positions: List[int]  # 词项在文档中的位置列表
    term_freq: int        # 词项在文档中的频率

# 索引结构
index: Dict[str, List[Posting]]  # 词项 -> 倒排列表
```

**职责：**
- 构建和更新倒排索引
- 维护词项到文档的映射
- 记录词项位置信息
- 提供索引查询接口

### 4. 查询处理器 (QueryProcessor)

负责处理用户查询并返回结果。

**接口：**
```python
class QueryProcessor:
    def __init__(self, inverted_index: InvertedIndex, 
                 text_preprocessor: TextPreprocessor,
                 document_store: DocumentStore)
    
    def search(self, query: str) -> List[int]
    def boolean_search(self, query: BooleanQuery) -> List[int]
    def and_query(self, terms: List[str]) -> List[int]
    def or_query(self, terms: List[str]) -> List[int]
    def not_query(self, include_terms: List[str], exclude_terms: List[str]) -> List[int]
```

**职责：**
- 处理单词项查询
- 处理布尔查询（AND/OR/NOT）
- 对查询词进行预处理
- 合并和排序查询结果

### 5. 索引系统 (IndexSystem)

系统的主入口，协调各个组件。

**接口：**
```python
class IndexSystem:
    def __init__(self, stopwords_path: str = None)
    def add_document(self, content: str, metadata: dict = None) -> int
    def search(self, query: str) -> List[int]
    def boolean_search(self, query: str) -> List[int]
    def get_statistics(self) -> dict
    def save_index(self, filepath: str) -> bool
    def load_index(self, filepath: str) -> bool
```

## 数据模型

### Document（文档）
```python
@dataclass
class Document:
    doc_id: int              # 文档唯一标识符
    content: str             # 文档内容
    metadata: dict           # 元数据（标题、分类等）
    token_count: int         # 词项数量
    created_at: datetime     # 创建时间
```

### Posting（倒排列表项）
```python
@dataclass
class Posting:
    doc_id: int              # 文档ID
    positions: List[int]     # 词项位置列表
    term_freq: int           # 词频
```

### IndexStatistics（索引统计）
```python
@dataclass
class IndexStatistics:
    document_count: int      # 文档总数
    vocabulary_size: int     # 唯一词项数
    total_tokens: int        # 总词项数
    avg_doc_length: float    # 平均文档长度
    term_frequencies: dict   # 词项文档频率
```

### BooleanQuery（布尔查询）
```python
@dataclass
class BooleanQuery:
    operator: str            # 'AND', 'OR', 'NOT'
    terms: List[str]         # 查询词列表
    exclude_terms: List[str] # NOT 运算的排除词
```

## 正确性属性

*属性是指在系统的所有有效执行中都应该成立的特征或行为——本质上是关于系统应该做什么的形式化陈述。属性是人类可读规范和机器可验证正确性保证之间的桥梁。*


### 文档管理属性

**属性 1：文档ID唯一性**
*对于任意*添加到系统的文档集合，每个文档应当被分配唯一的文档标识符，不存在两个文档具有相同的ID
**验证需求：1.1**

**属性 2：文档内容完整性**
*对于任意*文档和元数据，添加到系统后，通过文档ID查询应当返回完全相同的内容和元数据
**验证需求：1.2, 1.3**

### 文本预处理属性

**属性 3：停用词过滤**
*对于任意*包含停用词的文档，分词后的索引中不应包含任何停用词
**验证需求：2.2**

**属性 4：大小写规范化**
*对于任意*包含字母的文本，索引中的所有词项应当只包含小写字母
**验证需求：2.3**

**属性 5：标点符号处理**
*对于任意*包含标点符号的文档，索引中的词项不应包含非必要的标点符号
**验证需求：2.4**

### 索引构建属性

**属性 6：词项完整性**
*对于任意*文档，添加后该文档中的每个有效词项都应当能在索引中找到对应的倒排列表
**验证需求：3.1, 3.4**

**属性 7：位置信息准确性**
*对于任意*文档中的词项，索引应当准确记录该词项在文档中的所有出现位置
**验证需求：3.3**

**属性 8：索引往返一致性**
*对于任意*文档，添加文档后查询该文档中的任意词项，返回的结果应当包含该文档的ID
**验证需求：3.5**

### 查询处理属性

**属性 9：查询结果完整性**
*对于任意*存在于索引中的词项，查询应当返回所有包含该词项的文档ID
**验证需求：4.1**

**属性 10：查询预处理一致性**
*对于任意*查询词的不同形式（大小写、标点等），经过预处理后应当返回相同的查询结果
**验证需求：4.3**

**属性 11：结果排序性**
*对于任意*查询，返回的文档ID列表应当按照升序排列
**验证需求：4.4**

### 布尔查询属性

**属性 12：AND 运算正确性**
*对于任意*词项集合的 AND 查询，返回的每个文档都应当包含所有查询词项
**验证需求：5.1**

**属性 13：OR 运算正确性**
*对于任意*词项集合的 OR 查询，返回的每个文档都应当至少包含一个查询词项
**验证需求：5.2**

**属性 14：NOT 运算正确性**
*对于任意*NOT 查询，返回的文档都不应当包含被排除的词项
**验证需求：5.3**

**属性 15：布尔运算优先级**
*对于任意*包含多个运算符的复杂布尔查询，结果应当符合标准的布尔运算优先级（NOT > AND > OR）
**验证需求：5.4**

### 持久化属性

**属性 16：序列化往返一致性**
*对于任意*索引状态，保存到磁盘后再加载，应当恢复到完全相同的索引状态（包括所有文档、词项和倒排列表）
**验证需求：6.3**

**属性 17：文件操作健壮性**
*对于任意*不存在的文件路径，加载操作应当返回明确的错误信息而不是抛出未捕获的异常
**验证需求：6.4**

### 统计信息属性

**属性 18：文档计数准确性**
*对于任意*索引状态，统计信息中的文档总数应当等于实际添加的文档数量
**验证需求：7.1**

**属性 19：词汇表大小准确性**
*对于任意*索引状态，统计信息中的唯一词项数应当等于索引中实际的不同词项数量
**验证需求：7.2**

**属性 20：词频统计准确性**
*对于任意*词项，其文档频率应当等于包含该词项的文档数量
**验证需求：7.3**

**属性 21：平均文档长度准确性**
*对于任意*索引状态，平均文档长度应当等于所有文档词项总数除以文档数量
**验证需求：7.4**

## 错误处理

### 输入验证错误
- **空文档错误**：拒绝空字符串或纯空白字符串的文档
- **无效文档ID错误**：查询不存在的文档ID时返回 None
- **无效查询错误**：处理空查询或格式错误的布尔查询

### 文件操作错误
- **文件不存在错误**：加载不存在的索引文件时返回错误信息
- **文件读写错误**：处理文件权限或磁盘空间问题
- **序列化错误**：处理数据格式不兼容的情况

### 系统错误
- **内存不足错误**：处理大规模索引时的内存限制
- **分词错误**：处理分词器异常情况
- **并发错误**：如果支持并发访问，需要处理竞态条件

**错误处理原则：**
1. 所有错误都应该有明确的错误消息
2. 系统错误不应导致数据损坏
3. 错误后系统应保持一致状态
4. 提供足够的错误信息用于调试

## 测试策略

### 单元测试

单元测试用于验证各个组件的基本功能和特定边界情况：

**DocumentStore 测试：**
- 测试添加单个文档
- 测试添加空文档被拒绝
- 测试文档ID的唯一性
- 测试获取不存在的文档

**TextPreprocessor 测试：**
- 测试中文分词功能
- 测试停用词过滤
- 测试大小写转换
- 测试标点符号处理
- 测试空文本处理

**InvertedIndex 测试：**
- 测试索引构建
- 测试获取倒排列表
- 测试词项不存在的情况
- 测试索引统计信息

**QueryProcessor 测试：**
- 测试单词项查询
- 测试 AND/OR/NOT 查询
- 测试空查询结果
- 测试查询结果排序

**持久化测试：**
- 测试保存和加载索引
- 测试加载不存在的文件
- 测试文件损坏情况

### 基于属性的测试

基于属性的测试用于验证系统在各种输入下的通用正确性属性。我们将使用 **Hypothesis** 作为 Python 的属性测试库。

**测试配置：**
- 每个属性测试至少运行 100 次迭代
- 使用随机生成的文档、查询和索引状态
- 每个属性测试必须用注释标注对应的设计文档属性

**标注格式：**
```python
# Feature: inverted-index, Property 8: 索引往返一致性
@given(documents=st.lists(st.text(min_size=10)))
def test_index_roundtrip_consistency(documents):
    ...
```

**测试生成器设计：**

1. **文档生成器**：生成包含中文、英文、数字、标点的随机文档
2. **查询生成器**：从已索引的词项中随机选择查询词
3. **布尔查询生成器**：生成随机的 AND/OR/NOT 组合查询
4. **索引状态生成器**：生成包含不同数量文档的索引状态

**属性测试覆盖：**

- **属性 1-2**：文档管理的唯一性和完整性
- **属性 3-5**：文本预处理的一致性
- **属性 6-8**：索引构建的正确性
- **属性 9-11**：查询处理的完整性和一致性
- **属性 12-15**：布尔查询的逻辑正确性
- **属性 16-17**：持久化的往返一致性
- **属性 18-21**：统计信息的准确性

**边界情况测试：**
- 空文档处理（属性 1.4）
- 不存在的词项查询（属性 4.2）
- 不存在的文件加载（属性 6.4）

### 集成测试

集成测试验证各组件协同工作的正确性：

1. **端到端索引流程**：添加文档 → 构建索引 → 查询 → 验证结果
2. **持久化流程**：构建索引 → 保存 → 加载 → 验证一致性
3. **复杂查询流程**：多文档索引 → 复杂布尔查询 → 验证结果正确性

### 性能测试

虽然不是核心功能，但应该验证基本性能：

1. **索引构建性能**：测试大量文档的索引构建时间
2. **查询性能**：测试查询响应时间
3. **内存使用**：监控索引的内存占用

## 实现技术选择

### 编程语言
- **Python 3.8+**：简洁易读，丰富的文本处理库

### 核心依赖
- **jieba**：中文分词库，成熟稳定
- **pickle/json**：索引序列化，pickle 用于内部格式，json 用于可读格式

### 测试依赖
- **pytest**：测试框架
- **hypothesis**：基于属性的测试库
- **pytest-cov**：代码覆盖率

### 数据结构选择
- **Dict[str, List[Posting]]**：倒排索引的主结构，提供 O(1) 查找
- **List[int]**：文档ID列表，保持有序便于集合运算
- **Set[int]**：布尔查询的中间结果，提供高效的集合运算

## 性能考虑

### 索引构建优化
- 批量添加文档时避免重复构建
- 使用增量索引更新而非重建

### 查询优化
- 布尔查询时先处理最小的倒排列表
- 结果集合运算使用有序列表归并

### 内存优化
- 大规模索引考虑使用压缩存储
- 位置信息可选存储（如果不需要短语查询）

### 持久化优化
- 使用二进制格式（pickle）提高序列化速度
- 提供增量保存选项

## 扩展性考虑

系统设计支持以下扩展：

1. **短语查询**：利用位置信息实现短语匹配
2. **相关性排序**：添加 TF-IDF 或 BM25 评分
3. **字段索引**：支持对文档的不同字段建立索引
4. **实时更新**：支持文档的删除和更新操作
5. **分布式索引**：将索引分片到多个节点

## 实现顺序建议

1. 实现 Document 和 DocumentStore（基础数据层）
2. 实现 TextPreprocessor（文本处理）
3. 实现 InvertedIndex（核心索引）
4. 实现 QueryProcessor（查询功能）
5. 实现 IndexSystem（系统集成）
6. 实现持久化功能
7. 实现统计信息功能
8. 编写单元测试和属性测试
9. 性能优化和文档完善
