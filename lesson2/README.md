# 倒排索引系统

一个简单但功能完整的中文文档倒排索引和检索系统。

## 项目结构

```
lesson2/
├── src/                    # 源代码目录
│   ├── __init__.py
│   └── models.py          # 核心数据模型
├── tests/                  # 测试目录
│   ├── __init__.py
│   └── test_models.py     # 数据模型测试
├── data/                   # 数据目录
├── config/                 # 配置目录
├── requirements.txt        # 项目依赖
├── pytest.ini             # pytest 配置
└── README.md              # 项目说明
```

## 核心数据模型

- **Document**: 文档数据模型，包含文档ID、内容、元数据等
- **Posting**: 倒排列表项，记录词项在文档中的位置和频率
- **IndexStatistics**: 索引统计信息
- **BooleanQuery**: 布尔查询模型

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行测试

```bash
pytest tests/ -v
```

## 开发计划

详见 `.kiro/specs/inverted-index/` 目录下的需求、设计和任务文档。
