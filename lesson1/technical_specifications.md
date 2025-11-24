# 中文新闻稀疏检索系统技术规格

## 1. 使用的数据 (Data Specification)

### 1.1 数据来源与获取策略

**目标数据源**:
- 新浪新闻 (https://news.sina.com.cn/)
- 网易新闻 (https://news.163.com/)
- 腾讯新闻 (https://news.qq.com/)

**数据规模**: 100-1000篇中文新闻文章

**数据字段结构**:
```json
{
    "article_id": "唯一标识符",
    "title": "新闻标题",
    "content": "新闻正文内容",
    "category": "新闻分类 (政治/经济/体育/娱乐/科技等)",
    "publish_time": "发布时间 ISO格式",
    "source_url": "原文链接",
    "author": "作者信息",
    "tags": ["标签1", "标签2"],
    "word_count": "字符总数",
    "crawl_time": "爬取时间"
}
```

**数据质量标准**:
- 正文长度: 100-5000字符
- 标题长度: 10-100字符
- 内容完整性: 无截断、无乱码
- 时效性: 最近30天内的新闻
- 去重处理: 基于标题和内容相似度去重

### 1.2 数据爬取实现

```python
class NewsDataCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def collect_sina_news(self, max_articles=500):
        """爬取新浪新闻数据"""
        news_data = []
        categories = ['domestic', 'international', 'finance', 'sports', 'tech']
        
        for category in categories:
            articles = self.scrape_category(category, max_articles//len(categories))
            news_data.extend(articles)
        
        return news_data
    
    def scrape_category(self, category, limit):
        """爬取特定分类的新闻"""
        articles = []
        page = 1
        
        while len(articles) < limit:
            url = f"https://news.sina.com.cn/{category}/page_{page}.html"
            
            try:
                response = self.session.get(url, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 解析新闻列表
                news_links = soup.find_all('a', class_='news-link')
                
                for link in news_links:
                    if len(articles) >= limit:
                        break
                    
                    article_data = self.extract_article_content(link['href'])
                    if self.validate_article(article_data):
                        articles.append(article_data)
                
                page += 1
                time.sleep(1)  # 防止请求过快
                
            except Exception as e:
                print(f"爬取出错: {e}")
                break
        
        return articles
    
    def extract_article_content(self, url):
        """提取单篇文章内容"""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 提取标题
            title = soup.find('h1', class_='main-title').get_text().strip()
            
            # 提取正文
            content_div = soup.find('div', class_='article-content')
            content = content_div.get_text().strip()
            
            # 提取发布时间
            time_elem = soup.find('span', class_='time')
            publish_time = time_elem.get_text().strip() if time_elem else None
            
            return {
                'article_id': self.generate_id(url),
                'title': title,
                'content': content,
                'source_url': url,
                'publish_time': publish_time,
                'word_count': len(content),
                'crawl_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"提取文章内容失败: {e}")
            return None
    
    def validate_article(self, article):
        """验证文章数据质量"""
        if not article:
            return False
        
        # 检查必要字段
        required_fields = ['title', 'content']
        if not all(article.get(field) for field in required_fields):
            return False
        
        # 检查内容长度
        if len(article['content']) < 100 or len(article['content']) > 5000:
            return False
        
        # 检查标题长度
        if len(article['title']) < 10 or len(article['title']) > 100:
            return False
        
        return True
```

### 1.3 数据存储方案

```python
class DataStorage:
    def __init__(self, storage_path="data/"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def save_raw_data(self, news_data):
        """保存原始数据"""
        filename = f"raw_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.storage_path, "raw", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(news_data, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def load_raw_data(self, filepath):
        """加载原始数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_processed_data(self, processed_data, filename):
        """保存处理后的数据"""
        filepath = os.path.join(self.storage_path, "processed", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        return filepath
```

## 2. 数据统计分析 (Data Statistical Analysis)

### 2.1 原始数据统计指标

```python
class RawDataAnalyzer:
    def __init__(self, news_data):
        self.news_data = news_data
        self.analysis_results = {}
    
    def comprehensive_analysis(self):
        """综合数据分析"""
        self.analysis_results = {
            "基本统计": self.basic_statistics(),
            "文本长度分析": self.text_length_analysis(),
            "分类分布": self.category_distribution(),
            "时间分布": self.temporal_distribution(),
            "内容质量分析": self.content_quality_analysis()
        }
        
        return self.analysis_results
    
    def basic_statistics(self):
        """基本统计信息"""
        total_articles = len(self.news_data)
        total_words = sum(article['word_count'] for article in self.news_data)
        
        return {
            "文章总数": total_articles,
            "总字符数": total_words,
            "平均字符数": total_words / total_articles if total_articles > 0 else 0,
            "最长文章字符数": max(article['word_count'] for article in self.news_data),
            "最短文章字符数": min(article['word_count'] for article in self.news_data),
            "字符数标准差": np.std([article['word_count'] for article in self.news_data])
        }
    
    def text_length_analysis(self):
        """文本长度分析"""
        word_counts = [article['word_count'] for article in self.news_data]
        title_lengths = [len(article['title']) for article in self.news_data]
        
        # 长度分布区间
        length_bins = [0, 200, 500, 1000, 2000, 5000]
        length_distribution = {}
        
        for i in range(len(length_bins) - 1):
            count = sum(1 for wc in word_counts 
                       if length_bins[i] <= wc < length_bins[i+1])
            length_distribution[f"{length_bins[i]}-{length_bins[i+1]}"] = count
        
        return {
            "正文长度分布": length_distribution,
            "标题长度统计": {
                "平均长度": np.mean(title_lengths),
                "最长标题": max(title_lengths),
                "最短标题": min(title_lengths),
                "标准差": np.std(title_lengths)
            },
            "正文长度百分位数": {
                "25%": np.percentile(word_counts, 25),
                "50%": np.percentile(word_counts, 50),
                "75%": np.percentile(word_counts, 75),
                "90%": np.percentile(word_counts, 90)
            }
        }
    
    def category_distribution(self):
        """分类分布分析"""
        categories = [article.get('category', '未分类') for article in self.news_data]
        category_counts = Counter(categories)
        
        total = len(categories)
        category_stats = {}
        
        for category, count in category_counts.items():
            category_stats[category] = {
                "文章数量": count,
                "占比": f"{(count/total)*100:.2f}%",
                "平均字符数": np.mean([
                    article['word_count'] for article in self.news_data 
                    if article.get('category') == category
                ])
            }
        
        return category_stats
    
    def temporal_distribution(self):
        """时间分布分析"""
        # 解析发布时间
        publish_times = []
        for article in self.news_data:
            if article.get('publish_time'):
                try:
                    pub_time = datetime.fromisoformat(article['publish_time'])
                    publish_times.append(pub_time)
                except:
                    continue
        
        if not publish_times:
            return {"错误": "无有效时间数据"}
        
        # 按日期分组
        date_counts = Counter(pt.date() for pt in publish_times)
        
        # 按小时分组
        hour_counts = Counter(pt.hour for pt in publish_times)
        
        return {
            "时间跨度": {
                "最早": min(publish_times).isoformat(),
                "最晚": max(publish_times).isoformat(),
                "跨度天数": (max(publish_times) - min(publish_times)).days
            },
            "每日发布量": dict(sorted(date_counts.items())),
            "每小时发布量": dict(sorted(hour_counts.items())),
            "发布高峰时段": max(hour_counts.items(), key=lambda x: x[1])
        }
    
    def content_quality_analysis(self):
        """内容质量分析"""
        # 检查重复内容
        title_similarities = self.calculate_title_similarities()
        
        # 检查内容完整性
        incomplete_articles = [
            article for article in self.news_data
            if not article.get('content') or len(article['content']) < 50
        ]
        
        # 检查特殊字符
        special_char_articles = [
            article for article in self.news_data
            if any(char in article['content'] for char in ['□', '■', '???'])
        ]
        
        return {
            "数据完整性": {
                "完整文章数": len(self.news_data) - len(incomplete_articles),
                "不完整文章数": len(incomplete_articles),
                "完整率": f"{((len(self.news_data) - len(incomplete_articles)) / len(self.news_data)) * 100:.2f}%"
            },
            "内容质量": {
                "疑似重复标题数": len([s for s in title_similarities if s > 0.8]),
                "包含特殊字符文章数": len(special_char_articles),
                "质量评分": self.calculate_quality_score()
            }
        }
    
    def calculate_title_similarities(self):
        """计算标题相似度"""
        from difflib import SequenceMatcher
        
        titles = [article['title'] for article in self.news_data]
        similarities = []
        
        for i in range(len(titles)):
            for j in range(i+1, len(titles)):
                similarity = SequenceMatcher(None, titles[i], titles[j]).ratio()
                similarities.append(similarity)
        
        return similarities
    
    def calculate_quality_score(self):
        """计算数据质量评分"""
        score = 100
        
        # 根据各种质量指标扣分
        incomplete_rate = len([a for a in self.news_data if len(a['content']) < 100]) / len(self.news_data)
        score -= incomplete_rate * 30
        
        # 长度异常扣分
        length_variance = np.var([a['word_count'] for a in self.news_data])
        if length_variance > 1000000:  # 长度差异过大
            score -= 10
        
        return max(0, score)
```

### 2.2 数据可视化

```python
class DataVisualizer:
    def __init__(self, analysis_results):
        self.analysis_results = analysis_results
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
        plt.rcParams['axes.unicode_minus'] = False
    
    def create_all_visualizations(self):
        """创建所有可视化图表"""
        self.plot_length_distribution()
        self.plot_category_distribution()
        self.plot_temporal_distribution()
        self.plot_quality_metrics()
    
    def plot_length_distribution(self):
        """绘制文章长度分布图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 正文长度分布
        length_dist = self.analysis_results["文本长度分析"]["正文长度分布"]
        ax1.bar(length_dist.keys(), length_dist.values())
        ax1.set_title('文章正文长度分布')
        ax1.set_xlabel('字符数区间')
        ax1.set_ylabel('文章数量')
        ax1.tick_params(axis='x', rotation=45)
        
        # 长度百分位数
        percentiles = self.analysis_results["文本长度分析"]["正文长度百分位数"]
        ax2.boxplot([percentiles["25%"], percentiles["50%"], percentiles["75%"], percentiles["90%"]])
        ax2.set_title('文章长度百分位数')
        ax2.set_ylabel('字符数')
        
        plt.tight_layout()
        plt.savefig('data/analysis/length_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_category_distribution(self):
        """绘制分类分布图"""
        category_dist = self.analysis_results["分类分布"]
        
        categories = list(category_dist.keys())
        counts = [category_dist[cat]["文章数量"] for cat in categories]
        
        plt.figure(figsize=(10, 8))
        plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
        plt.title('新闻分类分布')
        plt.axis('equal')
        plt.savefig('data/analysis/category_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_temporal_distribution(self):
        """绘制时间分布图"""
        temporal_data = self.analysis_results["时间分布"]
        
        if "每小时发布量" in temporal_data:
            hours = list(temporal_data["每小时发布量"].keys())
            counts = list(temporal_data["每小时发布量"].values())
            
            plt.figure(figsize=(12, 6))
            plt.bar(hours, counts)
            plt.title('每小时新闻发布量分布')
            plt.xlabel('小时')
            plt.ylabel('文章数量')
            plt.xticks(range(0, 24, 2))
            plt.grid(axis='y', alpha=0.3)
            plt.savefig('data/analysis/temporal_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
```

## 3. 分词统计分析 (Tokenization Statistical Analysis)

### 3.1 结巴分词配置与优化

```python
class OptimizedChineseTokenizer:
    def __init__(self):
        # 设置结巴分词模式
        jieba.enable_parallel(4)  # 并行分词
        
        # 加载自定义词典
        self.load_custom_dictionaries()
        
        # 加载停用词
        self.stopwords = self.load_stopwords()
        
        # 词性过滤规则
        self.keep_pos = {'n', 'nr', 'ns', 'nt', 'nw', 'nz',  # 名词类
                        'v', 'vd', 'vn',                      # 动词类
                        'a', 'ad', 'an',                      # 形容词类
                        'i', 'j', 'l'}                       # 成语、简称、习用语
        
        # 统计信息
        self.tokenization_stats = {
            "total_documents": 0,
            "total_tokens": 0,
            "unique_tokens": set(),
            "token_frequencies": Counter(),
            "pos_distribution": Counter(),
            "token_length_distribution": Counter(),
            "oov_tokens": set()  # 未登录词
        }
    
    def load_custom_dictionaries(self):
        """加载自定义词典"""
        # 新闻领域专用词典
        news_dict = [
            "人工智能 5 n",
            "区块链 5 n", 
            "新冠肺炎 5 n",
            "碳达峰 5 n",
            "碳中和 5 n",
            "元宇宙 5 n",
            "数字经济 5 n"
        ]
        
        # 创建临时词典文件
        with open('config/news_dict.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(news_dict))
        
        jieba.load_userdict('config/news_dict.txt')
    
    def load_stopwords(self):
        """加载停用词表"""
        stopwords = set()
        
        # 基础停用词
        basic_stopwords = [
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", 
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "那", 
            "什么", "时候", "如果", "但是", "因为", "所以", "而且", "然后"
        ]
        
        # 标点符号
        punctuation = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〱〲〳〴〵〶〷〸〹〺〻〼〽〾〿"
        
        stopwords.update(basic_stopwords)
        stopwords.update(punctuation)
        
        return stopwords
    
    def tokenize_document(self, text, keep_pos=True):
        """对单个文档进行分词"""
        # 预处理：清理文本
        text = self.preprocess_text(text)
        
        # 结巴分词（带词性标注）
        if keep_pos:
            words_with_pos = list(pseg.cut(text))
            
            # 过滤词性和停用词
            filtered_tokens = []
            for word, pos in words_with_pos:
                if (len(word) > 1 and 
                    word not in self.stopwords and
                    pos in self.keep_pos and
                    not word.isdigit() and
                    not self.is_pure_punctuation(word)):
                    
                    filtered_tokens.append((word, pos))
                    
                    # 更新统计信息
                    self.update_stats(word, pos)
            
            return filtered_tokens
        else:
            # 只返回词汇，不带词性
            words = jieba.cut(text, cut_all=False)
            filtered_words = [word for word in words 
                            if len(word) > 1 and word not in self.stopwords]
            return filtered_words
    
    def preprocess_text(self, text):
        """文本预处理"""
        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        
        # 移除过多的重复字符
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        return text.strip()
    
    def is_pure_punctuation(self, word):
        """判断是否为纯标点符号"""
        return all(char in string.punctuation + "！？。，；：""''（）【】《》" for char in word)
    
    def update_stats(self, word, pos):
        """更新分词统计信息"""
        self.tokenization_stats["total_tokens"] += 1
        self.tokenization_stats["unique_tokens"].add(word)
        self.tokenization_stats["token_frequencies"][word] += 1
        self.tokenization_stats["pos_distribution"][pos] += 1
        self.tokenization_stats["token_length_distribution"][len(word)] += 1
```

### 3.2 分词效果统计分析

```python
class TokenizationAnalyzer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.analysis_results = {}
    
    def analyze_tokenization_results(self, documents):
        """分析分词结果"""
        print("开始分词分析...")
        
        # 对所有文档进行分词
        tokenized_docs = []
        for i, doc in enumerate(documents):
            if i % 100 == 0:
                print(f"处理进度: {i}/{len(documents)}")
            
            tokens = self.tokenizer.tokenize_document(doc['content'])
            tokenized_docs.append({
                'doc_id': doc.get('article_id', i),
                'title': doc['title'],
                'tokens': tokens,
                'token_count': len(tokens)
            })
        
        # 执行各项统计分析
        self.analysis_results = {
            "基础统计": self.basic_tokenization_stats(tokenized_docs),
            "词频分析": self.frequency_analysis(),
            "词性分析": self.pos_analysis(),
            "词长分析": self.length_analysis(),
            "分词质量评估": self.quality_assessment(documents, tokenized_docs)
        }
        
        return self.analysis_results
    
    def basic_tokenization_stats(self, tokenized_docs):
        """基础分词统计"""
        total_docs = len(tokenized_docs)
        total_tokens = sum(doc['token_count'] for doc in tokenized_docs)
        unique_tokens = len(self.tokenizer.tokenization_stats["unique_tokens"])
        
        token_counts = [doc['token_count'] for doc in tokenized_docs]
        
        return {
            "文档总数": total_docs,
            "词汇总数": total_tokens,
            "唯一词汇数": unique_tokens,
            "词汇丰富度": unique_tokens / total_tokens if total_tokens > 0 else 0,
            "平均每文档词数": total_tokens / total_docs if total_docs > 0 else 0,
            "词数分布": {
                "最大值": max(token_counts),
                "最小值": min(token_counts),
                "中位数": np.median(token_counts),
                "标准差": np.std(token_counts)
            }
        }
    
    def frequency_analysis(self):
        """词频分析"""
        token_freq = self.tokenizer.tokenization_stats["token_frequencies"]
        
        # 高频词分析
        top_50_words = token_freq.most_common(50)
        
        # 低频词分析（只出现1次的词）
        hapax_legomena = [word for word, freq in token_freq.items() if freq == 1]
        
        # 词频分布区间
        freq_distribution = {
            "1次": len([w for w, f in token_freq.items() if f == 1]),
            "2-5次": len([w for w, f in token_freq.items() if 2 <= f <= 5]),
            "6-10次": len([w for w, f in token_freq.items() if 6 <= f <= 10]),
            "11-50次": len([w for w, f in token_freq.items() if 11 <= f <= 50]),
            "50次以上": len([w for w, f in token_freq.items() if f > 50])
        }
        
        return {
            "高频词TOP50": top_50_words,
            "hapax_legomena数量": len(hapax_legomena),
            "hapax_legomena比例": f"{len(hapax_legomena)/len(token_freq)*100:.2f}%",
            "词频分布区间": freq_distribution,
            "词频统计": {
                "最高频词": token_freq.most_common(1)[0] if token_freq else None,
                "平均频次": np.mean(list(token_freq.values())),
                "频次标准差": np.std(list(token_freq.values()))
            }
        }
    
    def pos_analysis(self):
        """词性分析"""
        pos_dist = self.tokenizer.tokenization_stats["pos_distribution"]
        total_pos = sum(pos_dist.values())
        
        # 词性映射（中文说明）
        pos_mapping = {
            'n': '名词', 'nr': '人名', 'ns': '地名', 'nt': '机构名', 'nw': '作品名', 'nz': '其他专名',
            'v': '动词', 'vd': '副动词', 'vn': '名动词',
            'a': '形容词', 'ad': '副形词', 'an': '名形词',
            'i': '成语', 'j': '简称略语', 'l': '习用语'
        }
        
        pos_analysis = {}
        for pos, count in pos_dist.most_common():
            pos_name = pos_mapping.get(pos, pos)
            pos_analysis[pos_name] = {
                "数量": count,
                "占比": f"{count/total_pos*100:.2f}%"
            }
        
        return {
            "词性分布": pos_analysis,
            "主要词性": pos_dist.most_common(5),
            "词性丰富度": len(pos_dist),
            "词性集中度": max(pos_dist.values()) / total_pos if total_pos > 0 else 0
        }
    
    def length_analysis(self):
        """词长分析"""
        length_dist = self.tokenizer.tokenization_stats["token_length_distribution"]
        
        # 计算平均词长
        total_chars = sum(length * count for length, count in length_dist.items())
        total_words = sum(length_dist.values())
        avg_length = total_chars / total_words if total_words > 0 else 0
        
        return {
            "词长分布": dict(sorted(length_dist.items())),
            "平均词长": avg_length,
            "最常见词长": max(length_dist.items(), key=lambda x: x[1])[0],
            "词长统计": {
                "1字词占比": f"{length_dist.get(1, 0)/total_words*100:.2f}%",
                "2字词占比": f"{length_dist.get(2, 0)/total_words*100:.2f}%",
                "3字词占比": f"{length_dist.get(3, 0)/total_words*100:.2f}%",
                "4字及以上占比": f"{sum(count for length, count in length_dist.items() if length >= 4)/total_words*100:.2f}%"
            }
        }
    
    def quality_assessment(self, original_docs, tokenized_docs):
        """分词质量评估"""
        # 分词一致性检查
        consistency_scores = []
        
        for orig, tokenized in zip(original_docs[:10], tokenized_docs[:10]):  # 抽样检查
            # 重新分词同一文档
            tokens1 = [t[0] for t in self.tokenizer.tokenize_document(orig['content'])]
            tokens2 = [t[0] for t in tokenized['tokens']]
            
            # 计算一致性
            if tokens1 and tokens2:
                consistency = len(set(tokens1) & set(tokens2)) / len(set(tokens1) | set(tokens2))
                consistency_scores.append(consistency)
        
        # OOV（Out of Vocabulary）分析
        total_tokens = self.tokenizer.tokenization_stats["total_tokens"]
        oov_count = len(self.tokenizer.tokenization_stats["oov_tokens"])
        
        return {
            "分词一致性": {
                "平均一致性": np.mean(consistency_scores) if consistency_scores else 0,
                "一致性标准差": np.std(consistency_scores) if consistency_scores else 0
            },
            "OOV分析": {
                "OOV词数": oov_count,
                "OOV比例": f"{oov_count/total_tokens*100:.2f}%" if total_tokens > 0 else "0%"
            },
            "分词效率": {
                "平均每秒处理词数": "需要在实际运行时测量",
                "内存使用": "需要在实际运行时测量"
            },
            "质量评分": self.calculate_tokenization_quality_score(consistency_scores, oov_count, total_tokens)
        }
    
    def calculate_tokenization_quality_score(self, consistency_scores, oov_count, total_tokens):
        """计算分词质量评分"""
        score = 100
        
        # 一致性评分
        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            score *= avg_consistency
        
        # OOV惩罚
        if total_tokens > 0:
            oov_rate = oov_count / total_tokens
            score *= (1 - oov_rate)
        
        return min(100, max(0, score))
```

## 4. 检索算法设计 (Retrieval Algorithm Design)

### 4.1 TF-IDF算法实现

```python
class TFIDFRetrieval:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary = {}  # 词汇表 {word: index}
        self.idf_values = {}  # IDF值 {word: idf}
        self.doc_vectors = None  # 文档向量矩阵
        self.documents = []  # 原始文档
        
    def build_vocabulary(self, tokenized_documents):
        """构建词汇表"""
        all_tokens = set()
        
        for doc_tokens in tokenized_documents:
            tokens = [token[0] if isinstance(token, tuple) else token for token in doc_tokens]
            all_tokens.update(tokens)
        
        # 过滤低频词（可选）
        token_counts = Counter()
        for doc_tokens in tokenized_documents:
            tokens = [token[0] if isinstance(token, tuple) else token for token in doc_tokens]
            token_counts.update(tokens)
        
        # 只保留出现次数>=2的词
        filtered_tokens = {token for token, count in token_counts.items() if count >= 2}
        
        # 构建词汇表索引
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(filtered_tokens))}
        
        print(f"词汇表大小: {len(self.vocabulary)}")
        return self.vocabulary
    
    def calculate_tf(self, tokens):
        """计算词频 (Term Frequency)"""
        # 获取纯词汇列表
        word_list = [token[0] if isinstance(token, tuple) else token for token in tokens]
        
        tf_dict = Counter(word_list)
        doc_length = len(word_list)
        
        # 使用对数归一化的TF: 1 + log(tf)
        tf_normalized = {}
        for word, count in tf_dict.items():
            if word in self.vocabulary:
                tf_normalized[word] = 1 + math.log(count) if count > 0 else 0
        
        return tf_normalized
    
    def calculate_idf(self, tokenized_documents):
        """计算逆文档频率 (Inverse Document Frequency)"""
        N = len(tokenized_documents)  # 文档总数
        
        # 计算每个词出现在多少个文档中
        document_frequencies = {}
        
        for doc_tokens in tokenized_documents:
            word_list = [token[0] if isinstance(token, tuple) else token for token in doc_tokens]
            unique_words = set(word_list)
            
            for word in unique_words:
                if word in self.vocabulary:
                    document_frequencies[word] = document_frequencies.get(word, 0) + 1
        
        # 计算IDF: log(N / df)
        for word in self.vocabulary:
            df = document_frequencies.get(word, 0)
            if df > 0:
                self.idf_values[word] = math.log(N / df)
            else:
                self.idf_values[word] = 0
        
        return self.idf_values
    
    def calculate_tfidf_vector(self, tokens):
        """计算单个文档的TF-IDF向量"""
        tf_dict = self.calculate_tf(tokens)
        
        # 创建TF-IDF向量
        tfidf_vector = np.zeros(len(self.vocabulary))
        
        for word, tf_value in tf_dict.items():
            if word in self.vocabulary:
                word_index = self.vocabulary[word]
                idf_value = self.idf_values.get(word, 0)
                tfidf_vector[word_index] = tf_value * idf_value
        
        return tfidf_vector
    
    def build_document_matrix(self, tokenized_documents):
        """构建文档-词汇矩阵"""
        print("构建TF-IDF文档矩阵...")
        
        # 首先构建词汇表
        self.build_vocabulary(tokenized_documents)
        
        # 计算IDF值
        self.calculate_idf(tokenized_documents)
        
        # 计算每个文档的TF-IDF向量
        doc_vectors = []
        
        for i, doc_tokens in enumerate(tokenized_documents):
            if i % 100 == 0:
                print(f"处理文档: {i}/{len(tokenized_documents)}")
            
            tfidf_vector = self.calculate_tfidf_vector(doc_tokens)
            doc_vectors.append(tfidf_vector)
        
        # 转换为稀疏矩阵以节省内存
        self.doc_vectors = csr_matrix(np.array(doc_vectors))
        
        print(f"文档矩阵大小: {self.doc_vectors.shape}")
        return self.doc_vectors
    
    def vectorize_query(self, query_text):
        """将查询转换为TF-IDF向量"""
        # 对查询进行分词
        query_tokens = self.tokenizer.tokenize_document(query_text, keep_pos=False)
        
        # 计算查询的TF-IDF向量
        query_vector = self.calculate_tfidf_vector(query_tokens)
        
        return query_vector.reshape(1, -1)  # 转换为行向量
```

### 4.2 余弦相似度检索

```python
class CosineRetrieval:
    def __init__(self, tfidf_retrieval, documents):
        self.tfidf_retrieval = tfidf_retrieval
        self.documents = documents
        
    def cosine_similarity_manual(self, vec1, vec2):
        """手动实现余弦相似度计算"""
        # 计算点积
        dot_product = np.dot(vec1.flatten(), vec2.flatten())
        
        # 计算向量的L2范数
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # 避免除零错误
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        # 计算余弦相似度
        cosine_sim = dot_product / (norm_vec1 * norm_vec2)
        return cosine_sim
    
    def batch_cosine_similarity(self, query_vector, doc_matrix):
        """批量计算余弦相似度"""
        # 使用sklearn的实现（更高效）
        similarities = cosine_similarity(query_vector, doc_matrix).flatten()
        return similarities
    
    def search(self, query_text, top_k=10, similarity_threshold=0.01):
        """执行检索"""
        print(f"检索查询: '{query_text}'")
        
        # 1. 将查询向量化
        query_vector = self.tfidf_retrieval.vectorize_query(query_text)
        
        # 2. 计算与所有文档的相似度
        similarities = self.batch_cosine_similarity(
            query_vector, 
            self.tfidf_retrieval.doc_vectors
        )
        
        # 3. 过滤低相似度结果
        valid_indices = np.where(similarities > similarity_threshold)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # 4. 排序并获取top-k结果
        valid_similarities = similarities[valid_indices]
        sorted_indices = valid_indices[np.argsort(valid_similarities)[::-1]]
        
        top_indices = sorted_indices[:top_k]
        
        # 5. 构造结果
        results = []
        for idx in top_indices:
            results.append({
                'document_id': idx,
                'similarity_score': similarities[idx],
                'document': self.documents[idx],
                'title': self.documents[idx].get('title', ''),
                'content_preview': self.documents[idx].get('content', '')[:200] + '...'
            })
        
        print(f"找到 {len(results)} 条相关结果")
        return results
    
    def explain_similarity(self, query_text, document_id):
        """解释相似度计算过程"""
        # 获取查询向量
        query_vector = self.tfidf_retrieval.vectorize_query(query_text)
        
        # 获取文档向量
        doc_vector = self.tfidf_retrieval.doc_vectors[document_id].toarray()
        
        # 找出共同词汇
        query_terms = []
        doc_terms = []
        
        # 查询分词
        query_tokens = self.tfidf_retrieval.tokenizer.tokenize_document(query_text, keep_pos=False)
        
        for token in query_tokens:
            if token in self.tfidf_retrieval.vocabulary:
                idx = self.tfidf_retrieval.vocabulary[token]
                if query_vector[0, idx] > 0:
                    query_terms.append((token, query_vector[0, idx]))
        
        # 文档词汇
        for word, idx in self.tfidf_retrieval.vocabulary.items():
            if doc_vector[0, idx] > 0:
                doc_terms.append((word, doc_vector[0, idx]))
        
        # 共同词汇及其贡献
        common_terms = []
        for q_term, q_weight in query_terms:
            if q_term in [d_term for d_term, _ in doc_terms]:
                idx = self.tfidf_retrieval.vocabulary[q_term]
                d_weight = doc_vector[0, idx]
                contribution = q_weight * d_weight
                common_terms.append((q_term, q_weight, d_weight, contribution))
        
        # 计算最终相似度
        similarity = self.cosine_similarity_manual(query_vector, doc_vector)
        
        return {
            "查询词汇": query_terms,
            "共同词汇": sorted(common_terms, key=lambda x: x[3], reverse=True),
            "相似度": similarity,
            "文档ID": document_id,
            "文档标题": self.documents[document_id].get('title', '')
        }
```

### 4.3 检索系统集成

```python
class ChineseNewsSearchSystem:
    def __init__(self):
        self.tokenizer = OptimizedChineseTokenizer()
        self.tfidf_retrieval = TFIDFRetrieval(self.tokenizer)
        self.cosine_retrieval = None
        self.documents = []
        self.is_indexed = False
        
        # 性能统计
        self.performance_stats = {
            "index_build_time": 0,
            "avg_search_time": 0,
            "total_searches": 0
        }
    
    def index_documents(self, documents):
        """建立文档索引"""
        print("开始建立文档索引...")
        start_time = time.time()
        
        self.documents = documents
        
        # 1. 对所有文档进行分词
        print("文档分词中...")
        tokenized_docs = []
        for i, doc in enumerate(documents):
            if i % 100 == 0:
                print(f"分词进度: {i}/{len(documents)}")
            
            tokens = self.tokenizer.tokenize_document(doc['content'])
            tokenized_docs.append(tokens)
        
        # 2. 构建TF-IDF矩阵
        self.tfidf_retrieval.build_document_matrix(tokenized_docs)
        
        # 3. 初始化检索器
        self.cosine_retrieval = CosineRetrieval(self.tfidf_retrieval, documents)
        
        # 4. 记录性能
        build_time = time.time() - start_time
        self.performance_stats["index_build_time"] = build_time
        self.is_indexed = True
        
        print(f"索引构建完成，耗时: {build_time:.2f}秒")
        print(f"词汇表大小: {len(self.tfidf_retrieval.vocabulary)}")
        print(f"文档数量: {len(documents)}")
    
    def search(self, query, top_k=10):
        """执行检索"""
        if not self.is_indexed:
            raise Exception("请先调用 index_documents() 建立索引")
        
        start_time = time.time()
        
        # 执行检索
        results = self.cosine_retrieval.search(query, top_k)
        
        # 更新性能统计
        search_time = time.time() - start_time
        self.performance_stats["total_searches"] += 1
        
        # 更新平均搜索时间
        total_time = (self.performance_stats["avg_search_time"] * 
                     (self.performance_stats["total_searches"] - 1) + search_time)
        self.performance_stats["avg_search_time"] = total_time / self.performance_stats["total_searches"]
        
        return results, search_time
    
    def explain_search(self, query, document_id):
        """解释检索结果"""
        if not self.is_indexed:
            raise Exception("请先调用 index_documents() 建立索引")
        
        return self.cosine_retrieval.explain_similarity(query, document_id)
    
    def get_system_stats(self):
        """获取系统统计信息"""
        stats = {
            "索引状态": "已建立" if self.is_indexed else "未建立",
            "文档数量": len(self.documents),
            "词汇表大小": len(self.tfidf_retrieval.vocabulary) if self.is_indexed else 0,
            "索引构建时间": f"{self.performance_stats['index_build_time']:.2f}秒",
            "平均检索时间": f"{self.performance_stats['avg_search_time']:.4f}秒",
            "总检索次数": self.performance_stats["total_searches"]
        }
        
        if self.is_indexed:
            # 内存使用估算
            matrix_memory = self.tfidf_retrieval.doc_vectors.data.nbytes / (1024 * 1024)  # MB
            stats["索引内存使用"] = f"{matrix_memory:.2f} MB"
        
        return stats
    
    def save_index(self, filepath):
        """保存索引到文件"""
        if not self.is_indexed:
            raise Exception("没有可保存的索引")
        
        index_data = {
            'vocabulary': self.tfidf_retrieval.vocabulary,
            'idf_values': self.tfidf_retrieval.idf_values,
            'doc_vectors': self.tfidf_retrieval.doc_vectors,
            'documents': self.documents
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"索引已保存到: {filepath}")
    
    def load_index(self, filepath):
        """从文件加载索引"""
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.tfidf_retrieval.vocabulary = index_data['vocabulary']
        self.tfidf_retrieval.idf_values = index_data['idf_values']
        self.tfidf_retrieval.doc_vectors = index_data['doc_vectors']
        self.documents = index_data['documents']
        
        self.cosine_retrieval = CosineRetrieval(self.tfidf_retrieval, self.documents)
        self.is_indexed = True
        
        print(f"索引已从 {filepath} 加载")
```

## 5. 算法优化与评估

### 5.1 检索效果评估

```python
class RetrievalEvaluator:
    def __init__(self, search_system):
        self.search_system = search_system
    
    def create_test_queries(self):
        """创建测试查询集"""
        test_queries = [
            {"query": "人工智能", "type": "单词查询"},
            {"query": "新冠疫情防控", "type": "短语查询"},
            {"query": "经济发展 政策", "type": "多词查询"},
            {"query": "北京冬奥会开幕式", "type": "实体查询"},
            {"query": "碳达峰碳中和目标", "type": "复合概念查询"}
        ]
        return test_queries
    
    def evaluate_retrieval_quality(self, test_queries, human_judgments=None):
        """评估检索质量"""
        results = {}
        
        for query_data in test_queries:
            query = query_data["query"]
            search_results, search_time = self.search_system.search(query, top_k=10)
            
            # 基础指标
            results[query] = {
                "检索时间": search_time,
                "结果数量": len(search_results),
                "平均相似度": np.mean([r['similarity_score'] for r in search_results]) if search_results else 0,
                "最高相似度": max([r['similarity_score'] for r in search_results]) if search_results else 0,
                "相似度标准差": np.std([r['similarity_score'] for r in search_results]) if search_results else 0
            }
            
            # 如果有人工判断数据，计算precision等指标
            if human_judgments and query in human_judgments:
                relevant_docs = human_judgments[query]
                retrieved_docs = [r['document_id'] for r in search_results]
                
                precision = self.calculate_precision(retrieved_docs, relevant_docs)
                recall = self.calculate_recall(retrieved_docs, relevant_docs)
                f1 = self.calculate_f1(precision, recall)
                
                results[query].update({
                    "Precision@10": precision,
                    "Recall@10": recall,
                    "F1@10": f1
                })
        
        return results
    
    def calculate_precision(self, retrieved, relevant):
        """计算精确率"""
        if not retrieved:
            return 0
        
        relevant_retrieved = len(set(retrieved) & set(relevant))
        return relevant_retrieved / len(retrieved)
    
    def calculate_recall(self, retrieved, relevant):
        """计算召回率"""
        if not relevant:
            return 0
        
        relevant_retrieved = len(set(retrieved) & set(relevant))
        return relevant_retrieved / len(relevant)
    
    def calculate_f1(self, precision, recall):
        """计算F1分数"""
        if precision + recall == 0:
            return 0
        
        return 2 * (precision * recall) / (precision + recall)
```

这个技术规格文档详细说明了中文新闻稀疏检索系统的四个核心组件：

1. **数据获取**: 包含爬虫设计、数据结构、质量控制
2. **数据统计分析**: 全面的统计指标和可视化方案
3. **分词统计分析**: 结巴分词优化和详细的分词效果评估
4. **检索算法设计**: TF-IDF + 余弦相似度的完整实现

每个组件都提供了具体的代码实现和评估方法，确保系统的可实施性和效果可衡量性。