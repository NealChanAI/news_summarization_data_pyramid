# news_summarization_data_pyramid

## 数据预处理
### 对THUNews数据集进行随机采样处理
```
python src/data_prepare/get_thunews_datasets.py 
```

## 训练数据生成
### 抽取式摘要数据生成
```
python src/data_prepare/extractive_data.py
```

### 生成式摘要数据生成
```
python src/data_prepare/abstractive_data.py
```


