# BDCI 2021 产品观点提取 B榜Top34解决方案
---
NER部分使用5个Roberta-large+CRF

情感分类部分使用5个ERNIE模型

对于情感分类任务，复制扩增了标签0、1，使用分层训练的方法，首先训练dence层后训练BERT的最后一层Transformers，之后再训练倒数第二层Transformers，最后整体Fine-tune
