# pretrain4ir_tutorial
NLPIR tutorial: pretrain for IR. pre-train on raw textual corpus, fine-tune on MS MARCO Document Ranking

用作NLPIR实验室, Pre-training for IR方向入门.

代码包括了如下部分:
- ```tasks/``` : 生成预训练数据  
- ```pretrain/```: 在生成的数据上Pre-training (MLM + NSP) 
- ```finetune/```: Fine-tuning on [MS MARCO](https://github.com/microsoft/MSMARCO-Document-Ranking)


## Preinstallation

First, prepare a **Python3** environment, and run the following commands:
```
  git clone git@github.com:zhengyima/pretrain4ir_tutorial.git pretrain4ir_tutorial
  cd pretrain4ir_tutorial
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Besides, you should download the BERT model checkpoint in format of huggingface transformers, and save them in a directory ```BERT_MODEL_PATH```. In our paper, we use the version of ```bert-base-uncased```. you can download it from the huggingface official [model zoo](https://huggingface.co/bert-base-uncased/tree/main), or [Tsinghua mirror](https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/).

## 生成预训练数据

代码库提供了最简单易懂的预训练任务 ```rand```。该任务随机从文档中选取1~5个词作为query, 用来demo面向IR的预训练。

生成rand预训练任务数据命令:
```cd tasks/rand && bash gen.sh```

你可以自己编写脚本, 仿照rand任务, 生成你自己认为合理的预训练任务的数据。

**Notes**: 运行rand任务的shell之前, 你需要先将 ```gen.sh``` 脚本中的 ```msmarco_docs_path``` 参数改为MSMARCO数据集的 [文档tsv](https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz) 路径; 将```bert_model```参数改为下载好的bert模型目录; 


## 模型预训练

代码库提供了模型预训练的相关代码, 见```pretrain```。该代码完成了MLM+NSP两个任务的预训练。

模型预训练命令:
```cd pretrain && bash train_bert.sh```

**Notes**: 注意要修改```train_bert```中的相应参数：将```bert_model```参数改为下载好的bert模型目录;  ```train_file```改为你上一步生成好的预训练数据文件路径。

## 模型Fine-tune

代码库提供了在MSMARCO Document Ranking任务上进行Fine-tune的相关代码。见```finetune```。该代码完成了在MSMARCO上通过point-wise进行fine-tune的流程。

模型fine-tune命令:
```cd finetune && bash train_bert.sh```

## Leaderboard

| Tasks | MRR@100 on dev set | 
| :---------------- | :---------------|
| PROP-MARCO | 0.4201 |
| PROP-WIKI | 0.4188 |
| BERT-Base | 0.4184 |
| rand | 0.4123 |


## ~~Homework~~

设计一个你认为合理的预训练任务, 并对BERT模型进行预训练, 并在MSMARCO上完成fine-tune, 在Leaderboard上更新你在dev set上的结果。

你需要做的是:
- 编写你自己的预训练数据生成脚本, 放到 ```tasks/yourtask``` 目录下。
- 使用以上脚本, 生成自己的预训练数据。
- 运行代码库提供的pre-train与fine-tune脚本, 跑出结果, 更新Leaderboard。

## Links
- [Wikipedia dump](https://dumps.wikimedia.org/enwiki/)
- [WikiExtractor](https://github.com/attardi/wikiextractor)
- [MS MARCO Document Ranking](https://github.com/microsoft/MSMARCO-Document-Ranking)
- [TREC 2019 Deep Learning](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html)
- [PyTorch](https://pytorch.org)
- [Huggingface Transformers](https://huggingface.co/)

