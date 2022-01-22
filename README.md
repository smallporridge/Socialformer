# Socialformer
Source code of "Socialformer: Social Network Inspired Long Document Modeling for Document Ranking". 

+ For double-blind review, we use anonymous github to hide the information that may reveal the identity.
+ We would like to express our deep appreciation for all reviewers.

## Requirements
+ python >= 3.6.0
+ pytorch >= 1.9.0 (with GPU support)
+ Transformers >= 4.5.1
+ pytrec-eval == 0.5

## Dataset
+ [MS MARCO Document Ranking](https://github.com/microsoft/MSMARCO-Document-Ranking)
+ [TREC 2019 Deep Learning](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html)

## PreProcess
+ Referring to `/dataprocess`, including probability computing, graph construction, and graph partition.

```
cd dataprocess
bash ./run.sh
```

## Model training

+ Referring to `/model`, train and test the model.

```
bash ./model/run.sh
```
