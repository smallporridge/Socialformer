# Socialformer
Source code of WWW2022 long paper "Socialformer: Social Network Inspired Long Document Modeling for Document Ranking". 

## Requirements
+ python == 3.7.0
+ pytorch == 1.9.0 (with GPU support)
+ Transformers == 4.8.1
+ pytrec-eval == 0.5

More details please refer to 'requirements.txt'.

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

## Citations
If you use the code, please cite the following paper:  
```
@inproceedings{ZhouDYM22,
  author    = {Yujia Zhou and
               Zhicheng Dou and
               Huaying Yuan and
               Zhengyi Ma},
  title     = {Socialformer: Social Network Inspired Long Document Modeling for Document Ranking},
  booktitle = {{WWW}},
  publisher = {{ACM}},
  year      = {2022}
}
