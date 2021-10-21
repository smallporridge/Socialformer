# Preprocess
This is the data processing code of paper **"Socialformer: Social Network Inspired Long Document Modeling for Document Ranking"**

## Usage
To run the code, you need to run `./run.sh` or follow the steps below.

+ Run `python gen_xxxx_weight.py` to pre-calculate the weights and save it.
+ Run `bash paste.sh` to concat the strategy weight and original passage according to columns, this  can reduce I/O operation.
+ Run `python main.py` to build the social network and seperate it to subgraphs

In addition, if you want to accelerate the whole process, you can call function multiprocess.
