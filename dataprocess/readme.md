This is the data processing code of paper "Socialformer: Social Network Inspired Long Document Modeling for Document Ranking"

## Usage
To run the code, you need to run 
`cd dataprocess`
`bash ./run.sh` 
or follow the steps below.

+ Run `python gen_xxxx_weight.py` to pre-calculate the weights.
+ Run `paste -d $ data/demo.json weights/tfidf.txt weights/bert.txt weights/dynamic.txt > weights/total.txt` to concat the strategy weight and original passage according to columns, this  can reduce I/O operation.
+ Run `python main.py` to build the social network and seperate it to subgraphs


## Other description

+ If you want to accelerate the whole process, you can call function multiprocess.

+ We precalculate the probablity weights of the dataset, this will save a lot of time. More specific, we save probablity matrix for static distance method, and doucument word weight for the other three method.

