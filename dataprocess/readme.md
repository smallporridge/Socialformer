This repository is the data processing code of paper **"Socialformer: Social Network Inspired Long Document Modeling for Document Ranking"**

To run the code, you need to run `./run.sh` or follow the steps below.

**First**, run scripts `gen_xxxx_weight.py` to pre-calculate the weights and save it.

**Then**, run `paste.sh` to concat the strategy weight and original passage according to columns, this  can reduce I/O operation.

Through the above two operations, we can read the weights statics line by line and reduce memory usage.

**Last**, run `main.py` to build the social network and seperate it to subgraphs

In addition, if you want to accelerate the whole process, you can call function multiprocess.
