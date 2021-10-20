python train.py \
		--is_training \
		--per_gpu_batch_size 1 --per_gpu_test_batch_size 10 --task msmarco \
		--bert_model /home/yujia_zhou/long_doc/anchors-master/models/bert \
		--dataset_script_dir /home/yujia_zhou/long_doc/anchors-master/data_scripts \
		--dataset_cache_dir /home/yujia_zhou/long_doc/finetune/negs_cache_listwise_2048 \
		--log_path ./log.txt \
		--train_file /home/yujia_zhou/long_doc/anchors-master/new_data/list_train \
		--dev_file  /home/yujia_zhou/long_doc/anchors-master/new_data/marco_dev/cat_all.json \
		--dev_id_file /home/yujia_zhou/long_doc/anchors-master/new_data/marco_dev/ids.tsv \
		--msmarco_score_file_path ./score_bert_all.txt \
		--msmarco_dev_qrel_path /home/yujia_zhou/long_doc/anchors-master/data/msmarco/msmarco-docdev-qrels.tsv \
		--save_path /home/yujia_zhou/long_doc/anchors-master/output/bert_listwise_2048/pytorch_model.bin > /home/yujia_zhou/long_doc/anchors-master/logs/bert_listwise_2048.bs1.pointwise.log