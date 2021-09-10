python runBert.py \
		--per_gpu_batch_size 1 \
		--bert_model /home/dou/replearn/transformers_models/bert \
		--train_file ../datas/rand/ \
		--save_path ../outputs/output1 \
		--dataset_script_dir ../data_scripts \
		--dataset_cache_dir /tmp/negs_tutorial_cache \
		--log_path ../logs/pretrain.log.txt 
		
