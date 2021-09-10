python runBert.py \
		--per_gpu_batch_size 1 \
		--bert_model /home/dou/replearn/transformers_models/bert \
		--train_file /home/dou/replearn/pretrain4ir_tutorial/datas/rand \
		--save_path /home/dou/replearn/pretrain4ir_tutorial/outputs/output1 \
		--dataset_script_dir /home/dou/replearn/pretrain4ir_tutorial/data_scripts \
		--dataset_cache_dir /tmp/negs_tutorial_cache \
		--log_path /home/dou/replearn/pretrain4ir_tutorial/logs/pretrain.log.txt 
		
