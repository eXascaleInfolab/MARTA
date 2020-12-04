CUDA_VISIBLE_DEVICES=5 python main.py\
	--data_path '../data/processed_data/wiki_tech/'\
	--evaluation_file '../results/wiki_tech/test_scibert_time.csv'\
	--evaluation_file_val '../results/wiki_tech/val_scibert_time.csv'\
	--epochs_mil 25\
	--epochs_mil_retrain 20\
	--weight_dir '../results/wiki_tech/weight/'\
	--embedding_type 'scibert'\
	--compute_embedding 0\
	--shuffle 0\
	--max_iter 3\
