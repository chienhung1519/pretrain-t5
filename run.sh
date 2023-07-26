# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

python prepare_data.py \
  --data_file="./data/raw/pretrain_pathreports.xlsx" \
  --output_file="./data/processed/pretrain_pathreports.json"

python train_tokenizer.py \
  --pretrain_data_file="./data/processed/pretrain_pathreports.json" \
  --model_name_or_path="lmsys/fastchat-t5-3b-v1.0" \
  --output_tokenizer_file="./fastchat-t5-3b-lung/tokenizer.json" \
  --output_config_dir="./fastchat-t5-3b-lung/"

python run_t5_mlm_flax.py \
	--output_dir="./fastchat-t5-3b-lung/" \
	--model_type="t5" \
	--config_name="./fastchat-t5-3b-lung" \
	--tokenizer_name="./fastchat-t5-3b-lung" \
	--train_file="./data/processed/pretrain_pathreports.json" \
	--dataset_config_name="unshuffled_deduplicated_no" \
	--max_seq_length="2048" \
	--per_device_train_batch_size="4" \
	--per_device_eval_batch_size="4" \
	--adafactor \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--overwrite_output_dir \
	--logging_steps="500" \
	--save_steps="10000" \
	--eval_steps="2500"