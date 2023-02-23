# pretrain-t5

## Requirements

## Download Data for Pretraining

## Train Tokenizer

```
python train_tokenizer.py
```

## Create Configuration

```
python create_configuration.py
```

## Train Model

```
python run_t5_mlm_flax.py \
	--output_dir="./norwegian-t5-base" \
	--model_type="t5" \
	--config_name="./norwegian-t5-base" \
	--tokenizer_name="./norwegian-t5-base" \
	--dataset_name="oscar" \
	--dataset_config_name="unshuffled_deduplicated_no" \
	--max_seq_length="512" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--adafactor \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--overwrite_output_dir \
	--logging_steps="500" \
	--save_steps="10000" \
	--eval_steps="2500" \
	--push_to_hub
```

## Reference

https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling

