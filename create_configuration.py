from transformers import T5Config

config = T5Config.from_pretrained("google/t5-v1_1-base", vocab_size=tokenizer.get_vocab_size())
config.save_pretrained("./norwegian-t5-base")