import datasets
from transformers import T5Config
import argparse

from t5_tokenizer_model import SentencePieceUnigramTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_data_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=32_000)
    parser.add_argument("--input_sentence_size", type=int, default=None)
    parser.add_argument("--output_tokenizer_file", type=str, default="./tokenizer.json")
    parser.add_argument("--output_config_dir", type=str, default="./outputs/")
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    # Initialize a dataset
    dataset = datasets.load_dataset("json", data_files=args.pretrain_data_file, split="train")

    tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

    # Build an iterator over this dataset
    def batch_iterator(input_sentence_size=None):
        if input_sentence_size is None:
            input_sentence_size = len(dataset)
        batch_length = 100
        for i in range(0, input_sentence_size, batch_length):
            yield dataset[i: i + batch_length]["text"]

    # Train tokenizer
    tokenizer.train_from_iterator(
        iterator=batch_iterator(input_sentence_size=args.input_sentence_size),
        vocab_size=args.vocab_size,
        show_progress=True,
    )

    # Save files to disk
    tokenizer.save(args.output_tokenizer_file)

    # Save config
    config = T5Config.from_pretrained(args.model_name_or_path, vocab_size=tokenizer.get_vocab_size())
    config.save_pretrained(args.output_config_dir)

if __name__ == "__main__":
    main()