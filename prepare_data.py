import pandas as pd
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="./outputs.json")
    args = parser.parse_args()
    return args

def load_data(data_file):
    data = pd.read_excel(data_file, sheet_name=None)
    sheet = pd.ExcelFile(data_file)

    outputs = []
    for s_name in sheet.sheet_names:
        df = data.get(s_name)
        outputs.extend(df.iloc[:, 0].values.tolist())

    return outputs

def save_data(outputs, output_file):
    with open(output_file, "w") as f:
        for o in outputs:
            f.write(json.dumps({"text": o}, ensure_ascii=False) + "\n")
        
def main():
    args = parse_args()
    outputs = load_data(args.data_file)
    save_data(outputs, args.output_file)

if __name__ == "__main__":
    main()