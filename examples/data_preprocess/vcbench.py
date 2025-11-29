"""
Preprocess VCBench csv dataset to parquet format for VERL.

Expected CSV columns:
- success: 0 or 1 (label)
- anonymised_prose: text description (input)

Usage:
python vcbench_preprocess.py --csv_path /path/to/vcbench.csv --local_save_dir ~/data/vcbench
Optionally split train/test by ratio:
python vcbench_preprocess.py --csv_path vcbench.csv --local_save_dir ~/data/vcbench --test_ratio 0.1 --seed 42
"""

import argparse
import os
import pandas as pd
from datasets import Dataset, DatasetDict
import json
from verl.utils.hdfs_io import copy, makedirs


SYSTEM_PROMPT = """You are an expert in venture capital tasked with identifying successful founders from their unsuccessful counterparts. 
All founders under consideration are sourced from LinkedIn and Crunchbase profiles of companies that have raised between $100K and $4M in funding. 
A successful founder is defined as one whose company has achieved either a total funding of over $500M or an exit/IPO valued at over $500M."""

USER_PROMPT_TMPL = """Given the following founder description:
       {anonymised_prose},
       please output a json string with two keys; 
       1. prediction: 'Yes' or 'No' corresponding to whether or not the founder will be successful.
       2. reasoning: a short explanation for your prediction (at most 100 words).
    DO NOT return anything else"""


def label_to_text(x: int) -> str:
    return "Yes" if int(x) == 1 else "No"


def build_verl_rows(df: pd.DataFrame, split: str, data_source: str):
    rows = []
    for idx, r in df.iterrows():
        prose = str(r["anonymised_prose"])
        label = int(r["success"])
        gt = label_to_text(label)

        user_prompt = USER_PROMPT_TMPL.format(anonymised_prose=prose)

        rows.append(
            {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "ability": "vc",  # 你可以改成 "finance" / "classification" 等
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {
                    "split": split,
                    "index": int(idx),
                    "success": label,
                    "ground_truth": gt,
                    "anonymised_prose": prose,
                },
            }
        )
    return rows

def repeat_ranges(df, ranges, times=10):
    reps = []
    n = len(df)
    for (l, r) in ranges:
        l = max(0, l)
        r = min(n - 1, r)
        if l <= r:
            reps.append(df.iloc[l:r+1])
    if not reps:
        return df
    to_repeat = pd.concat(reps, ignore_index=True)

    # 复制 times-1 次（原本 df 里已有 1 份）
    repeated = pd.concat([to_repeat] * (times - 1), ignore_index=True)
    df_new = pd.concat([df, repeated], ignore_index=True)
    return df_new
    
def synthetic_json_to_df(synthetic_outputs, split_blocks=True):
    """
    synthetic_outputs: list[dict], 每个 dict 至少有 key "response"
    response 里可能包含多条 founder description（通常用空行分隔）
    返回一个 df，列与 vcbench 对齐：anonymised_prose, success
    """
    texts = []
    for item in synthetic_outputs:
        resp = item.get("response", "")
        if not isinstance(resp, str):
            continue
        resp = resp.strip()
        if not resp:
            continue

        if split_blocks:
            # 按空行切成多条（VCBench 风格一般每条之间空一行）
            blocks = [b.strip() for b in resp.split("\n\n") if b.strip()]
        else:
            blocks = [resp]

        for b in blocks:
            # 可选：简单过滤太短/明显不是描述的块
            if len(b) < 20:
                continue
            texts.append(b)

    df_syn = pd.DataFrame({
        "anonymised_prose": texts,
        "success": [1] * len(texts)
    })
    return df_syn

# python vcbench.py --csv_path VCbench.csv --local_save_dir $PWD/../../vcbench
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to vcbench.csv")
    parser.add_argument("--local_save_dir", default="~/data/vcbench")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    csv_path = os.path.expanduser(args.csv_path)
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # basic sanity checks
    assert "success" in df.columns and "anonymised_prose" in df.columns, \
        f"CSV must contain columns success, anonymised_prose. Got: {df.columns.tolist()}"

    data_source = "vcbench"

    split_idx = 3000
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)

    
    with open("synthetic_outputs.json", "r") as f:
        synthetic_outputs = json.load(f)

    # 把 synthetic response 变成 positive df
    df_syn = synthetic_json_to_df(synthetic_outputs, split_blocks=True)
    print(f"Synthetic positives loaded: {len(df_syn)}")

    # 拼到训练集里
    df_train = pd.concat([df_train, df_syn], ignore_index=True)
    
    
    # repeat_ranges_list = [(0, 136), (1502, 1636)]
    # df_train = repeat_ranges(df_train, repeat_ranges_list, times=15)
    
    df_train = df_train.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    
    train_rows = build_verl_rows(df_train, "train", data_source)
    test_rows = build_verl_rows(df_test, "test", data_source)
    
    dset = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "test": Dataset.from_list(test_rows),
        }
    )

    dset["train"].to_parquet(os.path.join(local_save_dir, "train.parquet"))
    dset["test"].to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)

    print(f"Saved parquet to: {local_save_dir}")