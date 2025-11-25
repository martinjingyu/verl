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

# python vcbench.py --csv_path VCbench.csv --local_save_dir /var/lib/condor/execute/slot1/dir_1843585/scratch/verl/data/vcbench
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