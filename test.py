#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import json
def make_example_parquet(path: str):
    """生成一个示例 parquet 文件"""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["alice", "bob", "carol"],
        "score": [95.5, 88.0, 76.5],
    })
    df.to_parquet(path, index=False)
    return df

def read_parquet(path: str, engine: str = "pyarrow", columns=None, head: int = 5, row: int = 0):
    """读取 parquet 并打印基本信息 + 完整打印一条记录"""
    df = pd.read_parquet(path, engine=engine, columns=columns)

    print(f"[OK] Loaded parquet: {path}")
    print(f"Shape: {df.shape}")
    print("Dtypes:")
    print(df.dtypes)

    print(f"\nHead({head}):")
    print(df.head(head))

    # ===== 完整打印一条 =====
    if len(df) == 0:
        print("\n[WARN] parquet is empty.")
    else:
        row = max(0, min(row, len(df)-1))
        print(f"\nFull row[{row}]:")
        print(df.iloc[row].to_string())  # 不截断、完整打印一条
        print(json.dumps(df.iloc[row].to_dict(), indent=2, ensure_ascii=False))

    return df

def main():
    parser = argparse.ArgumentParser(description="Read a parquet file (with an example).")
    parser.add_argument("--path", type=str, default="example.parquet",
                        help="parquet 文件路径；不提供将自动生成 example.parquet")
    parser.add_argument("--engine", type=str, default="pyarrow",
                        choices=["pyarrow", "fastparquet"],
                        help="读取引擎")
    parser.add_argument("--columns", type=str, nargs="*", default=None,
                        help="只读取指定列，比如 --columns id score")
    parser.add_argument("--head", type=int, default=5, help="打印前几行")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"[INFO] {args.path} not found, creating an example parquet...")
        make_example_parquet(args.path)

    read_parquet(args.path, engine=args.engine, columns=args.columns, head=args.head)

# python test.py --path geo3k/train.parquet --head 3
if __name__ == "__main__":
    main()