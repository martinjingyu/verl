import re
import math
import json
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np

# ---- embedding 模型（建议常用小模型，稳定省显存） ----
# 你也可以换成更强的 e5 / bge / gte
_EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def _cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

def _embed(text: str):
    return _EMB_MODEL.encode(text, normalize_embeddings=True)

def _input_output_sim_emb(founder_desc: str, solution_str: str) -> float:
    """
    embedding cosine similarity in [0,1] (roughly)
    并对过高相似度做轻微折扣，防止复读输入。
    """
    if not founder_desc:
        return 0.0

    vin = _embed(founder_desc)
    vout = _embed(solution_str)
    sim = _cosine(vin, vout)  # [-1,1] but usually [0,1] for these models

    # clamp to [0,1]
    sim = max(0.0, min(1.0, sim))

    # 复读折扣：>0.75 开始回落
    if sim <= 0.75:
        return sim
    else:
        # 0.75~1.0 区间线性回落到 0.75
        return 0.75 - (sim - 0.75) * 0.5  # sim=1 -> 0.625


# ===== 你原来的 reward 主体（保持不变） =====
def _extract_yes_no(solution_str: str):
    text = solution_str.strip().lower()
    tokens = re.findall(r"\b(yes|no)\b", text)
    return tokens[0] if tokens else None

def compute_score(
    data_source,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    **kwargs
) -> float:

    if extra_info is None:
        extra_info = {}
    breakdown: Dict[str, float] = {}

    # 1) base_score
    pred = _extract_yes_no(solution_str)
    gt_yes = (ground_truth.strip().lower() == "yes")

    if pred is None:
        base_score = 0.0
        pred_yes = None
    else:
        pred_yes = (pred == "yes")
        if pred_yes and gt_yes:
            base_score = 1.0
        elif pred_yes and not gt_yes:
            base_score = 0.5
        elif (not pred_yes) and (not gt_yes):
            base_score = 0.8
        else:
            base_score = 0.2
    breakdown["base_score"] = base_score

    # 2) decisiveness
    decisiveness_score = 1.0 if pred is not None else 0.0
    breakdown["decisiveness_score"] = decisiveness_score

    # 3) brevity
    n_words = len(re.findall(r"\w+", solution_str.lower()))
    n_words_thresh = 512
    n_words_thresh_max = 896
    if n_words <= n_words_thresh:
        brevity_score = 1.0
    else:
        brevity_score = max(
            0.0,
            1.0 - (n_words - n_words_thresh) / (n_words_thresh_max - n_words_thresh)
        )
    breakdown["brevity_score"] = brevity_score

    # 4) embedding similarity
    founder_desc = extra_info.get("anonymised_prose")
    sim_score = _input_output_sim_emb(founder_desc, solution_str)
    breakdown["sim_score_emb"] = sim_score

    # final weighted sum
    w_base = 0.80
    w_decisive = 0.05
    w_brevity = 0.05
    w_sim = 0.1  # 小权重，防止复读导向

    final_score = (
        w_base * base_score +
        w_decisive * decisiveness_score +
        w_brevity * brevity_score +
        w_sim * sim_score
    )

    extra_info["score_breakdown"] = breakdown
    extra_info["final_score"] = final_score
    return final_score