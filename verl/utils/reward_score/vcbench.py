import re
import json
import numpy as np
from functools import lru_cache
from typing import Dict, Any

# ---------- lazy embedding model ----------
_EMB_MODEL = None

def _get_emb_model():
    global _EMB_MODEL
    if _EMB_MODEL is None:
        from sentence_transformers import SentenceTransformer
        # 小模型足够做相似度 shaping，且更稳
        _EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMB_MODEL

def _cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))

@lru_cache(maxsize=4096)
def _embed_cached(text: str):
    model = _get_emb_model()
    v = model.encode(text, normalize_embeddings=True)
    return v.astype(np.float32)

def _sim_emb(founder_desc: str, reasoning: str) -> float:
    if not founder_desc or not reasoning:
        return 0.0
    vin = _embed_cached(founder_desc)
    vout = _embed_cached(reasoning)
    sim = _cosine(vin, vout)
    sim = max(0.0, min(1.0, sim))

    # 防复读：过高相似度轻微回落
    if sim <= 0.75:
        return sim
    return 0.75 - (sim - 0.75) * 0.5  # sim=1 -> 0.625


# ---------- robust prediction extraction ----------
def _extract_yes_no(solution_str: str):
    text = solution_str.strip().lower()
    tokens = re.findall(r"\b(yes|no)\b", text)
    if not tokens:
        return None
    return tokens[-1]  # 取最后一次明确表态更稳

def _extract_reasoning(solution_str: str):
    """
    尽量从 JSON 里拿 reasoning 来算 embedding 相似度。
    拿不到就退化成全串（但相似度权重很小，不会伤训练）。
    """
    try:
        if "</think>" in solution_str:
           reasoning = solution_str.split("</think>")[0].strip()
           return reasoning
        obj = json.loads(solution_str)
        r = obj.get("reasoning", "")
        if isinstance(r, str):
            return r.strip()
    except Exception:
        pass
    return solution_str


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
        if pred_yes and gt_yes: base_score = 1.0
        elif pred_yes and not gt_yes: base_score = 0.5
        elif (not pred_yes) and (not gt_yes): base_score = 0.8
        else: base_score = 0.2

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

    # 4) embedding similarity (reasoning vs input)
    founder_desc = (
        extra_info.get("anonymised_prose")
        or extra_info.get("input")
        or extra_info.get("prompt")
        or kwargs.get("founder_description")
        or ""
    )
    reasoning = _extract_reasoning(solution_str)
    sim_score = _sim_emb(founder_desc, reasoning)
    breakdown["sim_score_emb"] = sim_score

    # final weighted sum
    w_base = 0.85
    w_decisive = 0.05
    w_brevity = 0.05
    w_sim = 0.05   # embedding 必须小权重

    final_score = (
        w_base * base_score +
        w_decisive * decisiveness_score +
        w_brevity * brevity_score +
        w_sim * sim_score
    )

    extra_info["score_breakdown"] = breakdown
    extra_info["final_score"] = final_score
    return final_score