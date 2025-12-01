import re
import math
from typing import Dict, Any

POS_CUES = [
    "traction", "growing", "revenue", "retention", "product market fit",
    "experienced team", "repeat founder", "strong demand", "profit",
    "scalable", "enterprise customers", "moat"
]
NEG_CUES = [
    "no traction", "pivot", "burning cash", "unclear market",
    "solo founder", "high churn", "regulatory risk", "commoditized",
    "no revenue", "weak team", "no moat"
]

def _extract_yes_no(solution_str: str):
    text = solution_str.strip().lower()

    if "<think>" in text and "</think>" in text:
        try:
            text = text.split("<think>", 1)[1].split("</think>", 1)[0].strip().lower()
        except Exception:
            pass

    tokens = re.findall(r"\b(yes|no)\b", text)
    if not tokens:
        return None
    return tokens[0]

def _cue_score(text: str) -> float:
    t = text.lower()
    pos = sum(1 for w in POS_CUES if w in t)
    neg = sum(1 for w in NEG_CUES if w in t)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)

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

    # ---------- 1) base_score（硬标签） ----------
    pred = _extract_yes_no(solution_str)
    gt = ground_truth.strip().lower()
    gt_yes = (gt == "yes")

    if pred is None:
        base_score = 0.0
        pred_yes = None
    else:
        pred_yes = (pred == "yes")
        if pred_yes and gt_yes:
            base_score = 1.0      # TP
        elif pred_yes and not gt_yes:
            base_score = 0.5      # FP
        elif (not pred_yes) and (not gt_yes):
            base_score = 0.8      # TN
        else:
            base_score = 0.2      # FN

    breakdown["base_score"] = base_score

    # ---------- 2) decisiveness_score（是否明确表态） ----------
    decisiveness_score = 1.0 if pred is not None else 0.0
    breakdown["decisiveness_score"] = decisiveness_score


    # ---------- 4) brevity_score（短而准） ----------
    # 只做轻微 shaping：不看语义，只惩罚超长
    lower = solution_str.lower()
    n_words = len(re.findall(r"\w+", lower))
    # 经验阈值：>160 词开始线性惩罚；最低不低于 0
    n_words_thresh = 160
    n_words_thresh_max = 360
    if n_words <= n_words_thresh:
        brevity_score = 1.0
    else:
        brevity_score = max(0.0, 1.0 - (n_words - n_words_thresh) / (n_words_thresh_max-n_words_thresh))  # 约 360 词惩罚到 0
    breakdown["brevity_score"] = brevity_score

    # ---------- 5) evidence_align_score（证据线索对齐） ----------
    # 预测 yes -> 理由应偏正向线索；预测 no -> 偏负向线索
    cue = _cue_score(solution_str)  # [-1,1]
    if pred_yes is None:
        evidence_align_score = 0.0
    else:
        align = cue if pred_yes else -cue
        evidence_align_score = max(0.0, align)  # 只奖励一致，不额外惩罚
    breakdown["evidence_align_score"] = evidence_align_score

    # ---------- 最终加权 ----------
    # 权重设置：稳妥优先，软信号权重都很小
    w_base = 0.80
    w_decisive = 0.05
    w_brevity = 0.05
    w_evidence = 0.05  # 这项启发式最弱，千万别给大

    final_score = (
        w_base * base_score +
        w_decisive * decisiveness_score +
        w_brevity * brevity_score +
        w_evidence * evidence_align_score
    )

    extra_info["score_breakdown"] = breakdown
    extra_info["final_score"] = final_score

    return final_score