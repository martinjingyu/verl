


def compute_score(data_source, solution_str: str, ground_truth: str, extra_info=None, **kwargs) -> float:
    
    score = 0.0
    
    if "yes" in solution_str.lower():
        # False positive case, biggest penalty
        if ground_truth == "No":
           score = 0.0
        # True positive case, biggest reward
        else:
           score = 1
           
    if "no" in solution_str.lower():
        # True negative case, biggest reward
        if ground_truth == "No":
           score = 1
        # False negative case, medium penalty
        else:
           score = 0.5
           
    return score
