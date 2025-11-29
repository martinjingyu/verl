import re
from pathlib import Path
from typing import List, Dict

# ===== 1) 读 txt + 抽公司名 =====
def extract_company_names(txt_path: str) -> List[str]:
    """
    从 txt 每行抽公司名：
    - 若有 tab 分隔 -> 取第一个字段
    - 否则用多个空格分隔 -> 取第一个字段
    - 自动跳过空行
    """
    names = []
    for line in Path(txt_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        if "\t" in line:
            parts = line.split("\t")
        else:
            # 按 >=2 个空格切分，避免把公司名里的单空格拆了（如果有的话）
            parts = re.split(r"\s{2,}", line)

        company = parts[0].strip()
        if company:
            names.append(company)

    # 去重但保序
    seen = set()
    uniq = []
    for n in names:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq


# ===== 2) VCbench 风格生成 prompt 模板 =====
PROMPT_TEMPLATE = """You are a data synthesis assistant.

Task:
Given a successful startup and its founder(s), write founder descriptions in the SAME style as VCBench positives.

Rules:
1. Output format must strictly follow VCBench style:
   - One paragraph: "This founder leads a startup in the ... industry."
   - "Education:" section with bullet points
   - "Professional experience:" section with bullet points
2. DO NOT mention funding rounds, valuation, IPO, acquisition, VC names, accelerators, or the word "unicorn".
3. Use neutral tone. No bragging, no explicit success conclusion.
4. Only imply success via education + career trajectory + industries + company sizes + years.
5. Keep each description realistic and internally consistent.

VCBench positive examples (style reference):
Example 1:
This founder leads a startup in the Technology, Information & Internet Platforms industry.
Education:
* BA in Computer Science (Institution QS rank 1)
Professional experience:
* CTO for <2 years in the `Sports Teams & Leagues` industry (myself only employees)
* CTO for <2 years in the `E-Learning` industry (2-10 employees)
* Software Engineer for <2 years in the `Wellness & Community Health` industry (2-10 employees)
* Graduate Fellow (NSF) for 4-5 years in the `Environmental & Waste Services` industry (201-500 employees)

Now generate 5 NEW founder descriptions for the following successful company:

Company name: "{company_name}"

You may assume plausible founder background for this company that matches real-world typical successful founders,
but do NOT add any explicit success markers.
Return ONLY the 5 descriptions, separated by a blank line.
"""


def build_prompts(company_names: List[str]) -> Dict[str, str]:
    """对每个公司名生成一个 prompt"""
    return {name: PROMPT_TEMPLATE.format(company_name=name) for name in company_names}


# ===== 3) 主函数：读 -> 生成 prompt -> 保存 =====
def main(
    txt_path: str,
    out_dir: str = "prompts_out"
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    company_names = extract_company_names(txt_path)
    prompts = build_prompts(company_names)

    # 每个公司一个 prompt 文件
    for name, prompt in prompts.items():
        safe_name = re.sub(r"[^\w\-]+", "_", name)[:80]
        out_file = Path(out_dir) / f"{safe_name}.prompt.txt"
        out_file.write_text(prompt, encoding="utf-8")

    # 也额外写一个总表
    all_file = Path(out_dir) / "all_prompts.jsonl"
    with all_file.open("w", encoding="utf-8") as f:
        for name, prompt in prompts.items():
            f.write(
                f'{{"company_name": "{name}", "prompt": {prompt!r}}}\n'
            )

    print(f"✅ extracted companies: {len(company_names)}")
    print(f"✅ prompts saved to: {out_dir}/")


if __name__ == "__main__":
    # 用法：python make_prompts.py your_file.txt
    import sys
    if len(sys.argv) < 2:
        print("Usage: python make_prompts.py <companies.txt>")
        raise SystemExit(0)

    main(sys.argv[1])