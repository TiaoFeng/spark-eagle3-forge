#!/usr/bin/env python3
"""Render probe prompts with the target model's own chat template.

The coverage probe has to use the same prompt format as real serving, otherwise
the target goes off-distribution and the sampled tokens say nothing about the
workload. The template is read straight out of the GGUF, so it cannot drift from
the model.

Usage:
    python scripts/make_prompts.py --model M.gguf --out-dir datasets/prompts/rendered
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

try:
    import jinja2
except ImportError:  # pragma: no cover
    sys.exit("needs jinja2")
try:
    from gguf import GGUFReader
except ImportError:  # pragma: no cover
    sys.exit("needs the gguf package")

# workload -> (user text, enable_thinking)
PROBES = {
    "chat_think": (
        "一個袋子裡有 3 顆紅球和 5 顆藍球。隨機一次取出兩顆球，"
        "求兩顆同色的機率。請一步一步推理，最後給出分數與小數。",
        True,
    ),
    "chat_nothink": (
        "你好，請用繁體中文簡單介紹一下你自己，以及你擅長處理哪些任務。",
        False,
    ),
    "prose_think": (
        "請寫一段關於秋天午後山間小路的散文，大約兩百字，語調安靜。",
        True,
    ),
    "prose_nothink": (
        "請寫一段關於秋天午後山間小路的散文，大約兩百字，語調安靜。",
        False,
    ),
    "code": (
        "用 Python 寫一個函式，回傳最長遞增子序列的長度，並說明時間複雜度。",
        True,
    ),
    "copy": (
        "請完整重述下面這段文字，不要增刪任何字元：\n\n"
        "訓練特徵放記憶體，不要反覆寫 NVMe。這個約束是為了保護 SSD 壽命，"
        "因為特徵容量可能達到數十 GB，每輪訓練都重寫一次會快速消耗寫入壽命。",
        True,
    ),
}

# The usage pattern this project actually targets: Simplified Chinese prompts and
# a thinking channel that mixes English and Chinese. Reasoning traces in that
# regime are neither pure ASCII nor pure Chinese, so the synthetic set alone
# cannot tell us what p0 ceiling a real session gets.
REALISTIC = {
    "zh_tech": (
        "解释一下什么是 KV cache，以及它为什么能加速大模型推理。",
        True,
    ),
    "zh_daily": (
        "最近工作有点累，有什么放松的方法吗？",
        True,
    ),
    "zh_math": (
        "一个水池有甲、乙两个进水管。甲管单独注满需要 6 小时，乙管单独注满需要 4 小时。"
        "两管同时打开，注满水池需要多久？",
        True,
    ),
    "zh_code": (
        "写一个 Python 脚本，扫描指定目录下所有 .log 文件，统计每个错误关键字出现的次数，"
        "并输出前 10 名。",
        True,
    ),
    "zh_rewrite": (
        "把下面这句话润色得更简洁：由于最近天气状况十分不理想，因此我们决定将原定于本周六举行的"
        "户外活动延期至下周六举行。",
        True,
    ),
    "zh_summary": (
        "请用三点总结下面这段文字：\n\n"
        "投机解码先用一个较小的草稿模型生成若干候选 token，再让目标模型一次性验证这些候选。"
        "如果草稿质量高，验证通过的 token 会被直接接受，从而减少目标模型的前向次数；"
        "如果草稿质量差，被拒绝的 token 会被丢弃，反而增加了额外的验证开销。"
        "因此收益高度依赖草稿模型的接受率，而接受率又取决于草稿与目标模型在同一上下文下的分布是否接近。",
        True,
    ),
    "zh_advice": (
        "我在犹豫要不要换工作，新公司工资高 30% 但要经常加班，帮我分析一下利弊。",
        True,
    ),
    "zh_translate": (
        "把下面这句英文翻成简体中文：The KV cache trades memory for compute.",
        True,
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--set", choices=("synthetic", "realistic"), default="synthetic",
                    help="synthetic = five invented workloads; realistic = the actual usage "
                         "pattern (Simplified Chinese prompts, thinking on, mixed-language thoughts)")
    args = ap.parse_args()

    probes = REALISTIC if args.set == "realistic" else PROBES

    reader = GGUFReader(args.model)
    field = reader.fields.get("tokenizer.chat_template")
    if field is None:
        sys.exit("model has no tokenizer.chat_template")
    template = str(field.contents())

    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)

    def raise_exception(msg: str):
        raise RuntimeError(msg)

    env.globals["raise_exception"] = raise_exception
    tmpl = env.from_string(template)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name, (user, thinking) in probes.items():
        text = tmpl.render(
            messages=[{"role": "user", "content": user}],
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        path = out / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        manifest[name] = {
            "path": str(path),
            "thinking": thinking,
            "chars": len(text),
            "tail": text[-40:],
        }
        print(f"{name:14s} thinking={str(thinking):5s} chars={len(text):5d} tail={text[-32:]!r}")

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
