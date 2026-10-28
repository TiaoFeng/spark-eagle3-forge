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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

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
    for name, (user, thinking) in PROBES.items():
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
