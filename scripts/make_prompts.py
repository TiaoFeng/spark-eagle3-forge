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

BASE = pathlib.Path(__file__).resolve().parent.parent


def _section(rel: str, index: int, chars: int) -> str:
    """A stable excerpt: split the public doc on '## ' and take one section."""
    parts = (BASE / rel).read_text(encoding="utf-8").split("\n## ")
    if index >= len(parts):
        raise SystemExit("section %d not found in %s" % (index, rel))
    return parts[index][:chars].strip()


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

# The deployment mix for D0: six registers weighted roughly by token share in a
# real session (chat, chat without thinking, technical Q&A, code, structured
# output, long context). Entries are (user text, enable_thinking) or
# (user text, enable_thinking, [repo-relative context files]); the context is
# prepended to the user turn so the prompt stays a single message.
DAILY = {
    "daily_chat_01": ("最近工作有点累，有什么放松的方法吗？请给出至少五条具体建议。", True),
    "daily_chat_02": ("我想养成早起的习惯，但总是失败。请帮我分析原因，并给出一个可执行的两周计划。", True),
    "daily_chat_03": ("朋友最近情绪低落，我该怎么安慰他？请给出几种不同场景下的说法。", True),
    "daily_chat_04": ("请帮我设计一个适合周末在家做的三菜一汤菜单，并说明每道菜的做法。", True),
    "daily_chat_05": ("我想开始跑步，但膝盖不太好。请推荐几种替代运动，并说明注意事项。", True),
    "daily_chat_06": ("如何在三十天内提高英语听力？请给出每天的学习安排和材料建议。", True),
    "daily_chat_07": ("家里人总爱在吃饭时看手机，怎么沟通比较好？请给出一个具体的沟通方案。", True),
    "daily_chat_08": ("请推荐几本适合通勤时读的书，并说明每本书适合什么样的读者。", True),

    "daily_chatnt_01": ("你好，请用三句话介绍你自己，并说明你擅长哪些任务。", False),
    "daily_chatnt_02": ("请用一段一百五十字左右的文字解释什么是机器学习。", False),
    "daily_chatnt_03": ("把「今天天气很好」改成三种不同语气的说法，并说明适用场景。", False),
    "daily_chatnt_04": ("请推荐三个适合初学者的 Python 练习项目，并说明每个项目能练到什么。", False),
    "daily_chatnt_05": ("请用简体中文解释「沉没成本」，并举一个生活中的例子。", False),
    "daily_chatnt_06": ("请列出五种常见的 HTTP 状态码及其含义，并用一句话说明适用场景。", False),
    "daily_chatnt_07": ("请推荐三个适合周末短途旅行的目的地，并说明交通和住宿建议。", False),
    "daily_chatnt_08": ("请用一段话解释「少即是多」在设计中的含义，并举两个例子。", False),

    "daily_tech_01": ("解释一下什么是 KV cache，以及它为什么能加速大模型推理。请给出显存占用的估算方法。", True),
    "daily_tech_02": ("为什么 Transformer 需要位置编码？RoPE 相比绝对位置编码有什么优势？", True),
    "daily_tech_03": ("解释 speculative decoding 的原理，以及接受率如何影响端到端加速比。", True),
    "daily_tech_04": ("什么是模型量化？Q4_K 和 Q6_K 有什么区别？请从精度、显存和速度三方面说明。", True),
    "daily_tech_05": ("解释 MoE 模型的负载均衡问题，以及常见的辅助损失设计。", True),
    "daily_tech_06": ("为什么 GPU 上的大模型解码通常是显存带宽受限的？请给出估算公式和例子。", True),
    "daily_tech_07": ("解释 Flash Attention 的原理，它节省了哪些显存访问，又牺牲了什么。", True),
    "daily_tech_08": ("对比 LoRA 和全参数微调的优缺点，并说明什么情况下应该选哪一种。", True),

    "daily_code_01": ("用 Python 写一个函数，返回最长递增子序列的长度，并说明时间复杂度和空间复杂度。", True),
    "daily_code_02": ("写一个 Python 脚本，扫描指定目录下所有 .log 文件，统计每个错误关键字出现的次数，并输出前 10 名。", True),
    "daily_code_03": ("用 C++ 写一个线程安全的环形缓冲区，并解释其中使用的内存序。", True),
    "daily_code_04": ("写一个 bash 脚本，把目录下所有 PNG 转成 WebP，跳过已存在的文件，并打印转换统计。", True),
    "daily_code_05": ("用 Python 实现一个带过期时间的 LRU 缓存，并说明并发访问时如何保证正确性。", True),
    "daily_code_06": ("写一个 SQL 查询，找出每个部门工资最高的三名员工，并解释窗口函数的用法。", True),
    "daily_code_07": ("用 TypeScript 写一个 debounce 函数，说明 this 的处理方式，并给出使用示例。", True),
    "daily_code_08": ("写一个 Python 脚本，比较两个目录下同名文件的内容差异，并输出一份摘要报告。", True),

    "daily_struct_01": ("把下面这段会议记录整理成 JSON，字段包括结论、负责人和截止日期：\n\n"
                        "本周例会讨论了缓存层的改造。张工负责在下周五前完成方案评审，"
                        "李工负责补充压测数据。大家同意先在小流量环境灰度。", True),
    "daily_struct_02": ("把下面的需求写成 YAML 配置：服务名 search-api，副本数 3，镜像 tag v1.4.2，"
                        "CPU 请求 500m，内存请求 1Gi，端口 8080，健康检查路径 /health。", True),
    "daily_struct_03": ("下面这个函数有一个边界条件错误，请给出 unified diff 格式的修改建议：\n\n"
                        "```python\n"
                        "def clamp(x, lo, hi):\n"
                        "    if x < lo:\n"
                        "        return lo\n"
                        "    if x > hi:\n"
                        "        return hi\n"
                        "    return x\n"
                        "```\n\n"
                        "要求：当 lo > hi 时抛出 ValueError。", True),
    "daily_struct_04": ("把下面的数据整理成 Markdown 表格，并按得分从高到低排序：\n\n"
                        "Alice 92 分，Bob 78 分，Carol 85 分，Dave 78 分，Eve 95 分。", True),
    "daily_struct_05": ("写一个 systemd unit 文件，让 /opt/app/run.sh 开机自启、异常退出后自动重启，"
                        "并说明每个字段的作用。", True),
    "daily_struct_06": ("为下面这个 API 设计 OpenAPI 3.0 的路径定义：GET /v1/items 返回分页列表，"
                        "支持 page 和 page_size 查询参数，响应包含 items 数组和 total。", True),
    "daily_struct_07": ("写一个正则表达式，匹配形如 2026-10-30T14:05:00+08:00 的时间戳，"
                        "并逐段解释每个部分的含义。", True),
    "daily_struct_08": ("把下面这段文字改写成一条 50 字以内的 git commit message，并说明为什么这样写：\n\n"
                        "修复了训练脚本在最后一个 chunk 越界写入的问题，原因是预留行数少了一行。", True),

    "daily_longctx_01": ("请总结文档第 6 节和第 7 节的 gate 结论，并指出 G4 为什么被记为 missed。",
                         True, ["docs/notes.md"]),
    "daily_longctx_02": ("请解释为什么 p0 的上限是 draft 词表的 coverage，以及为什么原来的 32K 词表不够用。",
                         True, ["docs/notes.md"]),
    "daily_longctx_03": ("请列出模型的几何参数和 cost model 公式，并说明 n-max 为什么必须是 1。",
                         True, ["docs/facts.md"]),
    "daily_longctx_04": ("请说明 S2a 中哪些 register 提升最大、哪些几乎没有变化，并给出可能的原因。",
                         True, ["docs/notes.md"]),
    "daily_longctx_05": ("请说明 G1 的 offline 与 online 差异来源，以及为什么 copy 是 outlier。",
                         True, ["docs/notes.md"]),
    "daily_longctx_06": ("请说明 llama.cpp 需要哪些改动才能支持 EAGLE-3，以及为什么 shim 必须编译为 C++。",
                         True, ["docs/facts.md"]),
}

# D1: the NLP workload (translation, summarisation, document organisation,
# chat, long context). Sources are public docs in this repo or invented notes;
# the prompt text stays Chinese because the language is the subject.
NLP = {
    # Chinese -> English
    "nlp_zh2en_01": ("把下面这段中文翻译成英文，只输出译文，不要解释：\n\n"
                     + _section("README.cn.md", 0, 600), True),
    "nlp_zh2en_02": ("把下面这段中文翻译成英文，只输出译文，保留数字与专有名词：\n\n"
                     + _section("README.cn.md", 1, 600), True),
    "nlp_zh2en_03": ("把下面这段中文翻译成英文，只输出译文，不要解释：\n\n"
                     + _section("README.cn.md", 2, 600), True),
    "nlp_zh2en_04": ("把下面这段中文翻译成英文，只输出译文，保留列表结构：\n\n"
                     + _section("README.cn.md", 3, 600), True),
    "nlp_zh2en_05": ("把下面这段中文翻译成英文，只输出译文，不要解释：\n\n"
                     + _section("README.cn.md", 4, 600), True),
    "nlp_zh2en_06": ("把下面这段中文翻译成英文，只输出译文，保留代码与命令：\n\n"
                     + _section("README.cn.md", 5, 600), True),
    "nlp_zh2en_07": ("把下面这段中文翻译成英文，只输出译文，不要解释：\n\n"
                     + _section("README.cn.md", 7, 600), True),
    "nlp_zh2en_08": ("把下面这段中文翻译成英文，只输出译文，保留引用格式：\n\n"
                     + _section("README.cn.md", 9, 600), True),

    # English -> Chinese
    "nlp_en2zh_01": ("把下面这段英文翻译成简体中文，只输出译文，不要解释：\n\n"
                     + _section("README.md", 0, 600), True),
    "nlp_en2zh_02": ("把下面这段英文翻译成简体中文，只输出译文，保留数字与专有名词：\n\n"
                     + _section("README.md", 1, 600), True),
    "nlp_en2zh_03": ("把下面这段英文翻译成简体中文，只输出译文，不要解释：\n\n"
                     + _section("README.md", 2, 600), True),
    "nlp_en2zh_04": ("把下面这段英文翻译成简体中文，只输出译文，保留列表结构：\n\n"
                     + _section("README.md", 3, 600), True),
    "nlp_en2zh_05": ("把下面这段英文翻译成简体中文，只输出译文，不要解释：\n\n"
                     + _section("README.md", 4, 600), True),
    "nlp_en2zh_06": ("把下面这段英文翻译成简体中文，只输出译文，保留代码与命令：\n\n"
                     + _section("README.md", 5, 600), True),
    "nlp_en2zh_07": ("把下面这段英文翻译成简体中文，只输出译文，不要解释：\n\n"
                     + _section("README.md", 7, 600), True),
    "nlp_en2zh_08": ("把下面这段英文翻译成简体中文，只输出译文，保留引用格式：\n\n"
                     + _section("README.md", 9, 600), True),

    # Chinese document -> Chinese summary
    "nlp_sumzh_01": ("请用 150 字左右总结下面这段中文，保留关键数字与结论：\n\n"
                     + _section("README.cn.md", 1, 1200), True),
    "nlp_sumzh_02": ("请用 150 字左右总结下面这段中文，说明它解决了什么问题：\n\n"
                     + _section("README.cn.md", 2, 1200), True),
    "nlp_sumzh_03": ("请用 150 字左右总结下面这段中文，分三点列出：\n\n"
                     + _section("README.cn.md", 3, 1200), True),
    "nlp_sumzh_04": ("请用 150 字左右总结下面这段中文，按步骤说明：\n\n"
                     + _section("README.cn.md", 5, 1200), True),
    "nlp_sumzh_05": ("请用 150 字左右总结下面这段中文，指出已知问题：\n\n"
                     + _section("README.cn.md", 7, 1200), True),
    "nlp_sumzh_06": ("请用 150 字左右总结下面这段中文，保留许可与声明要点：\n\n"
                     + _section("README.cn.md", 9, 1200), True),

    # English document -> Chinese summary
    "nlp_sumen_01": ("请用 150 字左右的中文总结下面这段英文，保留关键数字与结论：\n\n"
                     + _section("README.md", 1, 1200), True),
    "nlp_sumen_02": ("请用 150 字左右的中文总结下面这段英文，说明它解决了什么问题：\n\n"
                     + _section("README.md", 2, 1200), True),
    "nlp_sumen_03": ("请用 150 字左右的中文总结下面这段英文，分三点列出：\n\n"
                     + _section("README.md", 3, 1200), True),
    "nlp_sumen_04": ("请用 150 字左右的中文总结下面这段英文，按步骤说明：\n\n"
                     + _section("README.md", 5, 1200), True),
    "nlp_sumen_05": ("请用 150 字左右的中文总结下面这段英文，指出已知问题：\n\n"
                     + _section("README.md", 7, 1200), True),
    "nlp_sumen_06": ("请用 150 字左右的中文总结下面这段英文，保留许可与声明要点：\n\n"
                     + _section("README.md", 9, 1200), True),

    # Document organisation
    "nlp_org_01": ("把下面这段会议记录整理成行动项表格，列为：事项、负责人、截止日期、状态。\n\n"
                   "本周例会讨论了搜索服务的缓存改造。张工负责在下周五前完成方案评审，"
                   "李工负责补充压测数据，王工负责确认灰度范围。大家同意先在小流量环境灰度，"
                   "观察一周后再全量。风险是缓存一致性，需要在评审中给出回滚方案。", True),
    "nlp_org_02": ("把下面的项目进展整理成三级大纲，并在每项后标注风险。\n\n"
                   "项目「星河」当前进展：接口开发完成 80%；数据迁移完成 60%，"
                   "遇到字段缺失问题；前端联调尚未开始；测试环境已就绪；"
                   "上线时间暂定下月 15 日。主要风险是数据质量和联调排期。", True),
    "nlp_org_03": ("把下面的需求整理成 JSON，字段包括 id、描述、优先级、验收标准。\n\n"
                   "需求：支持按标签筛选；支持导出 CSV；列表默认按更新时间倒序；"
                   "每页 20 条；管理员可以批量删除；普通用户只能删除自己的数据。", True),
    "nlp_org_04": ("把下面的方案说明整理成对比表格，列为：方案、成本、延迟、扩展性、运维复杂度。\n\n"
                   "方案 A：成本低，延迟 120ms，运维简单，但不支持水平扩展。"
                   "方案 B：成本中等，延迟 80ms，需要额外的消息队列，但可以水平扩展。"
                   "方案 C：成本高，延迟 50ms，运维复杂，支持多活。", True),
    "nlp_org_05": ("把下面的邮件往来整理成决策记录，列为：结论、依据、负责人、后续动作。\n\n"
                   "邮件一：客户希望把交付时间提前两周。"
                   "邮件二：研发表示提前两周需要砍掉报表功能。"
                   "邮件三：客户同意先交付核心功能，报表放到第二阶段。"
                   "邮件四：项目经理确认按核心功能排期，下周一给出新的里程碑。", True),
    "nlp_org_06": ("把下面的书单整理成分类表格，列为：书名、主题、适合的读者。\n\n"
                   "《设计数据密集型应用》讲数据系统；《代码整洁之道》讲工程规范；"
                   "《人月神话》讲项目管理；《深度工作》讲专注方法；《金字塔原理》讲表达结构。", True),
    "nlp_org_07": ("把下面的缺陷报告整理成摘要，包含：严重程度、影响范围、复现步骤、下一步。\n\n"
                   "用户反馈：导出 CSV 时，如果文件名包含空格，浏览器会截断文件名。"
                   "影响 Chrome 和 Edge；Firefox 正常。复现步骤：选择包含空格的文件名，点击导出。"
                   "日志里没有报错。", True),
    "nlp_org_08": ("把下面的行程整理成表格，列为：日期、时间、地点、预算备注。\n\n"
                   "三天行程：第一天上午到杭州，下午西湖；第二天上午灵隐寺，下午龙井村；"
                   "第三天上午西溪湿地，下午返程。预算每天 600 元，住在地铁沿线。", True),

    # Chat, including two multi-turn sessions
    "nlp_chat_01": ("最近工作有点累，有什么放松的方法吗？请给出至少五条具体建议。", True),
    "nlp_chat_02": ("我想系统学习数据可视化，应该按什么顺序学？请给出一个三个月的计划。", True),
    "nlp_chat_03": ("帮我把「少即是多」这个设计原则解释给一个刚入行的同事听。", True),
    "nlp_chat_04": ("周末想在家做一顿简单的饭，推荐三道容易上手的菜，并说明做法。", True),
    "nlp_chat_05": ([
        {"role": "user", "content": "我在整理一份中文技术资料，想先列一个提纲。"},
        {"role": "assistant", "content": "可以。先告诉我资料的主题和大致篇幅，我帮你按「背景—方法—结果—结论」列提纲。"},
        {"role": "user", "content": "主题是本地部署的推理加速，篇幅大约一万字。请直接给我一个四级提纲。"},
    ], True),
    "nlp_chat_06": ([
        {"role": "user", "content": "这段话有点啰嗦，帮我改简洁：由于最近天气状况十分不理想，因此我们决定将原定于本周六举行的户外活动延期至下周六举行。"},
        {"role": "assistant", "content": "可以改成：因天气不佳，本周六的户外活动顺延至下周六。"},
        {"role": "user", "content": "再正式一点，适合写在通知里。"},
    ], True),

    # Long-context NLP tasks
    "nlp_longctx_01": ("请把这份文档整理成三级大纲，并在每节下写一句摘要。",
                       True, ["docs/notes.md", "docs/facts.md"]),
    "nlp_longctx_02": ("请从文档中提取所有 gate 及其结论，整理成表格，列为：gate、条件、结果、状态。",
                       True, ["docs/notes.md"]),
    "nlp_longctx_03": ("请用 300 字总结第 6 节和第 7 节的核心结论。",
                       True, ["docs/notes.md"]),
    "nlp_longctx_04": ("请把第 8 节翻译成简体中文，保留数字与专有名词。",
                       True, ["docs/notes.md"]),
    "nlp_longctx_05": ("请比较 S2a 与 D0 的结论，列出相同点和不同点。",
                       True, ["docs/notes.md"]),
    "nlp_longctx_06": ("请把文档中所有带数字的结论整理成一张表，列为：指标、数值、上下文。",
                       True, ["docs/notes.md", "docs/facts.md"]),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--set", choices=("synthetic", "realistic", "daily", "nlp"),
                    default="synthetic",
                    help="synthetic = six invented workloads; realistic = the actual usage "
                         "pattern (Simplified Chinese prompts, thinking on, mixed-language "
                         "thoughts); daily = the D0 deployment mix across six registers; "
                         "nlp = translation, summarisation, organisation, chat, long context")
    args = ap.parse_args()

    probes = {"synthetic": PROBES, "realistic": REALISTIC, "daily": DAILY,
              "nlp": NLP}[args.set]

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
    for name, spec in probes.items():
        first, thinking = spec[0], spec[1]
        context = spec[2] if len(spec) > 2 else None
        if isinstance(first, list):
            messages = first
        else:
            user = first
            if context:
                parts = [(BASE / rel).read_text(encoding="utf-8") for rel in context]
                user = ("以下是我提供的参考资料，请只根据这些内容回答。\n\n"
                        + "\n\n---\n\n".join(parts)
                        + "\n\n问题：" + user)
            messages = [{"role": "user", "content": user}]
        text = tmpl.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        path = out / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        register = (name.split("_")[1]
                    if name.startswith(("daily_", "nlp_")) else name)
        manifest[name] = {
            "path": str(path),
            "register": register,
            "thinking": thinking,
            "turns": len(messages),
            "chars": len(text),
            "tail": text[-40:],
        }
        print(f"{name:18s} reg={register:8s} thinking={str(thinking):5s} "
              f"turns={len(messages)} chars={len(text):6d} tail={text[-32:]!r}")

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
