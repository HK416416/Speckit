"""
prompts.py — 测试 Prompt 集
============================
覆盖短/中/长三种长度，用于系统性 benchmark。
"""

TEST_PROMPTS = [
    # === 短 prompt (10-50 tokens) ===
    ("short_1", "请用一句话介绍人工智能。"),
    ("short_2", "什么是深度学习？请简要回答。"),
    ("short_3", "请解释机器学习的监督学习和无监督学习。"),
    ("short_4", "Python 中的列表和元组有什么区别？"),

    # === 中 prompt (50-150 tokens) ===
    ("medium_1", "请详细说明 Transformer 模型中的自注意力机制是如何工作的，以及它相比 RNN 的优势。"),
    ("medium_2", "请比较 CPU 和 GPU 在深度学习推理中的优劣。"),
    ("medium_3", "请解释什么是梯度下降算法，以及 SGD 与批量梯度下降的区别。"),
    ("medium_4", "请描述云计算和边缘计算的主要区别，并给出各自适合的应用场景。"),

    # === 长 prompt (150+ tokens) ===
    ("long_1", "假设你是一名计算机科学家，请向非技术人员解释大语言模型的工作原理，包括训练阶段和推理阶段。请用通俗易懂的语言。"),
    ("long_2", "请从性能、成本和部署难度三个维度，对比分析云计算和边缘计算在 AI 推理场景下的适用性。"),
]

def get_prompts_by_category(category: str = "all") -> list:
    """按类别获取 prompts"""
    if category == "all":
        return list(TEST_PROMPTS)
    return [(pid, text) for pid, text in TEST_PROMPTS if pid.startswith(category)]
