# 论文研读模块 (paper_notes/)

> 方向一：投机推理方向 —— 4篇指定论文的精读笔记

---

## 文件结构

```
paper_notes/
├── MANUAL.md                ← 本手册：模块使用说明
├── template.md              ← 论文阅读笔记通用模板
├── paper1_leviathan_speculative_decoding.md
├── paper2_specinfer_tree_verifier.md
├── paper3_eagle2_dynamic_draft_trees.md
├── paper4_dflash_block_diffusion.md
└── summary.md               ← 4篇论文对比总结
```

---

## 使用方式

### 1. 阅读顺序建议

```
第1篇: Leviathan (ICML 2023) — 投机推理奠基 ★必读最先
  ↓
第2篇: SpecInfer (2024) — 树形 draft + token tree verifier
  ↓
第3篇: EAGLE-2 (2024) — feature-level 草稿 + 动态 draft tree
  ↓
第4篇: DFlash (2024) — 块并行解码替代方案
```

### 2. 笔记填写方法

每篇论文笔记文件已预填了论文基本信息和结构化框架，需要补充的部分标注了 `[TODO: ...]`。

**操作步骤**：
1. 打开对应论文笔记文件（如 `paper1_leviathan_speculative_decoding.md`）
2. 找到所有 `[TODO: ...]` 标记
3. 阅读论文对应章节，用自己的话填写
4. 保存后即完成该篇笔记

### 3. 论文获取

| 论文 | arXiv 链接 |
|------|-----------|
| Leviathan et al. | https://arxiv.org/abs/2302.01318 |
| SpecInfer | https://arxiv.org/abs/2305.09781 |
| EAGLE-2 | https://arxiv.org/abs/2406.16858 |
| DFlash | 搜索 "DFlash Block Diffusion Speculative Decoding" |

---

## 笔记质量标准

每篇笔记应达到以下标准：

- [x] 论文基本信息完整（作者、会议/年份、链接）
- [ ] 核心问题用一句话概括清楚
- [ ] 核心方法用 3-5 句话解释关键思路（不要复制摘要）
- [ ] 至少 3 个关键实验数据/发现
- [ ] 与方向一（端侧投机推理）的关联分析
- [ ] 个人思考：哪些可以借鉴到自己的实验中
- [ ] 至少 1 个疑问点（后续讨论用）
