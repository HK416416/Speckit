# WSL2 完整安装与配置指南

> 本项目运行环境：WSL2 (Ubuntu 22.04) + CUDA + vLLM + PyTorch

---

## 目录

| 章节 | 内容 |
|------|------|
| [第一章](#第一章-为什么用-wsl2) | 为什么用 WSL2 |
| [第二章](#第二章-wsl2-安装) | WSL2 安装（一步完成） |
| [第三章](#第三章-cuda-与-gpu-配置) | CUDA 与 GPU 配置 |
| [第四章](#第四章-项目环境搭建) | 项目环境搭建（Conda + 依赖） |
| [第五章](#第五章-跨系统文件管理) | 跨系统文件管理（Windows ↔ WSL2） |
| [第六章](#第六章-vscode-集成) | VS Code 集成（WSL2 远程开发） |
| [第七章](#第七章-故障排查) | 故障排查 |

---

## 第一章 为什么用 WSL2

### 问题

vLLM 官方**不支持 Windows 原生运行**——`ModuleNotFoundError: No module named 'resource'`，因为 `resource` 是 Unix 专用模块。

### 可选方案对比

| 方案 | vLLM | GPU CUDA | 部署难度 | 性能 | 推荐 |
|------|:---:|:---:|:---:|------|:---:|
| **WSL2** | ⚠️ vLLM 0.25+ 与 WSL2 pinned memory 不兼容 | ✅ 原生 | ⭐⭐ | 接近原生 | ✅ (用 hf_server.py) |
| Windows (hf_server.py) | ❌ 替代方案 | ✅ | ⭐ | 良好 | ⭐ 备选 |
| 传统 VM (VirtualBox) | ✅ | ❌ | ⭐⭐⭐ | 极差 | ❌ |
| 传统 VM (GPU Passthrough) | ✅ | ⚠️ 笔记本不可行 | ⭐⭐⭐⭐⭐ | — | ❌ |
| 双系统 Linux | ✅ | ✅ | ⭐⭐⭐⭐ | 原生 | ⭐⭐ |

### WSL2 工作原理

WSL2 不是虚拟机——它是 Windows 内置的轻量 Linux 内核，运行在 Hyper-V 上：

```
┌─────────────────────────────────────────────┐
│                  Windows 11                  │
│  ┌─────────────────┐  ┌───────────────────┐ │
│  │   Windows 应用   │  │     WSL2 VM        │ │
│  │   VS Code       │  │  ┌───────────────┐ │ │
│  │   PowerShell    │  │  │  Ubuntu 22.04  │ │ │
│  │   文件管理器     │  │  │  Conda         │ │ │
│  └────────┬────────┘  │  │  PyTorch+CUDA  │ │ │
│           │           │  │  vLLM          │ │ │
│           │  共享     │  │  myDraft_Verify│ │ │
│           └───────────┼──┤  ...           │ │ │
│              /mnt/c/  │  └───────────────┘ │ │
│                       └───────────────────┘ │
│              NVIDIA Driver (WSL2-aware)      │
└─────────────────────────────────────────────┘
```

**关键优势**：
- **与 Windows 共享文件系统**：`/mnt/c/` 直接访问 C 盘，无需复制文件
- **原生 CUDA**：WSL2 GPU 驱动与 Windows 共享，性能几乎无损
- **VS Code WSL 远程开发**：在 Windows VS Code 中编辑 WSL2 中的代码，无缝体验

---

## 第二章 WSL2 安装

### 前置条件

- Windows 10 版本 2004+ 或 Windows 11
- 管理员权限
- 已安装 NVIDIA 驱动（≥ 510.x，RTX 4050 驱动 595.97 满足要求）

### 一键安装

以**管理员身份**打开 PowerShell，运行以下命令：

```powershell
# 安装 WSL2 + Ubuntu 最新 LTS（一步完成）
wsl --install -d Ubuntu

# 如果需要指定版本（可选）:
# wsl --install -d Ubuntu-22.04
# wsl --install -d Ubuntu-24.04
# wsl --install -d Ubuntu-26.04
```

> **Ubuntu 版本说明**：22.04/24.04/26.04 LTS 均兼容 CUDA + vLLM + PyTorch，无需追求特定版本。当前已安装显示 26.04 完全可用。

安装过程中会提示创建 Linux 用户名和密码。完成后自动进入 Ubuntu 终端。

### 验证安装

在 PowerShell 中：

```powershell
# 查看已安装的 WSL 发行版
wsl -l -v
# 预期输出:
#   NAME            STATE           VERSION
# * Ubuntu           Running         2

# 查看 Ubuntu 版本
wsl lsb_release -a
# 22.04/24.04/26.04 均可

# 检查 GPU
wsl nvidia-smi
# 预期: NVIDIA GeForce RTX 4050 Laptop GPU
```

### 常用 WSL 命令速查

| 命令 | 说明 |
|------|------|
| `wsl` | 进入默认 WSL 发行版 |
| `wsl -l -v` | 列出所有发行版及状态 |
| `wsl --shutdown` | 关闭所有 WSL2 实例 |
| `wsl --terminate Ubuntu-22.04` | 关闭特定发行版 |
| `wsl --set-version Ubuntu-22.04 2` | 确认 WSL 版本 2 |
| `exit` | 退出 WSL 终端 |

---

## 第三章 CUDA 与 GPU 配置

### 前置条件：安装 C 编译器

vLLM torch.compile 需要 gcc。WSL2 默认未安装。



### 重要说明

> **不需要安装 `cuda-toolkit`！** PyTorch/vLLM 的 GPU wheel 自带 CUDA 运行时。WSL2 内核对 NVIDIA 驱动的支持已内建在 Windows 驱动中。

### 验证 GPU 可用

```bash
# 检查 NVIDIA 驱动（WSL2 共享 Windows 驱动）
nvidia-smi
# 预期: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB
#       Driver Version: 595.xx, CUDA Version: 12.x 或 13.x
```

### WSL2 显存管理

WSL2 默认独占全部显存。如果需要限制：

```powershell
# 在 Windows 用户目录创建 .wslconfig
notepad $env:USERPROFILE\.wslconfig
```

内容：

```ini
[wsl2]
memory=8GB       # 限制内存（建议 8GB）
processors=4     # 限制 CPU 核心数
swap=4GB         # swap 大小
```

修改后需重启 WSL2：

```powershell
wsl --shutdown
```

---

## 第四章 项目环境搭建

### 进入项目目录

WSL2 可以访问 Windows 文件系统：

```bash
# Windows E 盘在 WSL2 中的路径
cd /mnt/e/Speckit

# 确认文件存在
ls -la
# 应看到: README.md, myDraft_Verify/, vllm_benchmark/, ...
```

### 安装 Miniconda

```bash
# 下载并安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3

# 初始化 conda
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 创建项目环境

```bash
# 创建环境
conda create -n spec_dec python=3.10 -y
conda activate spec_dec

# 安装 PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 vLLM（WSL2 原生支持！）
pip install vllm

# 安装项目其余依赖
pip install -r myDraft_Verify/requirements.txt
pip install matplotlib>=3.7.0 mcp httpx

# 验证
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
```

### 模型下载（可选，首次运行自动下载）

如果需要提前下载模型：

```bash
# HuggingFace 模型保存在 ~/.cache/huggingface/hub/
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
"
```

---

## 第五章 跨系统文件管理

### 路径映射

| Windows 路径 | WSL2 路径 |
|-------------|----------|
| `C:\` | `/mnt/c/` |
| `E:\Speckit\` | `/mnt/e/Speckit/` |
| `C:\Users\13777\` | `/mnt/c/Users/13777/` |

### 推荐工作流

```bash
# 1. 在 Windows VS Code 中编辑代码
# 2. 在 WSL2 终端中运行（所有命令在 /mnt/e/Speckit 下执行）
# 3. 结果文件（results/*.csv）在 WSL2 和 Windows 中都能访问

# 示例：在 WSL2 中运行实验
cd /mnt/e/Speckit
conda activate spec_dec

# vLLM 实验
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/baseline.json

# MyDraft_Verify
python myDraft_Verify/run.py --quick
```

### 注意事项

- **跨文件系统 I/O 性能**：`/mnt/e/` 的读写速度比原生 ext4 慢约 20-30%。对于大模型推理（瓶颈在 GPU 显存带宽，不在磁盘 I/O），影响可忽略。
- **不要从 WSL2 操作 Windows 系统文件**：可以读，不要写（可能损坏 Windows 文件）。
- **项目代码放在 Windows 侧**（`E:\Speckit\`），WSL2 通过 `/mnt/e/Speckit/` 访问——这样一份代码两边都能用。

---

## 第六章 VS Code 集成

### 从 Windows VS Code 连接 WSL2

1. 安装 VS Code 扩展：**WSL** (Microsoft 官方)
2. 打开 VS Code，按 `Ctrl+Shift+P`
3. 输入 `WSL: Connect to WSL` → 选择 `Ubuntu-22.04`
4. VS Code 窗口左下角显示 `WSL: Ubuntu-22.04`
5. 打开文件夹 → `/mnt/e/Speckit`

```
VS Code (Windows)                 WSL2 (Ubuntu)
┌──────────────────┐    SSH     ┌──────────────────┐
│  UI / Editor     │◄──────────►│  终端 / 运行环境   │
│  文件浏览         │            │  Python / Conda   │
│  代码编辑         │            │  vLLM / PyTorch   │
│  图形界面         │            │  CUDA / GPU       │
└──────────────────┘            └──────────────────┘
```

### Windows Terminal 配置

Windows Terminal 自动检测 WSL2，可以在标签页中同时打开 PowerShell 和 Ubuntu：

- 新标签页下拉菜单中选择 `Ubuntu-22.04`
- 或快捷键 `Ctrl+Shift+[数字]` 切换

---

## 第七章 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `wsl --install` 失败 | 虚拟化未开启 | BIOS 中启用 Intel VT-x / AMD-V |
| `nvidia-smi` 无输出 | WSL2 GPU 驱动未安装 | Windows 端更新 NVIDIA 驱动到 ≥ 510.x |
| `torch.cuda.is_available()` 返回 False | PyTorch 是 CPU 版本 | 重装: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `Error: cuInit: CUDA_ERROR_UNKNOWN` | WSL2 内核过旧 | `wsl --update` |
| WSL2 占用过高内存 | 默认限制宽松 | 创建 `%USERPROFILE%\.wslconfig` 限制 |
| `conda: command not found` | conda 未初始化 | `~/miniconda3/bin/conda init bash && source ~/.bashrc` |
| `/mnt/e/` 访问不到 | 磁盘未挂载 | `sudo mount -t drvfs E: /mnt/e` |
| vLLM OOM (Out of Memory) | 6GB 显存不足 | 降低 `--gpu-memory-utilization 0.85` 或换更小模型 |
