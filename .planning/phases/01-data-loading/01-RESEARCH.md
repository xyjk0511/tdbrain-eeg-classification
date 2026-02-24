# Phase 1: Data Loading - Research

**Researched:** 2026-02-23
**Domain:** TDBRAIN BIDS dataset + MNE-Python BrainVision I/O
**Confidence:** HIGH

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | 从 participants.tsv 筛出 MDD（320例）和 ADHD（200例）受试者列表 | `indication` 列直接含 "MDD"/"ADHD"；实测 MDD=320, ADHD=200 |
| DATA-02 | 加载对应的静息态 EEG 文件（.vhdr 格式，BIDS 结构） | `mne.io.read_raw_brainvision()` 实测成功：33ch/500Hz/120s |
| DATA-03 | 支持 EC/EO 两种条件，可配置选择 | BIDS 文件名含 `task-restEC` / `task-restEO`；participants.tsv 有 EC/EO 可用性列 |
</phase_requirements>

## Summary

TDBRAIN 数据集位于 `D:\eeg\TDBRAIN-dataset\`，BIDS 1.6.0 格式，N=1274。participants.tsv 的 `indication` 列直接存储诊断标签，MDD=320、ADHD=200，与需求完全吻合。每个受试者有 `ses-1/eeg/` 目录，包含 EC 和 EO 两个条件的 .vhdr/.eeg/.vmrk 三件套，文件名模式为 `{sub}_ses-1_task-rest{EC|EO}_eeg.vhdr`。

MNE 1.11.0 和 pandas 3.0.1 均已安装，Phase 1 无需安装任何新依赖。`mne.io.read_raw_brainvision()` 实测加载成功。

**Primary recommendation:** pandas 读 participants.tsv 按 `indication` 列筛选；`mne.io.read_raw_brainvision()` 按 BIDS 路径模式加载；condition 用字符串参数 `"EC"/"EO"` 控制文件名拼接。

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| mne | 1.11.0 | 读取 .vhdr，返回 Raw 对象 | 已安装；官方支持 BrainVision 格式 |
| pandas | 3.0.1 | 读取 participants.tsv，筛选诊断 | 已安装；TSV 表格数据标准工具 |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib | stdlib | 构建 BIDS 路径 | 跨平台路径拼接，避免 Windows 路径问题 |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pandas 直接读 TSV | mne-bids BIDSPath (0.18.0 已安装) | mne-bids 对本阶段过重；路径模式固定时 pandas 更简单 |

**Installation:** 无需安装新依赖。

## Architecture Patterns

### Recommended Project Structure

```
tdbrain/
├── config.py          # 数据集路径、条件参数
├── data_loader.py     # load_subjects(), load_raw()
└── main.py            # 验证入口脚本
```

### Pattern 1: 配置集中管理

```python
# config.py
from pathlib import Path

DATASET_ROOT = Path(r"D:\eeg\TDBRAIN-dataset")
PARTICIPANTS_TSV = DATASET_ROOT / "participants.tsv"
CONDITION = "EC"  # "EC" or "EO"
```

### Pattern 2: 按 indication 筛选受试者

```python
# Source: 实测 participants.tsv 结构
import pandas as pd

def load_subjects(tsv_path, condition: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    mask = df["indication"].isin(["MDD", "ADHD"]) & (df[condition] == 1)
    return df[mask][["participants_ID", "indication"]].reset_index(drop=True)
```

### Pattern 3: BIDS 路径构建 + Raw 加载

```python
# Source: 实测 TDBRAIN 目录结构 + MNE 1.11.0
import mne
from pathlib import Path

def load_raw(dataset_root: Path, subject_id: str, condition: str) -> mne.io.Raw:
    vhdr = (dataset_root / subject_id / "ses-1" / "eeg"
            / f"{subject_id}_ses-1_task-rest{condition}_eeg.vhdr")
    return mne.io.read_raw_brainvision(str(vhdr), preload=False, verbose=False)
```

### Anti-Patterns to Avoid

- **preload=True 批量加载:** 520 个受试者全部 preload 会耗尽内存（每个约 30MB）；保持 preload=False
- **用 formal_status 列筛选:** formal_status 有 816 条 "UNKNOWN"，应用 `indication` 列

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 读取 BrainVision 格式 | 自己解析 .vhdr/.eeg 二进制 | `mne.io.read_raw_brainvision` | BrainVision 有多种变体；MNE 处理所有边缘情况 |
| TSV 解析 | 手写解析器 | `pandas.read_csv(sep="\t")` | 处理引号、编码、缺失值等边缘情况 |

## Common Pitfalls

### Pitfall 1: 用 formal_status 而非 indication 筛选

**What goes wrong:** formal_status 有 816 条 "UNKNOWN"，筛出数量远少于预期
**Why it happens:** 两列都像诊断列
**How to avoid:** 始终用 `indication` 列；实测 MDD=320, ADHD=200
**Warning signs:** 筛出数量远小于 320/200

### Pitfall 2: Windows 路径传给 MNE

**What goes wrong:** `/d/eeg/...` Unix 风格路径在 Windows Python 中 FileNotFoundError
**Why it happens:** MNE 用 `open()` 打开文件，需要 Windows 原生路径
**How to avoid:** 用 `pathlib.Path` 构建路径，传 `str(path)` 给 MNE
**Warning signs:** FileNotFoundError 但文件确实存在

### Pitfall 3: 少数受试者缺少 EO 文件

**What goes wrong:** 1 个受试者 EO=0，加载时 FileNotFoundError
**Why it happens:** 数据采集时该受试者未完成 EO 条件
**How to avoid:** 筛选时加 `df[condition] == 1` 过滤条件
**Warning signs:** 加载 EO 条件时出现 FileNotFoundError

### Pitfall 4: 非 EEG 通道混入

**What goes wrong:** 33 通道中有 7 个非 EEG 通道（VPVA, VNVB, HPHL, HNHR, Erbs, OrbOcc, Mass）
**Why it happens:** BrainVision 文件包含 EOG/misc 通道
**How to avoid:** Phase 1 只加载不处理；Phase 2 预处理时 pick EEG 通道
**Warning signs:** 通道数为 33 而非 26

## Code Examples

### 完整验证脚本

```python
# Source: 实测 TDBRAIN 数据结构 + MNE 1.11.0
import pandas as pd
import mne
from pathlib import Path

DATASET_ROOT = Path(r"D:\eeg\TDBRAIN-dataset")
CONDITION = "EC"  # "EC" or "EO"

def load_subjects(condition: str) -> pd.DataFrame:
    df = pd.read_csv(DATASET_ROOT / "participants.tsv", sep="\t")
    mask = df["indication"].isin(["MDD", "ADHD"]) & (df[condition] == 1)
    return df[mask][["participants_ID", "indication"]].reset_index(drop=True)

def load_raw(subject_id: str, condition: str) -> mne.io.Raw:
    vhdr = (DATASET_ROOT / subject_id / "ses-1" / "eeg"
            / f"{subject_id}_ses-1_task-rest{condition}_eeg.vhdr")
    return mne.io.read_raw_brainvision(str(vhdr), preload=False, verbose=False)

if __name__ == "__main__":
    subjects = load_subjects(CONDITION)
    print(f"MDD: {(subjects['indication']=='MDD').sum()}")
    print(f"ADHD: {(subjects['indication']=='ADHD').sum()}")
    raw = load_raw(subjects.iloc[0]["participants_ID"], CONDITION)
    print(f"Loaded: {raw.info['nchan']} ch, {raw.info['sfreq']} Hz, {raw.times[-1]:.1f}s")
```

## Open Questions

1. **多 session 受试者**
   - What we know: participants.tsv 有 `nrSessions` 列，部分受试者有多次 session
   - What's unclear: 是否需要处理 ses-2, ses-3
   - Recommendation: Phase 1 只处理 ses-1（所有受试者都有）

## Sources

### Primary (HIGH confidence)

- 实测 `D:\eeg\TDBRAIN-dataset\participants.tsv` — indication 列分布、EC/EO 列含义、MDD=320/ADHD=200
- 实测 MNE 1.11.0 `read_raw_brainvision` — 成功加载 33ch/500Hz/120s
- 实测 BIDS 目录结构 — `{sub}/ses-1/eeg/{sub}_ses-1_task-rest{EC|EO}_eeg.vhdr`
- `dataset_description.json` — BIDS 1.6.0, N=1274

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — 实测验证，依赖已安装
- Architecture: HIGH — 基于实际数据结构，路径模式已验证
- Pitfalls: HIGH — 基于真实数据探查（formal_status 问题、EO=0 问题均来自实际数据）

**Research date:** 2026-02-23
**Valid until:** 2026-03-23
