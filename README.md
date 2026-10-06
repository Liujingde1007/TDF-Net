# 观测模型比架构更重要：学习型 OFDM 信道估计

本项目包含论文 *"The Observation Model Dominates the Architecture in Learned OFDM Channel
Estimation"*（投稿 IEEE Communications Letters）的完整代码、实验数据、图件、模型权重与成稿。

**论文中每一个数字都由 `code/` 中的脚本从实验产生，没有任何一个是手写的。**

---

## 0. 先看这里：成稿在哪

| 文件 | 说明 |
|---|---|
| **`paper/TDF-Net-ieeeCL.tex`** | **投稿源文件**：IEEEtran 双栏，Overleaf 或本地 LaTeX 可直接编译 |
| `paper/TDF-Net-ieeeCL.docx` | 同一内容的 Word 版（便于通读、批注、给导师看） |
| `paper/TDF-Net-manuscript.docx` / `.tex` | 完整版（IEEE Access 路线备用，篇幅更长） |
| `paper/manuscript_CL.md` | CL 版的 Markdown 文本（生成 `manuscript_CL_final.md`，**不是投稿源**） |
| `paper/manuscript.md` | 完整版的**源稿模板**（数字由脚本填入） |
| `CLEANUP_LOG.md` | 清理记录：删了什么、为什么、保留了什么 |
| `INVENTORY.md` | 逐文件说明清单 |

### 投稿用 LaTeX 的编译

`paper/TDF-Net-ieeeCL.tex` 需要 `IEEEtran` 文档类，图片从 `../figures/` 读取（显式引用矢量
PDF）。两种方式：

```powershell
# 方式 A：本地（需先安装 MiKTeX 或 TeX Live）
cd paper
pdflatex TDF-Net-ieeeCL
pdflatex TDF-Net-ieeeCL        # 跑两次以解析交叉引用
```

方式 B：把 `paper/TDF-Net-ieeeCL.tex` 与整个 `figures/` 目录上传到 Overleaf，编译器选
pdfLaTeX，`IEEEtran` 在 TeX Live 中自带，无需额外配置。

表格内容由脚本从实验 JSON 生成，不要手改：

```powershell
.\.venv\Scripts\python.exe code\make_latex_tables.py   # 从 results/*.json 生成三张表
.\.venv\Scripts\python.exe code\check_tex.py           # 结构验证（引用、图片、篇幅、数字）
```

### Markdown/Word 路线的生成链

```
paper/manuscript_CL.md  --build_cl.py-->  manuscript_CL_final.md  --(pandoc)-->  TDF-Net-ieeeCL.docx
paper/manuscript.md  --fill_manuscript.py--> manuscript_final.md --insert_figures.py--> manuscript_illustrated.md --build_paper.py--> TDF-Net-manuscript.docx/.tex
```

`build_cl.py` **不会**覆盖 `TDF-Net-ieeeCL.tex`——投稿源是手工维护的，pandoc 转换无法表达
IEEEtran 类、浮动体位置、附录与作者简介。

---

## 1. 环境搭建

实验全部在 CPU 上运行，无需 GPU。Python 3.11 + `numpy` / `scipy` / `matplotlib` / `torch`
（`pypandoc-binary` 仅在重建 Word 稿时需要）。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r release\requirements.txt
```

若环境无法访问软件源，可先用系统 pip 下载 wheel 再离线安装：

```powershell
python -m pip download --no-cache-dir --only-binary=:all: --dest wheelhouse numpy scipy matplotlib torch
.\.venv\Scripts\python.exe -m pip install --no-index --find-links wheelhouse numpy scipy matplotlib torch
```

自检（会打印系统参数、验证 Jakes 自相关与模型参数量）：

```powershell
.\.venv\Scripts\python.exe code\config.py
.\.venv\Scripts\python.exe code\channel.py      # 和正弦生成器 vs Jakes 理论
.\.venv\Scripts\python.exe code\models.py       # 各变体参数量与输出形状
```

---

## 2. 完整实验流程

```powershell
# ---- 1. 主模型与基线（每个约 25-45 分钟，16 线程 CPU）
.\.venv\Scripts\python.exe code\train.py --model tdfnet     --seq-len 4 --epochs 30 --n-train 3000 --n-val 400 --tag tdfnet_M4
.\.venv\Scripts\python.exe code\train.py --model rescnn     --seq-len 1 --epochs 30 --n-train 3000 --n-val 400 --tag rescnn_M1
.\.venv\Scripts\python.exe code\train.py --model rescnn     --seq-len 4 --epochs 30 --n-train 3000 --n-val 400 --with-rnn --tag rescnn_M4gru
.\.venv\Scripts\python.exe code\train.py --model channelnet              --epochs 30 --n-train 3000 --n-val 400 --tag channelnet

# ---- 2. 时隙窗与时序消融
.\.venv\Scripts\python.exe code\train.py --model tdfnet --seq-len 1 --epochs 30 --n-train 3000 --n-val 400 --tag tdfnet_M1
.\.venv\Scripts\python.exe code\train.py --model tdfnet --seq-len 2 --epochs 30 --n-train 3000 --n-val 400 --tag tdfnet_M2
.\.venv\Scripts\python.exe code\train.py --model tdfnet --seq-len 8 --epochs 30 --n-train 3000 --n-val 400 --tag tdfnet_M8
.\.venv\Scripts\python.exe code\train.py --model tdfnet --seq-len 4 --epochs 30 --n-train 3000 --n-val 400 --no-delay --tag tdfnet_nodelay
.\.venv\Scripts\python.exe code\train.py --model tdfnet --seq-len 4 --epochs 30 --n-train 3000 --n-val 400 --rnn lstm --tag tdfnet_lstm

# ---- 3. 延迟域先验强度扫描（论文第 IV-C 节）
foreach ($L in 1,2,4,8) {
  .\.venv\Scripts\python.exe code\train.py --model tdfnet --seq-len 4 --epochs 30 --n-train 3000 --n-val 400 --n-taps $L --tag "tdfnet_Ld$L"
}

# ---- 4. 输出域约束变体（论文第 IV-D 节，负结果）
foreach ($L in 2,4) {
  .\.venv\Scripts\python.exe code\train.py --model tdfnet --seq-len 4 --epochs 30 --n-train 3000 --n-val 400 --constraint output --n-taps $L --tag "out_Ld$L"
}

# ---- 5. 观测模型实验（论文第 IV-A 节，核心发现）
.\.venv\Scripts\python.exe code\exp_representation_full.py    # 完整训练分布 → 11.9 dB
.\.venv\Scripts\python.exe code\exp_input_representation.py   # 固定低速 → 19.2 dB

# ---- 6. 评测与配对显著性检验
.\.venv\Scripts\python.exe code\eval.py --tags tdfnet_M4 rescnn_M1 rescnn_M4gru channelnet `
    --names TDF-Net ResCNN ResCNN+GRU ChannelNet `
    --snr -5 0 5 10 15 20 25 --vel 3 30 100 200 --profiles TDL-C `
    --n-real 120 --ber-vel 30 100 --out eval_main
.\.venv\Scripts\python.exe code\eval_ablation.py --n-real 100
.\.venv\Scripts\python.exe code\paired_test.py --tags tdfnet_M4 tdfnet_Ld8 tdfnet_Ld4 tdfnet_Ld2 tdfnet_Ld1 `
    --names "Ld=16" "Ld=8" "Ld=4" "Ld=2" "Ld=1" --n-real 250 --out paired_Ld_full
.\.venv\Scripts\python.exe code\paired_test.py --tags tdfnet_M4 tdfnet_Ld4 tdfnet_Ld2 out_Ld2 out_Ld4 `
    --names "feat-Ld16" "feat-Ld4" "feat-Ld2" "out-Ld2" "out-Ld4" --n-real 250 --out paired_output
.\.venv\Scripts\python.exe code\paired_test.py --tags tdfnet_M2 tdfnet_M8 channelnet `
    --names "M=2" "M=8" "ChannelNet" --n-real 250 --out paired_temporal

# ---- 7. 出图、出表、成稿、审计
.\.venv\Scripts\python.exe code\plot.py --in eval_main --ablation eval_ablation
.\.venv\Scripts\python.exe code\make_paired_figures.py
.\.venv\Scripts\python.exe code\build_cl.py
.\.venv\Scripts\python.exe code\audit_cl.py          # 终稿审计，必须输出 PASS
```

`code/run_all.py` 按正确顺序串起第 6–7 步。

---

## 3. 文件职责

| 路径 | 内容 |
|---|---|
| `code/config.py` | 系统、信道、训练、评测参数 |
| `code/channel.py` | 3GPP TR 38.901 TDL 模型 + 和正弦时变生成器 |
| `code/data.py` | 特征生成（**导频去旋转**、噪声白化） |
| `code/baselines.py` | LS / DFT-denoise / OMP / 采样统计 LMMSE |
| `code/models.py` | TDF-Net（特征级与输出域约束）、ResCNN、ChannelNet |
| `code/train.py` | 训练主程序（`--n-taps` 控先验强度，`--constraint` 选约束方式） |
| `code/eval.py`, `eval_ablation.py` | NMSE/BER 扫描、消融评测、复杂度测量 |
| `code/paired_test.py` | **配对显著性检验**（论文所有变体对比的证据来源） |
| `code/exp_*.py` | 观测模型实验（论文最大发现） |
| `code/plot.py`, `make_paired_figures.py`, `make_tables.py` | 出图与出表 |
| `code/fill_manuscript.py` | 把实测数字填入完整版占位符 |
| `code/insert_figures.py` | 插图并编号 |
| `code/build_cl.py`, `build_paper.py` | 生成 docx（原生公式）与 tex |
| `code/audit_cl.py`, `audit_manuscript.py` | 终稿审计（数字 vs JSON、图表引用、篇幅） |
| `code/cleanup.py` | 清理无用文件，**默认拒删正在运行的 Python 环境** |
| `code/probe_*.py`, `_probe_channel/consistency/tune.py` | 论文引用的完整性验证探针 |
| `results/` | 评测 JSON、配对检验结果、训练历史、自动表格 |
| `figures/` | 论文图件（矢量 PDF + 600 dpi PNG） |
| `checkpoints/` | 论文引用的 15 个模型权重 |
| `release/` | 可发布包（含 LICENSE、requirements.txt、哈希校验） |
| `literature_survey_DL_channel_estimation.md` | 文献综述，DOI 经 CrossRef 核验 |

---

## 4. 可复现性说明

* 所有随机性来自显式种子（`TRAIN.seed` 及各评测脚本的固定种子）；信道生成、速度/SNR 抽样、
  测试时隙均可复现。
* 每个 checkpoint 保存参数量、最佳验证损失、训练轮数与完整参数，评测脚本从 checkpoint
  重建结构而非硬编码配置。
* 信道生成器经回归验证（`code/channel.py` 自检：E|H|²、离散抽头数、Jakes 自相关）；
  LS 观测经理论相关性验证（`code/_probe_consistency.py`）。
* `release/CHECKSUMS.sha256` 记录发布包全部文件的 SHA-256，`make_release.py --verify` 可自检。
* 权重文件过大未打包，但 `release/checkpoints.sha256` 记录了它们的哈希，可校验重现结果。

---

## 5. 硬件与耗时

全部结果在单台 **纯 CPU** 机器上产生（16 硬件线程）。训练一个 TDF-Net 模型（3000 序列、
30 epoch）约 25–45 分钟；含全部消融与配对检验的完整流程约 6–8 小时。**未使用 GPU**，
这本身也是可复现性的一部分。
