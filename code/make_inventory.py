"""Produce a complete inventory of the project folder: every file, its role, and its status.

Writes INVENTORY.md at the project root.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------------------
# role descriptions.  key = path relative to the project root (or a directory prefix)
# ---------------------------------------------------------------------------------------
ROLE = {
    "README.md": ("起点", "环境搭建、实验流程、复现说明。**先读这个**"),
    "literature_survey_DL_channel_estimation.md": (
        "文献", "26 篇先行工作的综述，DOI 经 CrossRef 逐条核验；含对两处常见误引的更正"),

    # ---- paper
    "paper/manuscript.md": (
        "论文源稿", "带结果占位符的模板；正文、公式、参考文献。**这是唯一需要手改的稿子**"),
    "paper/manuscript_final.md": (
        "论文源稿", "占位符已由脚本填入实测数字（自动生成，勿手改）"),
    "paper/manuscript_illustrated.md": (
        "论文源稿", "在上者基础上插入 7 张图并编号（自动生成，勿手改）"),
    "paper/TDF-Net-manuscript.docx": (
        "★ 交付件", "投稿用 Word 稿：216 个原生公式、7 图、5 表、22 条参考文献"),
    "paper/TDF-Net-manuscript.tex": (
        "★ 交付件", "投稿用 LaTeX 稿（Overleaf 可直接编译）"),
    "paper/reference.docx": (
        "工具", "pandoc 转换用的样式模板，产出 docx 时被引用"),
    "paper/RESULTS_NARRATIVE.md": (
        "实验记录", "写作过程中整理的实测结果叙述草稿；数字已被正文取代，存档用"),

    # ---- code: core
    "code/config.py": ("核心代码", "系统与信道参数（256 子载波、15 kHz、3.5 GHz、导频图样、训练超参）"),
    "code/channel.py": ("核心代码", "3GPP TR 38.901 TDL 信道 + 和正弦时变生成器；含 Jakes 自相关自检"),
    "code/data.py": ("核心代码", "训练数据生成：导频去旋转、噪声白化、特征平面"),
    "code/baselines.py": ("核心代码", "传统基线：LS/DFT-denoise/OMP/采样统计 LMMSE + BER 辅助"),
    "code/models.py": ("核心代码", "TDF-Net、ResCNN、ChannelNet 及参数统计"),
    "code/train.py": ("核心代码", "训练主程序，保存 checkpoint 与训练历史"),
    "code/eval.py": ("核心代码", "NMSE/BER 扫描与复杂度测量"),
    "code/eval_ablation.py": ("核心代码", "消融评测：时隙窗 M、延迟分支、LSTM"),
    "code/exp_input_representation.py": ("核心代码", "去旋转实验（固定低速，19.2 dB）"),
    "code/exp_representation_full.py": ("核心代码", "去旋转实验（完整训练分布，11.9 dB）"),

    # ---- code: tooling
    "code/plot.py": ("出图", "生成 8 张论文图（PDF 矢量 + 600 dpi PNG）"),
    "code/make_tables.py": ("出表", "由评测 JSON 生成 LaTeX 表与 RESULTS.md"),
    "code/fill_manuscript.py": ("成稿", "把实测数字填入论文占位符，并给表格编号"),
    "code/insert_figures.py": ("成稿", "把图插到首次引用处并按引用顺序编号"),
    "code/build_paper.py": ("成稿", "Markdown → docx（原生公式）+ .tex"),
    "code/audit_manuscript.py": ("质检", "终稿自动审计：占位符、图表交叉引用、陈旧数值"),
    "code/export_delay.py": ("出图", "导出训练后的延迟域窗（图 5a）"),
    "code/run_all.py": ("工具", "按正确顺序串起评测→出表→出图"),

    # ---- results
    "results/eval_main.json": ("★ 实验数据", "主实验：4 速度 × 7 SNR × 8 估计器 + BER + 复杂度。**论文主表来源**"),
    "results/eval_ablation.json": ("★ 实验数据", "消融实验数据（表 III、图 4）"),
    "results/exp_representation_full.json": ("★ 实验数据", "去旋转实验（完整协议）结果"),
    "results/exp_input_representation.json": ("★ 实验数据", "去旋转实验（固定低速）结果"),
    "results/eval_profiles.json": ("实验数据", "TDL-A / TDL-D 剖面鲁棒性（论文 §VII-B 引用）"),
    "results/delay_window.json": ("实验数据", "训练后的延迟域窗数值（图 5）"),
    "results/RESULTS.md": ("自动汇总", "由脚本汇总的 Markdown 结果表"),
    "results/tables.tex": ("自动汇总", "由脚本生成的 LaTeX 表格代码"),
    "results/eval_quick.json": ("过程数据", "早期快速校验（已被 eval_main 取代，可删）"),
    "results/eval_nmse.json": ("过程数据", "早期单模型速度扫描（已被取代，可删）"),
    "results/eval_baselines.json": ("过程数据", "早期基线校验（已被取代，可删）"),
    "results/eval_M_quick.json": ("过程数据", "早期 M 对照（已被 eval_ablation 取代，可删）"),

    # ---- figures
    "figures/fig_nmse_tdlc_v30.png": ("论文图", "图 1：NMSE vs SNR @30 km/h"),
    "figures/fig_nmse_vs_velocity.png": ("论文图", "图 2：NMSE vs 速度（排序反转）"),
    "figures/fig_training.png": ("论文图", "图 3：训练/验证曲线"),
    "figures/fig_ablation.png": ("论文图", "图 4：M 消融 + 延迟分支消融"),
    "figures/fig_delay_window.png": ("论文图", "图 5：学习到的延迟窗 + 真实抽头剖面"),
    "figures/fig_ber_v100.png": ("论文图", "图 6：BER @100 km/h"),
    "figures/fig_complexity.png": ("论文图", "图 7：参数量与推理耗时"),
    "figures/fig_ber_v30.png": ("备用图", "BER @30 km/h（论文未引用）"),
}

DIR_ROLE = {
    "code": "全部源码与脚本",
    "paper": "论文源稿与交付件",
    "results": "实验产出的 JSON 数据与自动汇总表",
    "figures": "论文图件",
    "checkpoints": "训练好的模型权重（可从零重训复现）",
    ".venv": "Python 虚拟环境（依赖，可重建）",
    "wheelhouse": "离线安装用的 wheel 包缓存（可删）",
    "_dlprobe": "早期依赖探测残留（可删）",
}

CHECKPOINT_ROLE = {
    "tdfnet_M4.pt": "TDF-Net 主模型（M=4，GRU）",
    "tdfnet_M2.pt": "时隙窗 M=2 变体",
    "tdfnet_M1.pt": "时隙窗 M=1 变体（无时序）",
    "tdfnet_M8.pt": "时隙窗 M=8 变体",
    "tdfnet_nodelay.pt": "无延迟域分支变体",
    "tdfnet_lstm.pt": "LSTM 替换 GRU 变体",
    "rescnn_M1.pt": "ResCNN 基线（单时隙、258 K 参数）",
    "rescnn_M4gru.pt": "ResCNN + GRU 基线",
    "channelnet.pt": "ChannelNet（SRCNN 风格）基线",
}

DELETABLE = {
    ".venv", "wheelhouse", "_dlprobe",
    "results/eval_quick.json", "results/eval_nmse.json",
    "results/eval_baselines.json", "results/eval_M_quick.json",
}


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return "%.0f %s" % (n, u)
        n /= 1024.0
    return "%.1f TB" % n


def main():
    rows = []
    collapsed = {}
    total = 0
    SKIP_PREFIXES = (".venv/", "wheelhouse/", "_dlprobe/")
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, ROOT).replace("\\", "/")
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            total += size
            head = rel.split("/")[0]
            if rel.startswith(SKIP_PREFIXES):
                c = collapsed.setdefault(head, [0, 0])
                c[0] += 1
                c[1] += size
                continue
            if rel in ROLE:
                cat, desc = ROLE[rel]
            elif rel.startswith("checkpoints/"):
                cat, desc = "模型权重", CHECKPOINT_ROLE.get(fn, "训练权重")
            elif rel.startswith("figures/") and rel.endswith(".pdf"):
                cat, desc = "论文图", "同名的矢量版本（LaTeX 用 %s）" % fn
            elif rel.startswith("results/history_"):
                cat, desc = "训练记录", "训练历史（loss 曲线原始数据）"
            elif rel.startswith("code/_probe"):
                cat, desc = "开发诊断", "排错过程中写的一次性脚本，保留以佐证诊断过程"
            elif rel.endswith(".pyc") or "__pycache__" in rel:
                cat, desc = "缓存", "Python 字节码缓存"
            else:
                cat, desc = "其他", ""
            dele = (rel in DELETABLE or rel.endswith(".pyc") or "__pycache__" in rel)
            rows.append((rel, cat, desc, size, dele))

    out = ["# 项目文件清单", "",
           "## 分类说明",
           "",
           "- **★ 交付件**：投稿用的成稿，直接可用。",
           "- **论文源稿**：Markdown 稿，`manuscript.md` 是唯一需要手改的；其余由脚本生成。",
           "- **★ 实验数据**：论文中数字的来源，改动它会改变论文结论，请勿手改。",
           "- **核心代码**：仿真、模型、训练、评测。",
           "- **论图**：论文图件，同名 `.pdf` 为矢量版（LaTeX 用），`.png` 为 600 dpi 位图（Word 用）。",
           "",
           "标注 (可删) 的文件是排错过程中的中间产物或可重建的依赖，删除不影响论文复现。",
           "",
           "## 文件清单", "",
           "| 路径 | 分类 | 说明 | 大小 |", "|---|---|---|---|"]
    order = {"★ 交付件": 0, "论文源稿": 1, "★ 实验数据": 2, "核心代码": 3, "论文图": 4}
    for rel, cat, desc, size, dele in sorted(rows, key=lambda r: (order.get(r[1], 9), r[0])):
        mark = " **(可删)**" if dele else ""
        out.append("| `%s` | %s | %s%s | %s |" % (rel, cat, desc, mark, human(size)))

    if collapsed:
        out += ["", "## 已折叠的目录（非项目内容，可整目录删除）", "",
                "| 目录 | 文件数 | 大小 | 说明 |", "|---|---|---|---|"]
        note = {".venv": "Python 虚拟环境；用 README 第 1 节的命令重建",
                "wheelhouse": "离线安装用的 wheel 包缓存；联网环境不需要",
                "_dlprobe": "早期 pip 依赖探测残留"}
        for d, (n, sz) in sorted(collapsed.items()):
            out.append("| `%s/` | %d | %s | %s |" % (d, n, human(sz), note.get(d, "")))

    with open(os.path.join(ROOT, "INVENTORY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    print("files: %d   total: %s" % (len(rows), human(total)))
    keep = sum(r[3] for r in rows if not r[4])
    drop = sum(r[3] for r in rows if r[4])
    print("needed : %s" % human(keep))
    print("deletable: %s" % human(drop))
    print("wrote INVENTORY.md")


if __name__ == "__main__":
    main()
