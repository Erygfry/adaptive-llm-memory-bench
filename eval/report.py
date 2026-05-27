"""Таблицы + графики из метрик. matplotlib в Agg-режиме (без дисплея)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


QA_TYPE_ORDER = ["recall", "multi_fact", "abstention", "update", "temporal", "gist", "synthesis"]
MODE_ORDER = ["NO_MEMORY", "FACTS_ONLY", "FACTS_PLUS_SUMMARY", "FULL_DIALOGUE"]


def write_report(qa_df: pd.DataFrame, snap_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not qa_df.empty:
        _plot_qa_by_type(qa_df, out_dir)
        _plot_ablation(qa_df, out_dir)
    if not snap_df.empty:
        _plot_db_diagnostics(snap_df, out_dir)
    _write_summary_md(qa_df, snap_df, out_dir)
    print(f"[report] графики + summary.md → {out_dir}")


def _plot_ablation(qa_df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar: pass rate по qa_type, серии = режимы. Главный график."""
    pivot = qa_df.pivot_table(index="qa_type", columns="mode",
                              values="passed", aggfunc="mean")
    # упорядочить оси
    rows = [t for t in QA_TYPE_ORDER if t in pivot.index]
    cols = [m for m in MODE_ORDER if m in pivot.columns]
    pivot = pivot.reindex(index=rows, columns=cols)

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot.bar(ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("pass rate")
    ax.set_xlabel("QA type")
    ax.set_title("Ablation: QA accuracy by type × memory mode")
    ax.legend(title="mode", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "ablation.png", dpi=150)
    plt.close(fig)


def _plot_qa_by_type(qa_df: pd.DataFrame, out_dir: Path) -> None:
    by_type = qa_df.groupby("qa_type")["passed"].mean()
    # упорядочим по канону, неизвестные — в конец
    order = [t for t in QA_TYPE_ORDER if t in by_type.index] + \
            [t for t in by_type.index if t not in QA_TYPE_ORDER]
    by_type = by_type.reindex(order)

    fig, ax = plt.subplots(figsize=(8, 5))
    by_type.plot.bar(ax=ax, color="steelblue")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("pass rate")
    ax.set_xlabel("QA type")
    ax.set_title("QA accuracy by type")
    for i, v in enumerate(by_type):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "qa_by_type.png", dpi=150)
    plt.close(fig)


def _plot_db_diagnostics(snap_df: pd.DataFrame, out_dir: Path) -> None:
    means = {
        "extraction\nrecall": snap_df["extraction_recall"].mean(),
        "summary\nfidelity": snap_df["summary_fidelity"].mean(),
    }
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(list(means.keys()), list(means.values()), color=["seagreen", "darkorange"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean")
    ax.set_title("DB diagnostics (secondary)")
    for i, v in enumerate(means.values()):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "db_diagnostics.png", dpi=150)
    plt.close(fig)


def _write_summary_md(qa_df: pd.DataFrame, snap_df: pd.DataFrame, out_dir: Path) -> None:
    lines: list[str] = ["# memory-bench results", ""]

    lines.append("## QA pass rate: type × mode (ablation, primary)")
    lines.append("")
    if qa_df.empty:
        lines.append("_нет QA-данных_")
    else:
        pivot = qa_df.pivot_table(index="qa_type", columns="mode",
                                  values="passed", aggfunc="mean")
        rows = [t for t in QA_TYPE_ORDER if t in pivot.index]
        cols = [m for m in MODE_ORDER if m in pivot.columns]
        pivot = pivot.reindex(index=rows, columns=cols)
        lines.append("| qa_type | " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for qa_type in pivot.index:
            cells = []
            for m in cols:
                v = pivot.loc[qa_type, m]
                cells.append("—" if pd.isna(v) else f"{v:.0%}")
            lines.append(f"| {qa_type} | " + " | ".join(cells) + " |")
        # overall by mode
        by_mode = qa_df.groupby("mode")["passed"].mean()
        overall = [f"{by_mode.get(m, float('nan')):.0%}" if m in by_mode else "—" for m in cols]
        lines.append(f"| **overall** | " + " | ".join(f"**{o}**" for o in overall) + " |")
    lines.append("")
    lines.append("![ablation](ablation.png)")
    lines.append("")

    lines.append("## DB diagnostics (secondary)")
    lines.append("")
    if snap_df.empty:
        lines.append("_нет snapshot-данных_")
    else:
        lines.append(f"- extraction recall (mean): {snap_df['extraction_recall'].mean():.0%}")
        lines.append(f"- summary fidelity (mean): {snap_df['summary_fidelity'].mean():.0%}")
    lines.append("")
    lines.append("![QA by type](qa_by_type.png)")
    lines.append("")
    lines.append("![DB diagnostics](db_diagnostics.png)")
    lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
