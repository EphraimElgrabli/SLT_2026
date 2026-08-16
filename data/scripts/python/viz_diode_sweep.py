#!/usr/bin/env python3

"""Render the DIODE-outdoor cap-sweep result as a slide-ready chart.

We already ran diag_diode_caps.py and have the numbers. Rather than re-run
inference (a few minutes), we hard-code the results here so the chart
generation is instant and self-contained.

The story we want the chart to tell: as the max_depth cap goes up, AbsRel
gets WORSE, and delta1 stays essentially flat. That visual disproves the
"cap is the culprit" hypothesis at a glance.

Output: data/outputs/slide_figures/diode_cap_sweep.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import resolve_root_dir  # noqa: E402


# Results from diag_diode_caps.py on 40 DIODE-outdoor samples (ViT-S).
# "no cap" is excluded from AbsRel because the unbounded value (75683)
# dwarfs the others and ruins the chart's readability; we annotate it
# in text instead. delta1 is bounded [0,1] so it plots fine.
CAPS = [80, 150, 200, 300]
ABS_REL = [0.3352, 0.3525, 0.3579, 0.3690]
DELTA1 = [0.5792, 0.5708, 0.5707, 0.5694]
NO_CAP_DELTA1 = 0.5694


def main() -> None:
    out_dir = resolve_root_dir() / "data" / "outputs" / "slide_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: AbsRel vs cap
    ax1.plot(CAPS, ABS_REL, marker="o", linewidth=2.2, markersize=9, color="#B91C1C")
    ax1.set_xlabel("max_depth cap (meters)", fontsize=11)
    ax1.set_ylabel("AbsRel  (lower is better)", fontsize=11)
    ax1.set_title("Raising the cap makes AbsRel WORSE,\nnot better", fontsize=12)
    ax1.grid(True, alpha=0.3)
    # Annotate each point with its value
    for cap, val in zip(CAPS, ABS_REL):
        ax1.annotate(f"{val:.3f}", (cap, val), textcoords="offset points",
                     xytext=(8, 6), fontsize=10)
    # Annotate the "no cap" extreme outside the plot
    ax1.text(0.97, 0.04, "no cap → AbsRel ≈ 75683  (chart-breaking)",
             transform=ax1.transAxes, ha="right", fontsize=9,
             style="italic", color="#666")

    # Right: delta1 vs cap, with the no-cap point shown for completeness
    ax2.plot(CAPS, DELTA1, marker="o", linewidth=2.2, markersize=9, color="#1F4E79",
             label="finite caps")
    # Plot no-cap as a separate marker far to the right (visual hint, not exact scale)
    ax2.plot([CAPS[-1] + 80], [NO_CAP_DELTA1], marker="s", markersize=10,
             color="#777", linestyle="", label="no cap")
    ax2.set_xlabel("max_depth cap (meters)", fontsize=11)
    ax2.set_ylabel("δ₁  (higher is better)", fontsize=11)
    ax2.set_title("δ₁ is essentially CONSTANT across caps\n(~0.57 regardless)", fontsize=12)
    ax2.set_ylim(0.55, 0.62)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)
    for cap, val in zip(CAPS, DELTA1):
        ax2.annotate(f"{val:.3f}", (cap, val), textcoords="offset points",
                     xytext=(8, 6), fontsize=10)

    fig.suptitle("DIODE-outdoor controlled cap sweep (40 samples, ViT-S)",
                 fontsize=13, y=1.02)
    # Caption below
    caption = ("Conclusion: the cap is not the cause of the DIODE-outdoor gap. "
               "AbsRel even worsens as the cap rises, and δ₁ stays flat — "
               "consistent with the model genuinely struggling on outdoor scenes, "
               "not with a mis-set evaluation range.")
    fig.text(0.5, -0.04, caption, ha="center", fontsize=9.5, style="italic",
             color="#444", wrap=True)
    fig.tight_layout()

    out_path = out_dir / "diode_cap_sweep.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote: {out_path}")


if __name__ == "__main__":
    main()