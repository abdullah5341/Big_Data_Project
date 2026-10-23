"""Draws the system architecture diagram for the report and slides.

Run:  python src/make_architecture_diagram.py
Output: figures/architecture.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import FIGURES_DIR

OFFLINE_COLOR, ONLINE_COLOR, STORE_COLOR = "#dbe8f8", "#fbe1d6", "#e9e8e4"
EDGE, TEXT = "#52514e", "#0b0b0b"
BOX_W, BOX_H = 2.15, 1.0

# (label, x, y, colour). x/y are the box's lower-left corner.
OFFLINE_STEPS = [
    ("CDC BRFSS 2015\n(UCI #891)\n253,680 rows", 0.2),
    ("Data loading &\nquality check\ndata_loader.py", 2.75),
    ("EDA\n01_eda.ipynb", 5.3),
    ("Deduplicate +\nfeature engineering\n+ 70/15/15 split\npreprocess.py", 7.85),
    ("Train + tune\nLR / RF / XGBoost\n(5-fold CV, train only)\ntrain.py", 10.4),
    ("Threshold on\nvalidation\n→ test once\nevaluate.py", 12.95),
]
ONLINE_STEPS = [
    ("User\n(questionnaire)", 0.2),
    ("Streamlit\nweb interface\napp.py", 2.75),
    ("Input validation\n(ranges, BMI from\nheight & weight)", 5.3),
    ("Same feature\nengineering +\nsaved pipeline", 7.85),
    ("XGBoost →\nrisk score +\nrisk band", 10.4),
    ("SHAP explanation\n→ result\ndashboard", 12.95),
]
OFFLINE_Y, ONLINE_Y, STORE_Y = 4.2, 0.4, 2.35


def draw_box(ax, x, y, label, color, width=BOX_W, height=BOX_H):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor=color, edgecolor=EDGE, linewidth=1))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=8.2, color=TEXT)


def draw_arrow(ax, start, end, style="-|>", dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12, color=EDGE,
                                 linewidth=1.1, linestyle="--" if dashed else "-"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 6.6))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(0, 6.3)
    ax.axis("off")

    ax.text(0.2, 5.75, "OFFLINE — training & evaluation pipeline (run once, laptop)", fontsize=10.5,
            fontweight="bold", color=TEXT)
    ax.text(0.2, 1.75, "ONLINE — prototype (per user request)", fontsize=10.5, fontweight="bold", color=TEXT)

    for steps, y, color in [(OFFLINE_STEPS, OFFLINE_Y, OFFLINE_COLOR), (ONLINE_STEPS, ONLINE_Y, ONLINE_COLOR)]:
        for i, (label, x) in enumerate(steps):
            draw_box(ax, x, y, label, color)
            if i < len(steps) - 1:
                next_x = steps[i + 1][1]
                draw_arrow(ax, (x + BOX_W, y + BOX_H / 2), (next_x, y + BOX_H / 2))

    # Saved artefacts shared by both halves.
    store_x, store_w = 9.1, 4.6
    draw_box(ax, store_x, STORE_Y, "Saved artefacts: models/xgboost.joblib (pipeline + threshold)\n"
             "results/*.csv · figures/*.png", STORE_COLOR, width=store_w, height=0.75)
    draw_arrow(ax, (11.475, OFFLINE_Y), (11.475, STORE_Y + 0.75), dashed=True)
    draw_arrow(ax, (14.0, OFFLINE_Y), (13.2, STORE_Y + 0.75), dashed=True)
    draw_arrow(ax, (10.2, STORE_Y), (8.95, ONLINE_Y + BOX_H), dashed=True)
    draw_arrow(ax, (12.6, STORE_Y), (14.0, ONLINE_Y + BOX_H), dashed=True)

    # Leakage guard note.
    ax.text(0.2, 3.35, "Leakage guards: duplicates removed before split · scaling inside Pipeline · "
            "tuning on train folds only\nthreshold chosen on validation · test set used once",
            fontsize=8.2, color=EDGE, va="center")
    ax.text(0.2, 2.55, "Future scaling: Parquet + Spark MLlib / distributed XGBoost (100× data)\n"
            "Kafka + streaming scoring + drift monitoring (real-time)",
            fontsize=8.2, color=EDGE, va="center", style="italic")

    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "architecture.png", bbox_inches="tight", dpi=300)
    print(f"Saved {FIGURES_DIR / 'architecture.png'}")


if __name__ == "__main__":
    main()
