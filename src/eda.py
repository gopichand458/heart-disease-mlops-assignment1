"""Exploratory data analysis plots. Run `python -m src.eda` to regenerate all figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import CATEGORICAL, FIG_DIR, NUMERIC, TARGET  # noqa: E402
from src.data import load_clean  # noqa: E402

PALETTE = {0: "#2a9d8f", 1: "#e76f51"}
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def class_balance(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(["No disease", "Disease"], counts.values, color=[PALETTE[0], PALETTE[1]])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, f"{v} ({v / len(df):.0%})", ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("patients")
    return _save(fig, "class_balance.png")


def numeric_histograms(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), NUMERIC):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, palette=PALETTE,
                     element="step", ax=ax)
        ax.set_title(f"Distribution of {col}")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Numeric features by target", y=1.02)
    fig.tight_layout()
    return _save(fig, "numeric_histograms.png")


def numeric_boxplots(df: pd.DataFrame):
    fig, axes = plt.subplots(1, len(NUMERIC), figsize=(15, 4))
    for ax, col in zip(axes, NUMERIC):
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, palette=PALETTE, legend=False, ax=ax)
        ax.set_title(col)
    fig.tight_layout()
    return _save(fig, "numeric_boxplots.png")


def categorical_counts(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), CATEGORICAL):
        sns.countplot(data=df, x=col, hue=TARGET, palette=PALETTE, ax=ax)
        ax.set_title(col)
        ax.legend(title="target", fontsize=8)
    fig.tight_layout()
    return _save(fig, "categorical_counts.png")


def correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation heatmap")
    return _save(fig, "correlation_heatmap.png")


def missing_values(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(df.isna().T, cbar=False, cmap="magma", ax=ax)
    ax.set_title("Missing values (bright = missing)")
    ax.set_xlabel("row")
    return _save(fig, "missing_values.png")


def generate_all(df: pd.DataFrame = None):
    df = load_clean() if df is None else df
    paths = [class_balance(df), numeric_histograms(df), numeric_boxplots(df),
             categorical_counts(df), correlation_heatmap(df), missing_values(df)]
    plt.close("all")
    return paths


if __name__ == "__main__":
    for p in generate_all():
        print("saved", p)
