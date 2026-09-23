"""Publication-ready charts for the repository README."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, PercentFormatter

INK = "#14251E"
MUTED = "#68766F"
GRID = "#DDE5E1"
ACCENT = "#176B57"
ACCENT_LIGHT = "#8DB8AA"
BLUE = "#4C7391"
PALE = "#EFF4F1"
NEGATIVE = "#B9655A"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.labelcolor": MUTED,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _finish(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def draw_segment_value(summary: pd.DataFrame, output_dir: Path) -> Path:
    """Compare customer share and revenue share for every RFM segment."""
    data = summary.sort_values("revenue_share", ascending=True).copy()
    data[["customer_share", "revenue_share"]] *= 100
    y = np.arange(len(data))
    height = 0.34

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.barh(
        y - height / 2,
        data["customer_share"],
        height,
        color=ACCENT_LIGHT,
        label="Доля клиентов",
    )
    ax.barh(
        y + height / 2,
        data["revenue_share"],
        height,
        color=ACCENT,
        label="Доля выручки",
    )
    ax.set_yticks(y, data.index)
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("Доля от общего итога")
    ax.set_title("Кто формирует выручку", loc="left", pad=20)
    ax.text(
        0,
        1.015,
        "Сравнение размера RFM-сегмента и его вклада в выручку",
        transform=ax.transAxes,
        color=MUTED,
        va="bottom",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncol=2, loc="lower right")

    for values, offset in (
        (data["customer_share"], -height / 2),
        (data["revenue_share"], height / 2),
    ):
        for index, value in enumerate(values):
            ax.text(
                value + 0.6,
                index + offset,
                f"{value:.1f}%",
                va="center",
                fontsize=9,
                color=INK,
            )
    ax.set_xlim(
        0, max(data["revenue_share"].max(), data["customer_share"].max()) * 1.18
    )
    fig.tight_layout()
    return _finish(fig, output_dir / "rfm-segment-value.png")


def draw_abc_xyz_matrix(assortment: pd.DataFrame, output_dir: Path) -> Path:
    """Show product count and revenue contribution in the ABC/XYZ matrix."""
    counts = assortment.pivot_table(
        index="ABC", columns="XYZ", values="revenue", aggfunc="size"
    )
    revenue = assortment.pivot_table(
        index="ABC", columns="XYZ", values="revenue", aggfunc="sum"
    )
    counts = counts.reindex(index=list("ABC"), columns=list("XYZ"), fill_value=0)
    revenue = revenue.reindex(index=list("ABC"), columns=list("XYZ"), fill_value=0)
    shares = revenue / revenue.to_numpy().sum() * 100

    cmap = LinearSegmentedColormap.from_list(
        "green_scale", [PALE, ACCENT_LIGHT, ACCENT]
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    image = ax.imshow(
        shares.to_numpy(), cmap=cmap, aspect="auto", vmin=0, vmax=shares.max().max()
    )
    ax.set_xticks(range(3), ["X - стабильный", "Y - умеренный", "Z - нестабильный"])
    ax.set_yticks(
        range(3), ["A - основной вклад", "B - средний вклад", "C - малый вклад"]
    )
    ax.set_xlabel("Стабильность спроса")
    ax.set_ylabel("Вклад в выручку")
    ax.set_title("Матрица ассортимента ABC/XYZ", loc="left", pad=28)
    ax.text(
        0,
        1.035,
        "В каждой ячейке: число товаров и доля общей выручки",
        transform=ax.transAxes,
        color=MUTED,
        va="bottom",
    )

    threshold = shares.max().max() * 0.52
    for row in range(3):
        for column in range(3):
            value = shares.iloc[row, column]
            text_color = "white" if value > threshold else INK
            ax.text(
                column,
                row,
                f"{int(counts.iloc[row, column]):,} товаров\n{value:.1f}% выручки".replace(
                    ",", " "
                ),
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold" if value > threshold else "normal",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.04)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    colorbar.set_label("Доля выручки")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return _finish(fig, output_dir / "abc-xyz-matrix.png")


def draw_cluster_profile(profile: pd.DataFrame, output_dir: Path) -> Path:
    """Render comparable cluster profiles with raw values in each cell."""
    columns = ["mean_recency", "mean_frequency", "mean_monetary"]
    values = profile[columns].astype(float)
    standardized = (values - values.mean()) / values.std(ddof=0).replace(0, 1)
    order = values.sort_values("mean_monetary", ascending=False).index
    values = values.loc[order]
    standardized = standardized.loc[order]

    labels = [
        "Давность, дней",
        "Заказов на клиента",
        "Выручка на клиента, £",
    ]
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    norm = TwoSlopeNorm(
        vmin=min(-2, standardized.min().min()),
        vcenter=0,
        vmax=max(2, standardized.max().max()),
    )
    cmap = LinearSegmentedColormap.from_list("profile", [BLUE, PALE, "#C4824A"])
    image = ax.imshow(standardized.to_numpy(), cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(3), labels)
    ax.set_yticks(
        range(len(values)),
        [
            f"Кластер {cluster} · {int(profile.loc[cluster, 'customers']):,} клиентов".replace(
                ",", " "
            )
            for cluster in values.index
        ],
    )
    ax.set_title("Профили клиентских кластеров", loc="left", pad=28)
    ax.text(
        0,
        1.035,
        "Цвет показывает отклонение от среднего, подпись - фактическое значение",
        transform=ax.transAxes,
        color=MUTED,
        va="bottom",
    )

    for row in range(len(values)):
        raw = values.iloc[row]
        annotations = [
            f"{raw.iloc[0]:.0f}",
            f"{raw.iloc[1]:.1f}",
            f"{raw.iloc[2]:,.0f}".replace(",", " "),
        ]
        for column, annotation in enumerate(annotations):
            intensity = abs(standardized.iloc[row, column])
            ax.text(
                column,
                row,
                annotation,
                ha="center",
                va="center",
                color="white" if intensity > 1.35 else INK,
                fontweight="bold" if intensity > 1.35 else "normal",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.04)
    colorbar.set_label("Отклонение от среднего, σ")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return _finish(fig, output_dir / "customer-cluster-profile.png")


def draw_ltv(ltv: pd.DataFrame, output_dir: Path) -> Path:
    """Compare historical and discounted revenue LTV across segments."""
    data = ltv.sort_values("discounted_revenue_ltv", ascending=True)
    y = np.arange(len(data))
    height = 0.34

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.barh(
        y - height / 2,
        data["historical_revenue_ltv"],
        height,
        color=ACCENT_LIGHT,
        label="Историческая выручка",
    )
    ax.barh(
        y + height / 2,
        data["discounted_revenue_ltv"],
        height,
        color=BLUE,
        label="Дисконтированный LTV",
    )
    ax.set_yticks(y, data.index)
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: (
                f"{value / 1000:.0f} тыс." if value >= 1000 else f"{value:.0f}"
            )
        )
    )
    ax.set_xlabel("Выручка на клиента, £")
    ax.set_title("Сценарный revenue LTV по сегментам", loc="left", pad=20)
    ax.text(
        0,
        1.015,
        "Горизонт 5 лет, ставка дисконтирования 15%; расчёт не является прибылью",
        transform=ax.transAxes,
        color=MUTED,
        va="bottom",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    for index, value in enumerate(data["discounted_revenue_ltv"]):
        ax.text(
            value + data["discounted_revenue_ltv"].max() * 0.012,
            index + height / 2,
            f"{value / 1000:.1f} тыс." if value >= 1000 else f"{value:.0f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, data["discounted_revenue_ltv"].max() * 1.12)
    fig.tight_layout()
    return _finish(fig, output_dir / "segment-ltv.png")


def build_readme_assets(artifacts_dir: Path, output_dir: Path) -> list[Path]:
    """Build every README chart from pipeline artifacts."""
    _apply_style()
    segment_summary = pd.read_csv(artifacts_dir / "segment_summary.csv", index_col=0)
    assortment = pd.read_csv(artifacts_dir / "abc_xyz.csv", index_col=0)
    cluster_profile = pd.read_csv(artifacts_dir / "cluster_profile.csv", index_col=0)
    ltv = pd.read_csv(artifacts_dir / "segment_ltv.csv", index_col=0)
    return [
        draw_segment_value(segment_summary, output_dir),
        draw_abc_xyz_matrix(assortment, output_dir),
        draw_cluster_profile(cluster_profile, output_dir),
        draw_ltv(ltv, output_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build charts for the repository README"
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("assets"))
    args = parser.parse_args()
    for path in build_readme_assets(args.artifacts, args.output):
        print(path)


if __name__ == "__main__":
    main()
