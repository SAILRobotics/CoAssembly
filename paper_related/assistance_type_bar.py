import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ============================================================
# DATA
# ============================================================

TOTAL = 851

groups = {
    "PHYSICAL SUPPORT": {
        "total": 208,
        "items": [
            ("Part out of reach", 172),
            ("Workpiece stabilization", 36),
        ]
    },

    "PART SEARCH & TASK DECISIONS": {
        "total": 247,
        "items": [
            ("Unable to locate part", 124),
            ("Uncertain about next step", 123),
        ]
    },

    "ASSEMBLY ERROR CORRECTION": {
        "total": 301,
        "items": [
            ("Incorrect part selection", 96),
            ("Assembly-order violation", 63),
            ("Incorrect part orientation", 63),
            ("Incorrect screwdriver / bit", 57),
            ("Missed fastener", 22),
        ]
    },

    "CONVERSATIONAL SUPPORT": {
        "total": 95,
        "items": [
            ("Confirmation / affirmation", 52),
            ("Other conversational support", 43),
        ]
    }
}


# ============================================================
# BUILD Y POSITIONS
# ============================================================

fig, ax = plt.subplots(figsize=(10, 8))

current_y = 0
group_gap = 1.4

all_positions = []
all_labels = []

for group_name, group_data in groups.items():

    items = group_data["items"]

    # current_y becomes fractional after adding group_gap, while range accepts
    # integers only. Arithmetic positions preserve the intended spacing.
    positions = [
        current_y + item_index
        for item_index in range(len(items))
    ]

    percentages = [
        count / TOTAL * 100
        for _, count in items
    ]

    # One bar call per group -> same color within group,
    # different default color between groups.
    bars = ax.barh(
        positions,
        percentages,
        height=0.62,
        label=group_name
    )

    # --------------------------------------------------------
    # Individual category labels and values
    # --------------------------------------------------------

    for bar, (item_name, count), pct in zip(
        bars, items, percentages
    ):
        y = bar.get_y() + bar.get_height() / 2

        ax.text(
            pct + 0.25,
            y,
            f"{count}/851 ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold"
        )

    all_positions.extend(positions)
    all_labels.extend([
        name for name, _ in items
    ])

    # --------------------------------------------------------
    # Group heading
    # --------------------------------------------------------

    group_total = group_data["total"]
    group_pct = group_total / TOTAL * 100

    header_y = positions[0] - 0.72

    ax.text(
        0,
        header_y,
        f"{group_name}  —  "
        f"{group_total}/851 ({group_pct:.1f}%)",
        fontsize=12.5,
        fontweight="bold",
        ha="left",
        va="center"
    )

    current_y += len(items) + group_gap


# ============================================================
# AXES
# ============================================================

ax.set_yticks(all_positions)
ax.set_yticklabels(
    all_labels,
    fontsize=11
)

ax.invert_yaxis()

ax.set_xlim(0, 24)

ax.set_xticks([0, 5, 10, 15, 20])
ax.xaxis.set_major_formatter(
    PercentFormatter(xmax=100, decimals=0)
)

ax.set_xlabel(
    "Percentage of All Assistance Events",
    fontsize=12
)

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.25
)

ax.set_axisbelow(True)


# ============================================================
# CLEAN PAPER STYLE
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.tick_params(
    axis="y",
    length=0,
    pad=8
)


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    "Assistance Needs Observed in Human–Human Coassembly",
    fontsize=17,
    fontweight="bold",
    y=0.98
)

ax.set_title(
    "851 annotated assistance events",
    fontsize=12,
    pad=22
)


# ============================================================
# LAYOUT / EXPORT
# ============================================================

plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)

plt.savefig(
    "assistance_needs_grouped.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "assistance_needs_grouped.svg",
    bbox_inches="tight"
)

plt.savefig(
    "assistance_needs_grouped.png",
    dpi=400,
    bbox_inches="tight"
)

plt.show()
