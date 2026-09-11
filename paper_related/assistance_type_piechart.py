import plotly.graph_objects as go

labels = [
    "All Assistance",

    "Physical Support",
    "Part Search &\nTask Decisions",
    "Assembly Error\nCorrection",
    "Conversational\nSupport",

    "Part Out\nof Reach",
    "Workpiece\nStabilization",

    "Unable to\nLocate Part",
    "Uncertain About\nNext Step",

    "Incorrect Part\nSelection",
    "Assembly-Order\nViolation",
    "Incorrect Part\nOrientation",
    "Incorrect\nScrewdriver / Bit",
    "Missed Fastener",

    "Confirmation /\nAffirmation",
    "Other Conversational\nSupport"
]

parents = [
    "",

    "All Assistance",
    "All Assistance",
    "All Assistance",
    "All Assistance",

    "Physical Support",
    "Physical Support",

    "Part Search &\nTask Decisions",
    "Part Search &\nTask Decisions",

    "Assembly Error\nCorrection",
    "Assembly Error\nCorrection",
    "Assembly Error\nCorrection",
    "Assembly Error\nCorrection",
    "Assembly Error\nCorrection",

    "Conversational\nSupport",
    "Conversational\nSupport"
]

values = [
    851,

    208,
    247,
    301,
    95,

    172,
    36,

    124,
    123,

    96,
    63,
    63,
    57,
    22,

    52,
    43
]

fig = go.Figure(
    go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        sort=False,
        maxdepth=3,

        insidetextorientation="radial",

        texttemplate="<b>%{label}</b><br>%{value}/851 (%{percentRoot:.1%})",

        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count: %{value}<br>"
            "Percent of all assistance: %{percentRoot:.1%}"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title=dict(
        text="Assistance Needs Observed in Human–Human Coassembly<br>"
             "<sup>851 annotated assistance events</sup>",
        x=0.5,
        xanchor="center",
        font=dict(size=34)
    ),

    font=dict(
        family="Times New Roman",
        size=22,
        color="black"
    ),

    uniformtext=dict(
        minsize=18,
        mode="hide"
    ),

    paper_bgcolor="white",
    plot_bgcolor="white",

    width=1800,
    height=1100,

    margin=dict(t=120, l=30, r=30, b=30),

    annotations=[
        dict(
            x=1.02,
            y=0.78,
            xref="paper",
            yref="paper",
            text=(
                "<b>DETAILED EVENT CATEGORIES</b><br><br>"
                "<span style='color:#4c5cff'><b>• PHYSICAL SUPPORT</b></span><br>"
                "Part Out of Reach — <b>172/851 (20.2%)</b><br>"
                "Workpiece Stabilization — <b>36/851 (4.2%)</b><br><br>"

                "<span style='color:#f05a3a'><b>• PART SEARCH & TASK DECISIONS</b></span><br>"
                "Unable to Locate Part — <b>124/851 (14.6%)</b><br>"
                "Uncertain About Next Step — <b>123/851 (14.5%)</b><br><br>"

                "<span style='color:#00c98d'><b>• ASSEMBLY ERROR CORRECTION</b></span><br>"
                "Incorrect Part Selection — <b>96/851 (11.3%)</b><br>"
                "Assembly-Order Violation — <b>63/851 (7.4%)</b><br>"
                "Incorrect Part Orientation — <b>63/851 (7.4%)</b><br>"
                "Incorrect Screwdriver / Bit — <b>57/851 (6.7%)</b><br>"
                "Missed Fastener — <b>22/851 (2.6%)</b><br><br>"

                "<span style='color:#a55cff'><b>• CONVERSATIONAL SUPPORT</b></span><br>"
                "Confirmation / Affirmation — <b>52/851 (6.1%)</b><br>"
                "Other Conversational Support — <b>43/851 (5.1%)</b>"
            ),
            showarrow=False,
            align="left",
            font=dict(
                family="Times New Roman",
                size=20,
                color="black"
            )
        )
    ]
)

fig.show()

fig.write_image("assistance_types_piechart_big.png", width=1800, height=1100, scale=3)
fig.write_image("assistance_types_piechart_big.pdf", width=1800, height=1100)
fig.write_image("assistance_types_piechart_big.svg", width=1800, height=1100)