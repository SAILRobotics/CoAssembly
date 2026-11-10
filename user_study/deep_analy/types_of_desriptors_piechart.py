import plotly.graph_objects as go

labels = [
    "Spatial Location / Deixis",
    "Object Shape",
    "Relative Size",
    "Number of Holes",
    "Object Color",
    "Screw Type"
]

values = [244, 132, 119, 83, 56, 22]

fig = go.Figure(
    go.Pie(
        labels=labels,
        values=values,
        hole=0.35,
        sort=False,

        texttemplate="<b>%{label}</b><br>%{value}/656 (%{percent:.1%})",
        textposition="inside",

        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count: %{value}<br>"
            "Percentage: %{percent:.1%}"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title=dict(
        text="Descriptors Used in Part Referring Expressions<br>"
             "<sup>656 annotated descriptor mentions</sup>",
        x=0.5,
        xanchor="center",
        font=dict(size=34)
    ),

    font=dict(
        family="Times New Roman",
        size=24,
        color="black"
    ),

    uniformtext=dict(
        minsize=20,
        mode="hide"
    ),

    paper_bgcolor="white",
    plot_bgcolor="white",

    width=1500,
    height=1000,

    margin=dict(t=120, l=30, r=30, b=30),

    legend=dict(
        font=dict(size=18),
        x=1.02,
        y=0.5,
        yanchor="middle"
    ),

    annotations=[
        dict(
            text="<b>656</b><br>Descriptor<br>Mentions",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                family="Times New Roman",
                size=24,
                color="black"
            )
        )
    ]
)

fig.show()

fig.write_image("part_referral_piechart_big.png", width=1500, height=1000, scale=3)
fig.write_image("part_referral_piechart_big.pdf", width=1500, height=1000)
fig.write_image("part_referral_piechart_big.svg", width=1500, height=1000)