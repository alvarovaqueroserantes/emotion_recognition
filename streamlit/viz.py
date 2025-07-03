from __future__ import annotations

import json
from io import BytesIO
from typing import Sequence

import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar, HeatMap, Line, Pie, Radar, Gauge, Timeline
from pyecharts.globals import ThemeType
from pyecharts.commons.utils import JsCode

from config import cfg, AppConfig


def _init(theme: ThemeType = ThemeType.LIGHT) -> opts.InitOpts:
    return opts.InitOpts(
        theme=theme,
        width="100%",
        height="100%",
        animation_opts=opts.AnimationOpts(
            animation_duration=1000,
            animation_easing="cubicOut",
            animation_delay=100,
        )
    )


def _percent_label(color: str, position: str = "top") -> opts.LabelOpts:
    return opts.LabelOpts(
        position=position,
        color=color,
        formatter="{c}%",
        font_weight="bold",
        font_size=12
    )


def emotion_bar(perc: dict[str, float], c: AppConfig = cfg) -> Bar:
    df = pd.DataFrame({
        "emotion": perc.keys(),
        "share": [round(v * 100, 1) for v in perc.values()]
    }).sort_values("share", ascending=False)

    return (
        Bar(init_opts=_init())
        .add_xaxis(df["emotion"].tolist())
        .add_yaxis(
            "Emotion Share", df["share"].tolist(),
            label_opts=_percent_label(c.palette["text"], "right"),
            itemstyle_opts=opts.ItemStyleOpts(color=c.palette["accent"]),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Emotion Distribution (Bar Chart)", pos_left="center",
                title_textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
            ),
            xaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(color=c.palette["text"]),
                axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color=c.palette["border"]))
            ),
            yaxis_opts=opts.AxisOpts(
                name="Percentage", max_=100,
                axislabel_opts=opts.LabelOpts(formatter="{value}%", color=c.palette["text"]),
                splitline_opts=opts.SplitLineOpts(
                    is_show=True, linestyle_opts=opts.LineStyleOpts(color=c.palette["border"], opacity=0.3)
                )
            ),
            tooltip_opts=opts.TooltipOpts(
                formatter="{b}: {c}%", background_color=c.palette["card"],
                textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
            ),
            legend_opts=opts.LegendOpts(is_show=False)
        )
        .reversal_axis()
    )


def emotion_pie(perc: dict[str, float], c: AppConfig = cfg) -> Pie:
    data = [[emotion, round(percentage * 100, 2)] for emotion, percentage in perc.items() if percentage > 0]
    data.sort(key=lambda x: x[1], reverse=True)

    color_mapping = {
        emotion: c.palette["chart_colors"][idx % len(c.palette["chart_colors"])]
        for idx, emotion in enumerate(c.emotion_labels)
    }

    return (
        Pie(init_opts=_init())
        .add(
            series_name="Emotions",
            data_pair=data,
            radius=["45%", "75%"],
            center=["50%", "60%"],  # alineado con radar
            rosetype="radius",
            label_opts=opts.LabelOpts(
                formatter="{b}: {d}%",
                color=c.palette["text"],
                font_size=12,
            ),
            itemstyle_opts=opts.ItemStyleOpts(
                border_width=2,
                border_color=c.palette["card"]
            ),
        )
        .set_colors([color_mapping[e[0]] for e in data])
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Emotion Distribution (Pie Chart)",
                pos_left="center",
                pos_top="5%",
                title_textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
            ),
            legend_opts=opts.LegendOpts(
                type_="scroll",
                orient="vertical",
                pos_left="left",     # <-- vertical en la izquierda
                pos_top="middle",    # <-- centrado verticalmente
                textstyle_opts=opts.TextStyleOpts(color=c.palette["text"]),
            ),
            tooltip_opts=opts.TooltipOpts(
                formatter="{b}: {d}%",
                background_color=c.palette["card"],
                textstyle_opts=opts.TextStyleOpts(color=c.palette["text"]),
            ),
        )
    )





def bullet_metric(value: float, title: str, c: AppConfig = cfg) -> Bar:
    pct = round(value * 100, 1)
    return (
        Bar(init_opts=_init())
        .add_xaxis([title])
        .add_yaxis(
            "", [pct],
            bar_width="40%",
            label_opts=opts.LabelOpts(
                position="insideRight",
                formatter="{c} %",
                font_size=18,
                font_weight="bold",
                color="#FFFFFF"
            ),
            itemstyle_opts=opts.ItemStyleOpts(
                color=c.palette["accent"],
                border_radius=5
            ),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                is_show=False
            ),
            yaxis_opts=opts.AxisOpts(
                is_show=False,
                max_=100
            ),
            tooltip_opts=opts.TooltipOpts(
                formatter=f"{title}: {{c}} %",
                background_color=c.palette["card"],
                textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
            ),
            title_opts=opts.TitleOpts(
                title=title,
                pos_left="center",
                pos_top="5%",
                title_textstyle_opts=opts.TextStyleOpts(
                    color=c.palette["text"],
                    font_size=14
                )
            )
        )
    )



def emotion_radar(perc: dict[str, float], c: AppConfig = cfg) -> Radar:
    data_values = [round(v * 100, 1) for v in perc.values()]
    data = [{"value": data_values, "name": "Distribution"}]

    schema = [
        opts.RadarIndicatorItem(name=emotion, max_=100)
        for emotion in c.emotion_labels
    ]

    return (
        Radar(
            init_opts=opts.InitOpts(
                theme=ThemeType.LIGHT,
                width="100%",
                height="100%",
                animation_opts=opts.AnimationOpts(
                    animation_duration=1000,
                    animation_easing="cubicOut",
                    animation_delay=100,
                )
            )
        )
        .add_schema(
            schema=schema,
            center=["50%", "60%"],  # Mueve el radar un poco hacia abajo
            splitarea_opt=opts.SplitAreaOpts(
                is_show=True,
                areastyle_opts=opts.AreaStyleOpts(opacity=0.08)
            ),
            textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
        )
        .add(
            series_name="Emotions",
            data=data,
            linestyle_opts=opts.LineStyleOpts(
                color=c.palette["accent"],
                width=2
            ),
            areastyle_opts=opts.AreaStyleOpts(
                opacity=0.3,
                color=c.palette["accent"]
            ),
            symbol="circle",
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Emotion Distribution (Radar Chart)",
                pos_left="center",
                pos_top="5%",  # Deja el título arriba con un 5% de margen
                title_textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
            ),
            tooltip_opts=opts.TooltipOpts(
                background_color=c.palette["card"],
                textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
            ),
            legend_opts=opts.LegendOpts(is_show=False)
        )
    )




def sentiment_gauge(score: float, c: AppConfig = cfg) -> Gauge:
    display_score = round(score, 2)
    mapped_score = (score + 1) * 50  # maps -1..1 to 0..100

    ordered_emotions = ["Sad", "Fear", "Angry", "Neutral", "Disgust", "Surprise", "Happy"]

    # buscar colores robustos
    emotion_labels_lc = [e.lower() for e in c.emotion_labels]
    emotion_colors = []
    for emotion in ordered_emotions:
        try:
            idx = emotion_labels_lc.index(emotion.lower())
            color = c.palette["chart_colors"][idx]
        except ValueError:
            color = "#888888"
        emotion_colors.append(color)

    # todos los segmentos iguales
    num_emotions = len(ordered_emotions)
    color_steps = [
        [(i + 1) / num_emotions, emotion_colors[i]]
        for i in range(num_emotions)
    ]

    return (
        Gauge(init_opts=_init())
        .add(
            series_name="Sentiment",
            data_pair=[("Score", mapped_score)],
            axisline_opts=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(
                    color=color_steps,
                    width=25
                )
            ),
            pointer={
                "length": "70%",
                "width": 8,
                "itemStyle": {"color": "#888888"}  # puntero gris
            },
            axislabel_opts=opts.LabelOpts(is_show=False),
            min_=0,
            max_=100,
            split_number=num_emotions,
        )
        .set_series_opts(
            detail={
                "formatter": JsCode("function(value){ return (value/50-1).toFixed(2); }"),
                "fontSize": 28,
                "color": c.palette["text"]
            }
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Overall Sentiment Score",
                pos_left="center",
                title_textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
            ),
            tooltip_opts=opts.TooltipOpts(
                formatter=f"Sentiment: {display_score}",
                background_color=c.palette["card"],
                textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
            ),
            legend_opts=opts.LegendOpts(is_show=False)
        )
    )











def rolling_share_line(
    timeline: Sequence[dict[str, int]], window: int = 60, c: AppConfig = cfg
) -> Line:
    if not timeline:
        return (
            Line(init_opts=_init())
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="No data for Rolling Emotion Share",
                    pos_left="center",
                    title_textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
                ),
                xaxis_opts=opts.AxisOpts(
                    name="Frame",
                    axislabel_opts=opts.LabelOpts(color=c.palette["text"])
                ),
                yaxis_opts=opts.AxisOpts(
                    name="Share (%)",
                    axislabel_opts=opts.LabelOpts(color=c.palette["text"])
                ),
                legend_opts=opts.LegendOpts(is_show=False)
            )
        )

    df = pd.DataFrame(timeline).fillna(0)
    df_rolling = df.rolling(window, min_periods=1).sum()
    total = df_rolling.sum(axis=1).replace(0, 1)
    share = df_rolling.div(total, axis=0) * 100
    share.insert(0, "frame", share.index + 1)

    line = Line(init_opts=_init())
    line.add_dataset(source=share.values.tolist())

    for idx, emotion in enumerate(share.columns[1:], start=1):
        color_idx = (idx - 1) % len(c.palette["chart_colors"])
        line.add_yaxis(
            series_name=emotion,
            y_axis=[],
            encode={"x": 0, "y": idx},
            is_smooth=True,
            stack="total",
            areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
            linestyle_opts=opts.LineStyleOpts(
                width=2,
                color=c.palette["chart_colors"][color_idx]
            ),
            label_opts=opts.LabelOpts(is_show=False),
            symbol="none"
        )

    line.set_global_opts(

        tooltip_opts=opts.TooltipOpts(
            trigger="axis",
            axis_pointer_type="cross",
            background_color=c.palette["card"],
            textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
        ),
        datazoom_opts=[
            opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
            opts.DataZoomOpts(
                type_="slider", pos_bottom="1%",
                pos_left="10%", pos_right="10%"
            )
        ],
        xaxis_opts=opts.AxisOpts(
            name="Frame",
            axisline_opts=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(color=c.palette["border"])
            ),
            axislabel_opts=opts.LabelOpts(color=c.palette["text"])
        ),
        yaxis_opts=opts.AxisOpts(
            name="Share (%)",
            max_=100,
            axislabel_opts=opts.LabelOpts(
                formatter="{value}%",
                color=c.palette["text"]
            ),
            splitline_opts=opts.SplitLineOpts(
                is_show=True,
                linestyle_opts=opts.LineStyleOpts(
                    opacity=0.3,
                    color=c.palette["border"]
                )
            )
        ),
        legend_opts=opts.LegendOpts(
            pos_left="right",
            orient="vertical",
            textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
        )
    )

    return line


def transition_heatmap(matrix: np.ndarray, c: AppConfig = cfg) -> HeatMap:
    mat = np.asarray(matrix, dtype=int)
    vmax = mat.max() or 1

    emotion_colors = c.palette["chart_colors"]

    data = []
    for i, from_emotion in enumerate(c.emotion_labels):
        for j, to_emotion in enumerate(c.emotion_labels):
            value = int(mat[i, j])
            if value > 0:
                # tomar colores de la paleta
                color_from = emotion_colors[i % len(emotion_colors)]
                color_to = emotion_colors[j % len(emotion_colors)]
                
                # promedio RGB simple
                def hex_to_rgb(hex_color):
                    hex_color = hex_color.lstrip("#")
                    return tuple(int(hex_color[k:k+2], 16) for k in (0, 2, 4))
                
                def rgb_to_hex(rgb):
                    return "#{:02X}{:02X}{:02X}".format(*rgb)
                
                rgb_from = hex_to_rgb(color_from)
                rgb_to = hex_to_rgb(color_to)
                
                rgb_avg = tuple((f + t) // 2 for f, t in zip(rgb_from, rgb_to))
                mixed_color = rgb_to_hex(rgb_avg)

                data.append({
                    "value": [j, i, value],
                    "itemStyle": {"color": mixed_color}
                })

    tooltip_fmt = JsCode(f"""
        function(params) {{
            var emotions = {json.dumps(list(c.emotion_labels))};
            return emotions[params.value[1]] + " → " + emotions[params.value[0]] + "<br/>Transitions: " + params.value[2];
        }}
    """)

    return (
        HeatMap(init_opts=_init())
        .add_xaxis(c.emotion_labels)
        .add_yaxis(
            series_name="Transitions",
            yaxis_data=c.emotion_labels,
            value=data,
            # el label interno de las celdas
            label_opts=opts.LabelOpts(is_show=True, color="#FFFFFF", font_size=10),
            # color de los cuadros de la leyenda
            itemstyle_opts=opts.ItemStyleOpts(
                color="#888888",  # cuadrado de la leyenda gris
                border_color="#DDDDDD",
                border_width=1
            )
        )
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(
                formatter=tooltip_fmt,
                background_color=c.palette["card"],
                textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
            ),
            xaxis_opts=opts.AxisOpts(
                name="To Emotion",
                axislabel_opts=opts.LabelOpts(
                    color=c.palette["text"], rotate=45, interval=0
                ),
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color=c.palette["border"])
                )
            ),
            yaxis_opts=opts.AxisOpts(
                name="From Emotion",
                axislabel_opts=opts.LabelOpts(color=c.palette["text"]),
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color=c.palette["border"])
                )
            )
        )
    )




def emotion_timeline(timeline: list, c: AppConfig = cfg) -> Timeline:
    tl = Timeline(init_opts=_init())

    if not timeline:
        empty_chart = (
            Bar(init_opts=_init())
            .add_xaxis([])
            .add_yaxis("Detections", [])
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="No data available",
                    pos_left="center",
                    title_textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
                )
            )
        )
        tl.add(empty_chart, "No data")
        return tl

    sample_interval = 1  # no skip, todos los frames

    for i in range(0, len(timeline), sample_interval):
        frame_data = timeline[i]
        data = [frame_data.get(e, 0) for e in c.emotion_labels]

        chart = (
            Bar(init_opts=_init())
            .add_xaxis(c.emotion_labels)
            .add_yaxis(
                series_name="Detections",
                y_axis=data,
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode(f"""
                        function(params) {{
                            var colors = {json.dumps(c.palette["chart_colors"])};
                            return colors[params.dataIndex % colors.length];
                        }}
                    """)
                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"Emotion Counts at Frame {i}",
                    pos_left="center",
                    title_textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
                ),
                xaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(color=c.palette["text"], rotate=30),
                    axisline_opts=opts.AxisLineOpts(
                        linestyle_opts=opts.LineStyleOpts(color=c.palette["border"])
                    )
                ),
                yaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(color=c.palette["text"]),
                    splitline_opts=opts.SplitLineOpts(
                        is_show=True,
                        linestyle_opts=opts.LineStyleOpts(
                            color=c.palette["border"], opacity=0.3
                        )
                    )
                ),
                tooltip_opts=opts.TooltipOpts(
                    background_color=c.palette["card"],
                    textstyle_opts=opts.TextStyleOpts(color=c.palette["text"])
                ),
                legend_opts=opts.LegendOpts(is_show=False)
            )
        )
        tl.add(chart, f"Frame {i}")

    tl.add_schema(
        play_interval=200,  # milisegundos (200 ms = 5 fps)
        is_auto_play=True,
        is_loop_play=True,
        control_position="bottom"
    )
    return tl




def metrics_to_dataframe(metrics: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(metrics).sort_values(["source", "emotion"])


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes | None:
    try:
        import openpyxl
    except ModuleNotFoundError:
        return None
    buf = BytesIO()
    df.to_excel(buf, index=False, sheet_name="metrics")
    return buf.getvalue()
