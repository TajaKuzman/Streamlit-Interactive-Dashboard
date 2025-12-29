import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io


# ---------------------
# Data loading
# ---------------------
@st.cache_data
def load_data(path: str = "results.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"task", "model", "language", "metric", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"results.csv is missing required columns: {', '.join(sorted(missing))}"
        )
    return df


# ---------------------
# Helper functions
# ---------------------
def format_metric_name(metric: str) -> str:
    if not isinstance(metric, str):
        return str(metric)
    metric = metric.replace("-", "_")
    parts = metric.split("_")
    return " ".join(p.capitalize() if p.lower() != "f1" else "F1" for p in parts)


def language_task_metric_summary(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []

    grouped = (
        df.groupby(["language", "task"])["metric"]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
    )

    lang_map: dict[str, dict[str, list[str]]] = {}
    for _, row in grouped.iterrows():
        lang = row["language"]
        task = row["task"]
        metrics = [format_metric_name(m) for m in row["metric"]]
        lang_map.setdefault(lang, {})[task] = metrics

    lines: list[str] = []
    for lang in sorted(lang_map.keys()):
        task_parts = []
        for task in sorted(lang_map[lang].keys()):
            metrics_str = ", ".join(lang_map[lang][task])
            task_parts.append(f"{task} ({metrics_str})")
        if task_parts:
            lines.append(f"{lang}: " + "; ".join(task_parts))
    return lines


def compute_model_rankings(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    if df.empty:
        return (
            pd.DataFrame(
                columns=[
                    "Model",
                    "Average ranking",
                    "Average ranking English",
                    "Average ranking South Slavic",
                ]
            ),
            {},
            [],
        )

    df_all = df.copy()
    tasks_all = sorted(df_all["task"].unique())
    n_tasks = len(tasks_all)

    tasks_per_model = df_all.groupby("model")["task"].nunique()
    full_models = tasks_per_model[tasks_per_model == n_tasks].index.tolist()

    if not full_models:
        return (
            pd.DataFrame(
                columns=[
                    "Model",
                    "Average ranking",
                    "Average ranking English",
                    "Average ranking South Slavic",
                ]
            ),
            {},
            [],
        )

    df_rank = df_all[df_all["model"].isin(full_models)].copy()

    agg = (
        df_rank.groupby(["task", "language", "model"])["value"]
        .mean()
        .reset_index(name="score")
    )

    if agg.empty:
        return (
            pd.DataFrame(
                columns=[
                    "Model",
                    "Average ranking",
                    "Average ranking English",
                    "Average ranking South Slavic",
                ]
            ),
            {},
            full_models,
        )

    agg["rank"] = agg.groupby(["task", "language"])["score"].rank(
        ascending=False, method="average"
    )

    overall_rank = agg.groupby("model")["rank"].mean().rename("Average ranking")

    per_lang_rank = agg.groupby(["model", "language"])["rank"].mean().reset_index()

    languages_all = sorted(per_lang_rank["language"].unique())
    english_langs = [l for l in languages_all if l.strip().lower() in {"english", "en"}]

    if english_langs:
        english_rank = (
            per_lang_rank[per_lang_rank["language"].isin(english_langs)]
            .groupby("model")["rank"]
            .mean()
            .rename("Average ranking English")
        )
    else:
        english_rank = pd.Series(
            index=overall_rank.index,
            data=float("nan"),
            name="Average ranking English",
        )

    if english_langs:
        south_slavic_rank = (
            per_lang_rank[~per_lang_rank["language"].isin(english_langs)]
            .groupby("model")["rank"]
            .mean()
            .rename("Average ranking South Slavic")
        )
    else:
        south_slavic_rank = overall_rank.rename("Average ranking South Slavic")

    lang_pivot = (
        per_lang_rank.pivot(index="model", columns="language", values="rank")
        if not per_lang_rank.empty
        else pd.DataFrame()
    )

    rank_df = overall_rank.to_frame().join(english_rank, how="left").join(
        south_slavic_rank, how="left"
    )

    if not lang_pivot.empty:
        rank_df = rank_df.join(lang_pivot, how="left")

    rank_df = rank_df.reset_index().rename(columns={"model": "Model"})

    base_cols = [
        "Model",
        "Average ranking",
        "Average ranking English",
        "Average ranking South Slavic",
    ]
    lang_cols = [c for c in rank_df.columns if c not in base_cols]
    rank_df = rank_df[base_cols + lang_cols]

    rank_df = rank_df.sort_values("Average ranking", ascending=True).reset_index(
        drop=True
    )

    tasks_per_lang_series = agg.groupby("language")["task"].apply(
        lambda s: sorted(s.unique())
    )
    tasks_per_lang = tasks_per_lang_series.to_dict()

    return rank_df, tasks_per_lang, full_models


def apply_filters(
    df: pd.DataFrame,
    tasks: list[str] | None,
    languages: list[str],
    models: list[str],
) -> pd.DataFrame:
    data = df.copy()

    if tasks:
        data = data[data["task"].isin(tasks)]

    if languages:
        data = data[data["language"].isin(languages)]

    if models:
        data = data[data["model"].isin(models)]

    return data


def order_metrics(metrics: list[str]) -> list[str]:
    def metric_key(m: str) -> tuple[int, str]:
        m_norm = m.lower().replace("-", "_")
        if "macro" in m_norm and "f1" in m_norm:
            return (0, m_norm)
        if "micro" in m_norm and "f1" in m_norm:
            return (1, m_norm)
        return (2, m_norm)

    return sorted(metrics, key=metric_key)


def sanitize_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def fig_bytes(fig: plt.Figure, fmt: str, dpi: int | None = None) -> bytes:
    buf = io.BytesIO()
    if fmt.lower() == "pdf":
        fig.savefig(buf, format="pdf", bbox_inches="tight")
    elif fmt.lower() == "svg":
        fig.savefig(buf, format="svg", bbox_inches="tight")
    else:
        fig.savefig(buf, format=fmt, dpi=dpi or 300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def plot_task_heatmap(
    df_task: pd.DataFrame,
    metric_label: str
):
    if df_task.empty:
        return

    matrix = df_task.pivot_table(
        index="model",
        columns="language",
        values="value",
        aggfunc="mean",
    )
    if matrix.empty:
        return

    matrix["Average"] = matrix.mean(axis=1)
    data = matrix.sort_values(by="Average", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        data,
        annot=False,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.3,
        vmax=1.0,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": metric_label},
        ax=ax,
    )

    n_rows, n_cols = data.shape
    for x in range(n_cols):
        col = data.iloc[:, x]
        max_val = col.max()
        for y in range(n_rows):
            val = col.iloc[y]
            if pd.isna(val):
                continue
            ax.text(
                x + 0.5,
                y + 0.5,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold" if val == max_val else "normal",
            )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    plt.close(fig)


def plot_task_bar_chart(
    df_task: pd.DataFrame,
    metric_label: str
):
    if df_task.empty:
        return

    matrix = df_task.pivot_table(
        index="language",
        columns="model",
        values="value",
        aggfunc="mean",
    )
    if matrix.empty:
        return

    data_min = float(np.nanmin(matrix.values))
    ymin = round(data_min - 0.01, 2)

    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    matrix.plot(kind="bar", width=0.9, colormap="tab20", ax=ax)

    # Smaller legend (font + box), and constrain layout
    leg = ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        title="Model",
        fontsize=7,
        title_fontsize=8,
        frameon=True,
        framealpha=0.85,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    if leg is not None:
        leg._legend_box.sep = 4  # keeps legend compact (matplotlib internal, but stable)

    ax.set_ylim(ymin, None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric_label, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)

    ymin2, ymax2 = ax.get_ylim()
    ymin2 = max(0.0, round(ymin2, 2))
    ymax2 = round(ymax2 + 0.05, 2)
    if ymax2 > ymin2:
        ticks = np.arange(ymin2, ymax2 + 1e-9, 0.05)
        ax.set_yticks(ticks)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)
    ax.set_axisbelow(True)

    # Give extra room on the right for the legend so it doesn't overlap / resize plot badly
    fig.tight_layout()
    fig.subplots_adjust(right=0.72)

    st.pyplot(fig, clear_figure=True)

    plt.close(fig)


def multiselect_with_select_all(
    label: str,
    options: list[str],
    default_all: bool = True,
    key: str | None = None,
) -> list[str]:
    if not options:
        return []

    select_all_label = "Select all"
    display_options = [select_all_label] + options

    default = display_options if default_all else []
    selected = st.multiselect(label, options=display_options, default=default, key=key)

    if not selected or select_all_label in selected:
        return options

    return [s for s in selected if s != select_all_label]


def show_metric_results_table(df_metric: pd.DataFrame, task_name: str, metric_label: str):
    if df_metric.empty:
        st.info(f"No data available for task {task_name} and metric {metric_label}.")
        return

    grouped = df_metric.groupby(["model", "language"])["value"].mean().reset_index()

    table = grouped.pivot_table(
        index="model",
        columns="language",
        values="value",
        aggfunc="mean",
    )

    if table.empty:
        st.info(f"No data available for task {task_name} and metric {metric_label}.")
        return

    table["Average"] = table.mean(axis=1)
    table = table.sort_values("Average", ascending=False)

    col_order = [c for c in table.columns if c != "Average"] + ["Average"]
    table = table[col_order]

    table = table.reset_index().rename(columns={"model": "Model"})

    num_cols = table.select_dtypes(include=["float", "int"]).columns
    styler = table.style.format("{:.3f}", subset=num_cols)

    st.dataframe(styler)


# ---------------------
# Streamlit layout
# ---------------------
def main():
    st.set_page_config(
        page_title="CLASSLA LLM Evaluation Dashboard for South Slavic Languages",
        layout="wide",
    )

    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
        html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
        p, span, div { font-size: 18px !important; }
        .stSidebar, .stSidebar p, .stSidebar span, .stSidebar label {
            font-size: 18px !important;
            font-family: 'Inter', sans-serif !important;
        }
        h1 { font-size: 38px !important; font-weight: 600 !important; }
        h2 { font-size: 30px !important; font-weight: 500 !important; }
        h3 { font-size: 24px !important; font-weight: 500 !important; }
        .stDataFrame div, .stDataFrame table, .stDataFrame th, .stDataFrame td {
            font-size: 16px !important;
            font-family: 'Inter', sans-serif !important;
        }
        [data-baseweb="tag"] { font-size: 16px !important; font-family: 'Inter', sans-serif !important; }
        [data-baseweb="select"] * { font-family: 'Inter', sans-serif !important; font-size: 18px !important; }
        [data-baseweb="tag"] { background-color: #0080aa !important; color: white !important; }
        [data-baseweb="tag"]:hover { background-color: #006c8d !important; color: white !important; }
        [data-baseweb="select"] div[data-baseweb="option"]:hover { background-color: #0080aa20 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_title, col_logos = st.columns([6, 1])

    with col_title:
        st.markdown(
            "<h1>CLASSLA LLM Evaluation Dashboard<br>for South Slavic Languages</h1>",
            unsafe_allow_html=True,
        )

    with col_logos:
        st.image("CLASSLA-k-centre-transparent-background.png", width='stretch')
        st.image("logo-clarin.si-en.png", width='stretch')

    st.markdown(
        """
        This interactive dashboard shows the performance of large language models (LLMs) and other technologies on various text classification 
        and commonsense reasoning benchmarks for South Slavic languages.
        """
    )

    try:
        df = load_data()
    except Exception as e:
        st.error(
            "Could not load results.csv. Make sure it exists next to app.py "
            "and has columns: task, model, language, metric, value."
        )
        st.exception(e)
        return

    lang_summary_lines = language_task_metric_summary(df)

    with st.expander(
        "Language coverage: for each language, which tasks and metrics are available",
        expanded=False,
    ):
        if not lang_summary_lines:
            st.info("No data available to summarise language coverage.")
        else:
            for line in lang_summary_lines:
                st.write(line)

    st.markdown("---")

    st.subheader("Global summary")

    st.markdown(
        """
        Models are ranked separately for each task–language pair based on their average score
        (across all available metrics). Ranks (1 = best) are then averaged across all tasks and
        languages. Only models evaluated on all tasks are included.
        """
    )

    ranking_df, tasks_per_lang, full_models = compute_model_rankings(df)

    if ranking_df.empty:
        st.info("No global ranking can be computed: no model was evaluated on all tasks.")
    else:
        chart_rank = (
            alt.Chart(ranking_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Model:N",
                    sort=alt.EncodingSortField(field="Average ranking", order="ascending"),
                    title="Model",
                ),
                y=alt.Y("Average ranking:Q", title="Average ranking (lower is better)"),
                tooltip=list(ranking_df.columns),
            )
        )
        st.altair_chart(chart_rank, width="stretch")

        base_cols = ["Model", "Average ranking", "Average ranking English", "Average ranking South Slavic"]
        lang_cols = [c for c in ranking_df.columns if c not in base_cols]

        if lang_cols:
            selected_rank_langs = st.multiselect(
                "Languages to display in ranking table",
                options=lang_cols,
                default=lang_cols,
            )
        else:
            selected_rank_langs = []

        display_cols = base_cols + selected_rank_langs

        st.dataframe(
            ranking_df[display_cols].style.format(
                {col: "{:.3f}" for col in ranking_df.columns if col != "Model"}
            )
        )

        if full_models:
            st.caption(
                "Models included in the global ranking (tested on all tasks): "
                + ", ".join(full_models)
            )

        if tasks_per_lang:
            st.markdown("Tasks included in the ranking per language:")
            for lang in sorted(tasks_per_lang.keys()):
                tasks_str = ", ".join(tasks_per_lang[lang])
                st.caption(f"{lang}: {tasks_str}")

    st.markdown("---")

    st.subheader("Task-level view")
    st.markdown("Use the filters below to inspect detailed per-task, per-language, per-model results.")

    tasks = sorted(df["task"].unique())
    languages_all = sorted(df["language"].unique())
    models_all = sorted(df["model"].unique())

    selected_tasks = multiselect_with_select_all("Tasks", options=tasks, key="tasks_multiselect")
    selected_languages = multiselect_with_select_all("Languages / Dialects", options=languages_all, key="languages_multiselect")
    selected_models = multiselect_with_select_all("Models", options=models_all, key="models_multiselect")

    filtered = apply_filters(df, tasks=selected_tasks, languages=selected_languages, models=selected_models)

    if filtered.empty:
        st.warning("No rows match your current filters.")
        return

    tasks_in_filtered = sorted(filtered["task"].unique())
    for t in tasks_in_filtered:
        df_t_all = filtered[filtered["task"] == t]
        metrics_for_task = order_metrics(sorted(df_t_all["metric"].unique()))
        if not metrics_for_task:
            continue

        st.markdown(f"Task: {t}")

        for metric in metrics_for_task:
            df_tm = df_t_all[df_t_all["metric"] == metric]
            if df_tm.empty:
                continue

            metric_label = format_metric_name(metric)

            safe_task = sanitize_filename(str(t))
            safe_metric = sanitize_filename(str(metric))
            base = f"{safe_task}__{safe_metric}"

            st.markdown(f"Metric: {metric_label}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Heatmap (models × languages):")
                plot_task_heatmap(df_tm, metric_label)
            with col2:
                st.markdown("Bar plot (languages on X axis, models in legend):")
                plot_task_bar_chart(df_tm, metric_label)

            st.markdown(f"Results table for task {t} ({metric_label})")
            show_metric_results_table(df_tm, t, metric_label)

            st.markdown("---")


if __name__ == "__main__":
    main()
