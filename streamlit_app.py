from __future__ import annotations

import os
import hmac
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Анализ советских журналов",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------- Appearance ----------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1.5rem; max-width: 1600px;}
      div[data-testid="stMetric"] {border: 1px solid rgba(49,51,63,.16); border-radius: 8px; padding: 10px 14px;}
      div[data-testid="stDataFrame"] {border: 1px solid rgba(49,51,63,.12); border-radius: 6px;}
      h1, h2, h3 {letter-spacing: -0.02em;}
    </style>
    """,
    unsafe_allow_html=True,
)


REQUIRED_COLUMNS = [
    "publication",
    "dictionary",
    "sentence_id",
    "year",
    "original_sentence",
    "lemmatized_sentence",
    "category",
    "matched_term",
]


def _get_secret(name: str, default=None):
    """Read Streamlit secret first, then environment variable."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def require_shared_password() -> None:
    expected = _get_secret("DASHBOARD_PASSWORD", "")
    if not expected:
        return

    if st.session_state.get("authenticated"):
        return

    st.title("Анализ советских журналов")
    st.caption("Введите пароль проекта")
    entered = st.text_input("Пароль", type="password")
    if st.button("Войти", type="primary"):
        if hmac.compare_digest(entered, str(expected)):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Неверный пароль")
    st.stop()


@st.cache_data(show_spinner="Загрузка данных…")
def load_parquet_data(path: str) -> pd.DataFrame:
    """Load the bundled dataset used by the deployed dashboard."""
    return pd.read_parquet(path)


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Не хватает столбцов: " + ", ".join(missing))

    optional_columns = [c for c in ["emotion", "attitude"] if c in df.columns]
    columns_to_keep = REQUIRED_COLUMNS + optional_columns
    out = df[columns_to_keep].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["sentence_id"] = pd.to_numeric(out["sentence_id"], errors="coerce").astype("Int64")

    # Power BI calculated columns recreated here.
    out["unique_sentence_id"] = (
        out["publication"].fillna("").astype(str)
        + "_"
        + out["sentence_id"].astype("string").fillna("")
    )
    out["period"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[out["year"].notna() & (out["year"] <= 1991), "period"] = "До 1991 включительно"
    out.loc[out["year"].notna() & (out["year"] > 1991), "period"] = "После 1991"

    text_columns = ["publication", "dictionary", "category", "matched_term"]
    for optional_col in ["emotion", "attitude"]:
        if optional_col in out.columns:
            text_columns.append(optional_col)
    for col in text_columns:
        out[col] = out[col].astype("string")
    return out


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    from io import BytesIO

    bio = BytesIO(file_bytes)
    lower = filename.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(bio)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(bio)
    if lower.endswith(".parquet"):
        return pd.read_parquet(bio)
    raise ValueError("Поддерживаются CSV, XLSX и Parquet")


def load_data_file(
    default_name: str,
    secret_name: str,
    uploader_key: str,
) -> pd.DataFrame:
    """Load one dashboard dataset from the app folder, with manual upload fallback."""
    data_file = str(_get_secret(secret_name, default_name))

    if not os.path.isabs(data_file):
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_file)

    if os.path.exists(data_file):
        try:
            return normalize_data(load_parquet_data(data_file))
        except Exception as exc:
            st.error(f"Не удалось прочитать файл данных: {os.path.basename(data_file)}")
            with st.expander("Техническая информация"):
                st.exception(exc)
            st.stop()

    st.warning(
        f"Файл {default_name} не найден рядом со streamlit_app.py. "
        "Добавьте его в репозиторий или загрузите файл вручную ниже."
    )
    uploaded = st.file_uploader(
        f"Файл данных: {default_name}",
        type=["csv", "xlsx", "xls", "parquet"],
        key=uploader_key,
    )
    if uploaded is None:
        return pd.DataFrame()
    return normalize_data(read_uploaded_file(uploaded.getvalue(), uploaded.name))


def load_data() -> pd.DataFrame:
    return load_data_file("all_publications.parquet", "DATA_FILE", "upload_soviet")


def load_us_data() -> pd.DataFrame:
    return load_data_file("all_publications_us.parquet", "DATA_FILE_US", "upload_us")


def sorted_values(series: pd.Series) -> list[str]:
    values = series.dropna().astype(str).unique().tolist()
    return sorted(values, key=lambda x: x.casefold())


def filter_equal(df: pd.DataFrame, col: str, value: str) -> pd.DataFrame:
    if value == "Все":
        return df
    return df[df[col].astype("string") == value]


def filter_multi(df: pd.DataFrame, col: str, values: Iterable[str]) -> pd.DataFrame:
    values = list(values)
    if not values:
        return df
    return df[df[col].astype("string").isin(values)]


def year_bounds(df: pd.DataFrame) -> tuple[int, int]:
    years = df["year"].dropna().astype(int)
    if years.empty:
        return (1945, 2000)
    return int(years.min()), int(years.max())


def apply_common_filters(
    df: pd.DataFrame,
    dictionary: str,
    publication: str,
    period: str,
    years: tuple[int, int],
) -> pd.DataFrame:
    out = filter_equal(df, "dictionary", dictionary)
    out = filter_equal(out, "publication", publication)
    out = filter_equal(out, "period", period)
    out = out[out["year"].notna()]
    out = out[(out["year"].astype(int) >= years[0]) & (out["year"].astype(int) <= years[1])]
    return out


def page_texts(df: pd.DataFrame, key_prefix: str = "txt", title: str = "Тексты") -> None:
    st.subheader(title)

    # First layer: dictionary/publication/period/year. Category and matched term cascade from it.
    c1, c2, c3, c4, c5, c6 = st.columns([1.0, 1.15, 1.15, 1.0, 0.85, 1.55])

    dictionaries = ["Все"] + sorted_values(df["dictionary"])
    with c1:
        dictionary = st.selectbox("Словарь", dictionaries, key=f"{key_prefix}_dictionary")

    base = filter_equal(df, "dictionary", dictionary)

    publications = ["Все"] + sorted_values(base["publication"])
    with c4:
        publication = st.selectbox("Журнал", publications, key=f"{key_prefix}_publication")
    base = filter_equal(base, "publication", publication)

    periods = ["Все", "До 1991 включительно", "После 1991"]
    with c5:
        period = st.selectbox("Период", periods, key=f"{key_prefix}_period")
    base = filter_equal(base, "period", period)

    ymin, ymax = year_bounds(base if not base.empty else df)
    with c6:
        years = st.slider("Год", ymin, ymax, (ymin, ymax), key=f"{key_prefix}_years")
    base = base[base["year"].notna()]
    base = base[(base["year"].astype(int) >= years[0]) & (base["year"].astype(int) <= years[1])]

    categories = ["Все"] + sorted_values(base["category"])
    with c2:
        category = st.selectbox("Категория", categories, key=f"{key_prefix}_category")
    base2 = filter_equal(base, "category", category)

    terms = ["Все"] + sorted_values(base2["matched_term"])
    with c3:
        matched_term = st.selectbox("Совпадающий термин", terms, key=f"{key_prefix}_term")

    filtered = filter_equal(base2, "matched_term", matched_term)
    count = filtered["unique_sentence_id"].nunique(dropna=True)

    m1, _ = st.columns([1, 5])
    with m1:
        st.metric("Найдено предложений", f"{count:,}".replace(",", " "))

    display = filtered[
        ["publication", "year", "sentence_id", "category", "matched_term", "original_sentence"]
    ].rename(
        columns={
            "publication": "Издание",
            "year": "Год",
            "sentence_id": "ID предложения",
            "category": "Категория",
            "matched_term": "Совпадающий термин",
            "original_sentence": "Оригинальный текст",
        }
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=610,
        column_config={
            "Издание": st.column_config.TextColumn(width="small"),
            "Год": st.column_config.NumberColumn(format="%d", width="small"),
            "ID предложения": st.column_config.NumberColumn(format="%d", width="small"),
            "Категория": st.column_config.TextColumn(width="medium"),
            "Совпадающий термин": st.column_config.TextColumn(width="medium"),
            "Оригинальный текст": st.column_config.TextColumn(width="large"),
        },
    )


def page_analytics(df: pd.DataFrame, key_prefix: str = "an", title: str = "Анализ категорий") -> None:
    st.subheader(title)

    c1, c2, c3, c4 = st.columns([1.1, 1.1, 0.9, 1.6])

    dictionaries = sorted_values(df["dictionary"])
    if not dictionaries:
        st.warning("Нет значений dictionary")
        return
    with c1:
        dictionary = st.selectbox("Словарь", dictionaries, key=f"{key_prefix}_dictionary")

    base = filter_equal(df, "dictionary", dictionary)

    publications = sorted_values(base["publication"])
    with c2:
        selected_publications = st.multiselect(
            "Журналы",
            options=publications,
            default=[],
            key=f"{key_prefix}_publications",
            placeholder="Все издания",
            help="Можно выбрать несколько изданий. Если ничего не выбрано, используются все издания.",
        )
    base = filter_multi(base, "publication", selected_publications)

    with c3:
        period = st.selectbox(
            "Период",
            ["Все", "До 1991 включительно", "После 1991"],
            key=f"{key_prefix}_period",
        )
    base = filter_equal(base, "period", period)

    ymin, ymax = year_bounds(base if not base.empty else df)
    with c4:
        years = st.slider("Год", ymin, ymax, (ymin, ymax), key=f"{key_prefix}_years")

    base = base[base["year"].notna()]
    base = base[(base["year"].astype(int) >= years[0]) & (base["year"].astype(int) <= years[1])]

    if base.empty:
        st.info("По выбранным фильтрам данных нет.")
        return

    # Descriptive statistics
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Предложения", f"{base['unique_sentence_id'].nunique():,}".replace(",", " "))
    s2.metric("Категории", f"{base['category'].nunique(dropna=True):,}".replace(",", " "))
    s3.metric("Термины", f"{base['matched_term'].nunique(dropna=True):,}".replace(",", " "))
    s4.metric("Издания", f"{base['publication'].nunique(dropna=True):,}".replace(",", " "))
    yr = base["year"].dropna().astype(int)
    s5.metric("Диапазон лет", f"{yr.min()}–{yr.max()}" if not yr.empty else "—")

    # Counts by category, distinct sentences (same logic as DAX Sentences Count).
    cat_counts = (
        base.dropna(subset=["category"])
        .groupby("category", as_index=False)["unique_sentence_id"]
        .nunique()
        .rename(columns={"unique_sentence_id": "count"})
        .sort_values("count", ascending=False)
    )
    top10 = cat_counts.head(10).sort_values("count", ascending=True)

    left, right = st.columns([1.0, 2.35])

    with left:
        st.markdown("### Топ-10 категорий")
        fig_bar = px.bar(
            top10,
            x="count",
            y="category",
            orientation="h",
            labels={"count": "Количество предложений", "category": ""},
        )
        fig_bar.update_layout(height=315, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        available_categories = cat_counts["category"].tolist()
        default_categories = available_categories[: min(8, len(available_categories))]
        selected_categories = st.multiselect(
            "Категории для динамики и долей",
            options=available_categories,
            default=default_categories,
            key=f"{key_prefix}_categories",
            help="Можно выбрать несколько категорий. Топ-10 выше от этого выбора не меняется.",
        )

        pie_data = cat_counts
        if selected_categories:
            pie_data = pie_data[pie_data["category"].isin(selected_categories)]
        else:
            pie_data = pie_data.head(10)

        st.markdown("### Доли категорий")
        fig_pie = px.pie(
            pie_data,
            names="category",
            values="count",
            hole=0,
        )
        fig_pie.update_traces(textinfo="percent", hovertemplate="%{label}<br>%{value} предложений<br>%{percent}<extra></extra>")
        fig_pie.update_layout(height=340, margin=dict(l=5, r=5, t=5, b=5), legend_title_text="Категория")
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown("### Распределение категорий по годам")
        line_base = base
        if selected_categories:
            line_base = line_base[line_base["category"].isin(selected_categories)]

        line_data = (
            line_base.dropna(subset=["year", "category"])
            .groupby(["year", "category"], as_index=False)["unique_sentence_id"]
            .nunique()
            .rename(columns={"unique_sentence_id": "count"})
            .sort_values(["year", "count"], ascending=[True, False])
        )

        fig_line = px.line(
            line_data,
            x="year",
            y="count",
            color="category",
            markers=False,
            labels={"year": "Год", "count": "Количество упоминаний", "category": "Категория"},
        )
        # Компактная unified-подсказка: сверху остается только год,
        # а для каждой линии показывается только категория и значение Y.
        # Plotly не поддерживает прокрутку внутри стандартного hover tooltip,
        # поэтому убираем повторяющиеся поля/фильтры, чтобы список был максимально компактным.
        fig_line.update_traces(
            hovertemplate="%{fullData.name}: %{y}<extra></extra>"
        )
        fig_line.update_layout(
            height=650,
            hovermode="x unified",
            margin=dict(l=15, r=15, t=10, b=10),
            legend_title_text="Категория",
        )
        fig_line.update_xaxes(dtick=5)
        st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": True})

        st.caption("Количество считается по уникальным сочетаниям «издание + sentence_id», как мера Sentences Count в исходном Power BI.")



def page_publication_analytics(df: pd.DataFrame) -> None:
    st.subheader("Аналитика изданий")

    c1, c2, c3, c4, c5 = st.columns([0.95, 1.8, 1.5, 0.9, 1.45])

    dimension_labels = {
        "Словарь": "dictionary",
        "Категория": "category",
        "Термин": "matched_term",
    }
    with c1:
        dimension_label = st.selectbox(
            "Что анализируем",
            options=list(dimension_labels),
            key="pub_dimension",
        )
    dimension = dimension_labels[dimension_label]

    available_values = sorted_values(df[dimension])
    with c2:
        selected_values = st.multiselect(
            dimension_label,
            options=available_values,
            default=[],
            key="pub_values",
            placeholder=f"Все: {dimension_label.lower()}",
            help=(
                "Можно выбрать одно или несколько значений. "
                "Если ничего не выбрано, используются все значения выбранного уровня."
            ),
        )

    base = filter_multi(df, dimension, selected_values)

    publications = sorted_values(base["publication"])
    with c3:
        selected_publications = st.multiselect(
            "Издания",
            options=publications,
            default=[],
            key="pub_publications",
            placeholder="Все издания",
            help="Можно выбрать одно или несколько изданий. Если ничего не выбрано, используются все издания.",
        )
    base = filter_multi(base, "publication", selected_publications)

    with c4:
        period = st.selectbox(
            "Период",
            ["Все", "До 1991 включительно", "После 1991"],
            key="pub_period",
        )
    base = filter_equal(base, "period", period)

    ymin, ymax = year_bounds(base if not base.empty else df)
    with c5:
        years = st.slider("Год", ymin, ymax, (ymin, ymax), key="pub_years")

    base = base[base["year"].notna()].copy()
    base = base[(base["year"].astype(int) >= years[0]) & (base["year"].astype(int) <= years[1])]

    if base.empty:
        st.info("По выбранным фильтрам данных нет.")
        return

    # One bar per publication: number of distinct sentences containing any selected value.
    bar_data = (
        base.groupby("publication", as_index=False)["unique_sentence_id"]
        .nunique()
        .rename(columns={"unique_sentence_id": "count"})
        .sort_values("count", ascending=False)
    )

    st.markdown("### Частота упоминаний по изданиям")
    fig_bar = px.bar(
        bar_data,
        x="publication",
        y="count",
        labels={"publication": "Издание", "count": "Количество упоминаний"},
        text_auto=True,
    )
    fig_bar.update_traces(hovertemplate="%{x}: %{y}<extra></extra>")
    fig_bar.update_layout(
        height=430,
        margin=dict(l=15, r=15, t=10, b=10),
        showlegend=False,
    )
    fig_bar.update_xaxes(categoryorder="total descending")
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # Yearly dynamics with publications as lines. Fill missing publication/year combinations with zeros
    # so the chart does not connect non-adjacent observations across years with no mentions.
    line_counts = (
        base.groupby(["year", "publication"], as_index=False)["unique_sentence_id"]
        .nunique()
        .rename(columns={"unique_sentence_id": "count"})
    )

    active_publications = bar_data["publication"].astype(str).tolist()
    all_years = list(range(years[0], years[1] + 1))
    full_index = pd.MultiIndex.from_product(
        [all_years, active_publications],
        names=["year", "publication"],
    )
    line_data = (
        line_counts.assign(
            year=line_counts["year"].astype(int),
            publication=line_counts["publication"].astype(str),
        )
        .set_index(["year", "publication"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values(["publication", "year"])
    )

    st.markdown("### Динамика упоминаний по годам")
    fig_line = px.line(
        line_data,
        x="year",
        y="count",
        color="publication",
        markers=False,
        labels={"year": "Год", "count": "Количество упоминаний", "publication": "Издание"},
    )
    # Plotly does not dynamically sort entries in a standard unified hover tooltip.
    # Hide the native per-trace hover rows and add one invisible helper trace whose
    # tooltip is pre-sorted by count (descending) separately for every year.
    fig_line.update_traces(hoverinfo="skip")

    hover_rows = []
    hover_y = []
    for year in all_years:
        year_data = (
            line_data[line_data["year"] == year][["publication", "count"]]
            .sort_values(["count", "publication"], ascending=[False, True])
        )
        hover_rows.append(
            "<br>".join(
                f"{publication}: {int(count)}"
                for publication, count in year_data.itertuples(index=False, name=None)
            )
        )
        hover_y.append(int(year_data["count"].max()) if not year_data.empty else 0)

    fig_line.add_scatter(
        x=all_years,
        y=hover_y,
        mode="markers",
        marker=dict(size=10, opacity=0),
        customdata=hover_rows,
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
        name="",
    )

    fig_line.update_layout(
        height=650,
        hovermode="x unified",
        margin=dict(l=15, r=15, t=10, b=10),
        legend_title_text="Издание",
    )
    fig_line.update_xaxes(dtick=5)
    st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": True})

    selection_text = ", ".join(selected_values) if selected_values else "все значения"
    st.caption(
        f"Уровень анализа: {dimension_label.lower()}; выбрано: {selection_text}. "
        "Количество считается по уникальным сочетаниям «издание + sentence_id»."
    )



def page_attitudes(df: pd.DataFrame) -> None:
    st.subheader("Отношение")

    if "emotion" not in df.columns:
        st.warning(
            "В данных нет колонки emotion. Добавьте её в исходный CSV/XLSX/Parquet, "
            "чтобы построить распределение отношений."
        )
        return

    target_dictionaries = [
        "Отношение американцев к СССР/России",
        "Отношение к США",
    ]

    available_target_dictionaries = set(df["dictionary"].dropna().astype(str).unique())
    missing_dictionaries = [
        dictionary for dictionary in target_dictionaries
        if dictionary not in available_target_dictionaries
    ]
    if missing_dictionaries:
        st.warning(
            "В данных не найдены словари: " + ", ".join(missing_dictionaries)
        )

    relation_base = df[df["dictionary"].astype("string").isin(target_dictionaries)].copy()
    if relation_base.empty:
        st.info("Для двух выбранных словарей данных нет.")
        return

    c1, c2, c3 = st.columns([1.6, 0.9, 1.5])

    publications = sorted_values(relation_base["publication"])
    with c1:
        selected_publications = st.multiselect(
            "Издания",
            options=publications,
            default=[],
            key="rel_publications",
            placeholder="Все издания",
            help="Можно выбрать одно или несколько изданий. Если ничего не выбрано, используются все издания.",
        )
    relation_base = filter_multi(relation_base, "publication", selected_publications)

    with c2:
        period = st.selectbox(
            "Период",
            ["Все", "До 1991 включительно", "После 1991"],
            key="rel_period",
        )
    relation_base = filter_equal(relation_base, "period", period)

    ymin, ymax = year_bounds(relation_base if not relation_base.empty else df)
    with c3:
        years = st.slider("Год", ymin, ymax, (ymin, ymax), key="rel_years")

    relation_base = relation_base[relation_base["year"].notna()].copy()
    relation_base = relation_base[
        (relation_base["year"].astype(int) >= years[0])
        & (relation_base["year"].astype(int) <= years[1])
    ]
    relation_base = relation_base.dropna(subset=["emotion"])
    relation_base = relation_base[relation_base["emotion"].astype(str).str.strip() != ""]

    if relation_base.empty:
        st.info("По выбранным фильтрам данных нет.")
        return

    left, right = st.columns(2)

    def render_emotion_pie(container, dictionary: str, title: str) -> None:
        subset = relation_base[
            relation_base["dictionary"].astype("string") == dictionary
        ]

        with container:
            st.markdown(f"### {title}")
            if subset.empty:
                st.info("По выбранным фильтрам данных нет.")
                return

            pie_data = (
                subset.groupby("emotion", as_index=False)["unique_sentence_id"]
                .nunique()
                .rename(columns={"unique_sentence_id": "count"})
                .sort_values("count", ascending=False)
            )

            fig = px.pie(
                pie_data,
                names="emotion",
                values="count",
                hole=0,
            )
            fig.update_traces(
                textinfo="percent+label",
                hovertemplate=(
                    "%{label}<br>"
                    "%{value} предложений<br>"
                    "%{percent}<extra></extra>"
                ),
            )
            fig.update_layout(
                height=520,
                margin=dict(l=5, r=5, t=10, b=5),
                legend_title_text="Отношение",
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

            total = subset["unique_sentence_id"].nunique(dropna=True)
            st.caption(
                f"Уникальных предложений: {total:,}".replace(",", " ")
            )

    render_emotion_pie(
        left,
        "Отношение американцев к СССР/России",
        "Отношение американцев к СССР/России",
    )
    render_emotion_pie(
        right,
        "Отношение к США",
        "Отношение к США",
    )

    st.caption(
        "Доли рассчитываются по уникальным сочетаниям «издание + sentence_id» внутри каждой категории emotion."
    )


def page_us_emotions(df: pd.DataFrame) -> None:
    st.subheader("Эмоции и отношение")

    if df.empty:
        st.info("Данные американских публикаций не загружены.")
        return

    author_dictionary = "american_author_emotions_to_russia"
    imagined_dictionary = "imagined_russian_attitudes_to_usa"

    available = set(df["dictionary"].dropna().astype(str).unique())
    missing = [
        name for name in [author_dictionary, imagined_dictionary]
        if name not in available
    ]
    if missing:
        st.warning("В данных не найдены словари: " + ", ".join(missing))

    relation_base = df[
        df["dictionary"].astype("string").isin(
            [author_dictionary, imagined_dictionary]
        )
    ].copy()

    if relation_base.empty:
        st.info("Для словарей с эмоциями и отношением данных нет.")
        return

    c1, c2, c3 = st.columns([1.6, 0.9, 1.5])

    publications = sorted_values(relation_base["publication"])
    with c1:
        selected_publications = st.multiselect(
            "Издания",
            options=publications,
            default=[],
            key="us_em_publications",
            placeholder="Все издания",
            help="Если ничего не выбрано, используются все издания.",
        )
    relation_base = filter_multi(relation_base, "publication", selected_publications)

    with c2:
        period = st.selectbox(
            "Период",
            ["Все", "До 1991 включительно", "После 1991"],
            key="us_em_period",
        )
    relation_base = filter_equal(relation_base, "period", period)

    ymin, ymax = year_bounds(relation_base if not relation_base.empty else df)
    with c3:
        years = st.slider("Год", ymin, ymax, (ymin, ymax), key="us_em_years")

    relation_base = relation_base[relation_base["year"].notna()].copy()
    relation_base = relation_base[
        (relation_base["year"].astype(int) >= years[0])
        & (relation_base["year"].astype(int) <= years[1])
    ]

    if relation_base.empty:
        st.info("По выбранным фильтрам данных нет.")
        return

    left, right = st.columns(2)

    def render_pie(
        container,
        dictionary: str,
        value_column: str,
        title: str,
        legend_title: str,
    ) -> None:
        with container:
            st.markdown(f"### {title}")

            if value_column not in relation_base.columns:
                st.warning(f"В данных нет колонки {value_column}.")
                return

            subset = relation_base[
                relation_base["dictionary"].astype("string") == dictionary
            ].copy()
            subset = subset.dropna(subset=[value_column])
            subset = subset[subset[value_column].astype(str).str.strip() != ""]

            if subset.empty:
                st.info("По выбранным фильтрам данных нет.")
                return

            pie_data = (
                subset.groupby(value_column, as_index=False)["unique_sentence_id"]
                .nunique()
                .rename(columns={"unique_sentence_id": "count"})
                .sort_values("count", ascending=False)
            )

            fig = px.pie(
                pie_data,
                names=value_column,
                values="count",
                hole=0,
            )
            fig.update_traces(
                textinfo="percent+label",
                hovertemplate=(
                    "%{label}<br>"
                    "%{value} предложений<br>"
                    "%{percent}<extra></extra>"
                ),
            )
            fig.update_layout(
                height=520,
                margin=dict(l=5, r=5, t=10, b=5),
                legend_title_text=legend_title,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

            total = subset["unique_sentence_id"].nunique(dropna=True)
            st.caption(f"Уникальных предложений: {total:,}".replace(",", " "))

    render_pie(
        left,
        author_dictionary,
        "emotion",
        "Эмоции американских авторов по отношению к СССР/России",
        "Эмоция",
    )
    render_pie(
        right,
        imagined_dictionary,
        "attitude",
        "Воображаемое отношение русских к США",
        "Отношение",
    )

    st.caption(
        "Доли рассчитываются по уникальным сочетаниям «издание + sentence_id» "
        "отдельно для каждого словаря."
    )

def main() -> None:
    require_shared_password()
    df = load_data()
    df_us = load_us_data()

    st.title("Анализ советских и американских журналов")

    section_soviet, section_us = st.tabs(
        ["Советские публикации", "Американские публикации"]
    )

    with section_soviet:
        tab_texts, tab_analytics, tab_publications, tab_attitudes = st.tabs(
            ["Тексты", "Анализ категорий", "Аналитика изданий", "Отношение"]
        )
        with tab_texts:
            page_texts(df, key_prefix="txt", title="Тексты")
        with tab_analytics:
            page_analytics(df, key_prefix="an", title="Анализ категорий")
        with tab_publications:
            page_publication_analytics(df)
        with tab_attitudes:
            page_attitudes(df)

    with section_us:
        if df_us.empty:
            st.info(
                "Добавьте all_publications_us.parquet рядом со streamlit_app.py "
                "или загрузите его через форму выше."
            )
        else:
            us_texts, us_analytics, us_emotions = st.tabs(
                ["Тексты", "Анализ категорий", "Эмоции"]
            )
            with us_texts:
                page_texts(df_us, key_prefix="us_txt", title="Тексты американских публикаций")
            with us_analytics:
                page_analytics(df_us, key_prefix="us_an", title="Анализ категорий американских публикаций")
            with us_emotions:
                page_us_emotions(df_us)


if __name__ == "__main__":
    main()
