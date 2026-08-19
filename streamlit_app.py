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

    out = df[REQUIRED_COLUMNS].copy()
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

    for col in ["publication", "dictionary", "category", "matched_term"]:
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


def load_data() -> pd.DataFrame:
    """
    Cloud-friendly data loading.

    By default the app expects `all_publications.parquet` next to streamlit_app.py.
    The filename can be changed with DATA_FILE in Streamlit secrets/environment.
    If the file is absent, the app offers a manual upload as a fallback.
    """
    data_file = str(_get_secret("DATA_FILE", "all_publications.parquet"))

    # Resolve relative paths from the folder containing this script, not from cwd.
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
        "Файл all_publications.parquet не найден рядом со streamlit_app.py. "
        "Добавьте его в репозиторий или загрузите файл вручную ниже."
    )
    uploaded = st.file_uploader("Файл данных", type=["csv", "xlsx", "xls", "parquet"])
    if uploaded is None:
        st.stop()
    return normalize_data(read_uploaded_file(uploaded.getvalue(), uploaded.name))


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


def page_texts(df: pd.DataFrame) -> None:
    st.subheader("Тексты")

    # First layer: dictionary/publication/period/year. Category and matched term cascade from it.
    c1, c2, c3, c4, c5, c6 = st.columns([1.0, 1.15, 1.15, 1.0, 0.85, 1.55])

    dictionaries = ["Все"] + sorted_values(df["dictionary"])
    with c1:
        dictionary = st.selectbox("Словарь", dictionaries, key="txt_dictionary")

    base = filter_equal(df, "dictionary", dictionary)

    publications = ["Все"] + sorted_values(base["publication"])
    with c4:
        publication = st.selectbox("Журнал", publications, key="txt_publication")
    base = filter_equal(base, "publication", publication)

    periods = ["Все", "До 1991 включительно", "После 1991"]
    with c5:
        period = st.selectbox("Период", periods, key="txt_period")
    base = filter_equal(base, "period", period)

    ymin, ymax = year_bounds(base if not base.empty else df)
    with c6:
        years = st.slider("Год", ymin, ymax, (ymin, ymax), key="txt_years")
    base = base[base["year"].notna()]
    base = base[(base["year"].astype(int) >= years[0]) & (base["year"].astype(int) <= years[1])]

    categories = ["Все"] + sorted_values(base["category"])
    with c2:
        category = st.selectbox("Категория", categories, key="txt_category")
    base2 = filter_equal(base, "category", category)

    terms = ["Все"] + sorted_values(base2["matched_term"])
    with c3:
        matched_term = st.selectbox("Совпадающий термин", terms, key="txt_term")

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


def page_analytics(df: pd.DataFrame) -> None:
    st.subheader("Аналитика")

    c1, c2, c3, c4 = st.columns([1.1, 1.1, 0.9, 1.6])

    dictionaries = sorted_values(df["dictionary"])
    if not dictionaries:
        st.warning("Нет значений dictionary")
        return
    with c1:
        dictionary = st.selectbox("Словарь", dictionaries, key="an_dictionary")

    base = filter_equal(df, "dictionary", dictionary)

    publications = ["Все"] + sorted_values(base["publication"])
    with c2:
        publication = st.selectbox("Журнал", publications, key="an_publication")
    base = filter_equal(base, "publication", publication)

    with c3:
        period = st.selectbox(
            "Период",
            ["Все", "До 1991 включительно", "После 1991"],
            key="an_period",
        )
    base = filter_equal(base, "period", period)

    ymin, ymax = year_bounds(base if not base.empty else df)
    with c4:
        years = st.slider("Год", ymin, ymax, (ymin, ymax), key="an_years")

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
            key="an_categories",
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


def main() -> None:
    require_shared_password()
    df = load_data()

    st.title("Анализ советских журналов")
    st.caption("Перенесено из Power BI: страницы «Тексты» и «Аналитика», каскадные фильтры и основные визуализации.")

    tab_texts, tab_analytics = st.tabs(["Тексты", "Аналитика"])
    with tab_texts:
        page_texts(df)
    with tab_analytics:
        page_analytics(df)


if __name__ == "__main__":
    main()
