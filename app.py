from __future__ import annotations

import pandas as pd
import streamlit as st

from predictor import DATA_PATH, MODEL_PATH, load_dataset, load_model, recommend_colleges


st.set_page_config(
    page_title="JEE College Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #182b49;
        --body: #30445f;
        --muted: #52657d;
        --paper: #f5f8fc;
        --line: #d5e0ec;
        --blue: #176b9b;
        --teal: #087f78;
        --coral: #c84d38;
        --gold: #a86a08;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 92% 3%, rgba(255, 215, 153, 0.25), transparent 30rem),
            radial-gradient(ellipse at 4% 24%, rgba(126, 207, 220, 0.18), transparent 32rem),
            var(--paper);
        color: var(--body);
    }
    [data-testid="stAppViewContainer"] { color: var(--body); }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li { color: var(--body); }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    .source-line { color: var(--muted) !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #e4f4ff 0%, #eef0ff 54%, #fff0e4 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] h2 { color: #155b83; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #344b67 !important; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    h1 { font-family: Georgia, "Times New Roman", serif; font-size: 2.55rem; }
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: #243b58 !important;
        font-weight: 650;
    }
    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div {
        background: #fff !important;
        border-color: #9fb4ca !important;
    }
    [data-baseweb="input"] input,
    [data-baseweb="select"] *,
    [data-baseweb="popover"] * { color: #203650 !important; }
    [data-baseweb="input"] input::placeholder { color: #667b91 !important; opacity: 1; }
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid #d5e3ef;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(27, 62, 94, 0.07);
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #fff 20%, #edf7ff 100%);
        border: 1px solid #cbdceb;
        border-top: 4px solid var(--blue);
        border-radius: 7px;
        padding: 14px 16px;
        box-shadow: 0 5px 16px rgba(27, 62, 94, 0.06);
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) [data-testid="stMetric"] {
        background: linear-gradient(145deg, #fff 20%, #e9f8f2 100%);
        border-top-color: var(--teal);
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3) [data-testid="stMetric"] {
        background: linear-gradient(145deg, #fff 20%, #fff3e7 100%);
        border-top-color: var(--coral);
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: #435a74 !important; }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: var(--ink) !important; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(100deg, #176b9b, #087f78);
        color: #fff;
        border: 1px solid #176b9b;
        border-radius: 6px;
        min-height: 2.8rem;
        font-weight: 650;
        box-shadow: 0 5px 12px rgba(23, 107, 155, 0.18);
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(100deg, #12567d, #066760);
        border-color: #12567d;
        color: #fff;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 5px; }
    .eyebrow {
        color: var(--coral);
        font-size: 0.75rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .notice {
        border-left: 4px solid var(--gold);
        background: linear-gradient(100deg, #fff4d9, #fff9ec);
        padding: 0.8rem 1rem;
        color: #533b13;
        border-radius: 0 4px 4px 0;
        box-shadow: 0 4px 14px rgba(112, 80, 20, 0.07);
    }
    .notice strong { color: #704b08; }
    .source-line { font-size: 0.88rem; }
    @media (max-width: 700px) {
        h1 { font-size: 2rem; }
        [data-testid="stMetric"] { padding: 10px 12px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained college model...")
def cached_model():
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner="Loading historical JoSAA records...")
def cached_dataset() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


st.markdown('<div class="eyebrow">JoSAA historical data · 2021–2024</div>', unsafe_allow_html=True)
st.title("JEE College Predictor")
st.markdown(
    "Explore programs whose model-estimated closing rank may include your JEE rank, "
    "using real JoSAA counselling records."
)
st.markdown(
    '<div class="notice"><strong>Estimate, not admission advice.</strong> '
    "Results are model estimates based on historical records, not a guarantee of admission "
    "or a probability of admission.</div>",
    unsafe_allow_html=True,
)

try:
    dataset = cached_dataset()
    model = cached_model()
except Exception as exc:
    st.error(f"The application could not load its required project data or model: {exc}")
    st.caption("Check that the CSV and model artifact are present beside app.py and that dependencies are installed.")
    st.stop()

years = sorted(int(year) for year in dataset["year"].unique())
with st.sidebar:
    st.header("Your counselling profile")
    selected_year = st.selectbox(
        "Historical year",
        options=years,
        index=len(years) - 1,
        help="Only years present in the repository's historical dataset are available.",
    )
    available_rounds = sorted(
        int(round_no)
        for round_no in dataset.loc[dataset["year"] == selected_year, "round"].unique()
    )

    with st.form("prediction_form"):
        rank = st.number_input(
            "JEE rank",
            min_value=1,
            value=None,
            step=1,
            placeholder="Enter your rank",
            help="Use your rank number, not your percentile.",
        )
        selected_round = st.selectbox("Counselling round", options=available_rounds)
        first_row, second_row = st.columns(2)
        with first_row:
            selected_quota = st.selectbox("Quota", options=sorted(dataset["quota"].unique()))
            selected_seat_type = st.selectbox("Seat type", options=sorted(dataset["seat_type"].unique()))
        with second_row:
            selected_gender = st.selectbox(
                "JoSAA seat gender category",
                options=sorted(dataset["gender"].unique()),
                help=(
                    "The source data has Gender-Neutral and Female-only seat pools. "
                    "Gender-Neutral includes male applicants; there is no separate Male category."
                ),
            )
        st.caption(
            "Male applicants should select Gender-Neutral; the historical data has no separate Male seat pool."
        )
        selected_institutes = st.multiselect(
            "Institutes (optional)",
            options=sorted(dataset["institute"].unique()),
            help="Leave blank to include every institute available for this selection.",
        )
        selected_programs = st.multiselect(
            "Programs (optional)",
            options=sorted(dataset["academic_program_name"].unique()),
            help="Leave blank to include every program available for this selection.",
        )
        submitted = st.form_submit_button("Find matching programs", width="stretch")

    st.caption("Historical years and categories are limited to records in the included CSV.")

if submitted:
    if rank is None:
        st.sidebar.error("Enter a JEE rank greater than zero to continue.")
    else:
        try:
            results = recommend_colleges(
                dataset,
                model,
                user_rank=rank,
                year=selected_year,
                round_no=selected_round,
                seat_type=selected_seat_type,
                gender=selected_gender,
                quota=selected_quota,
                institutes=selected_institutes,
                programs=selected_programs,
            )
        except ValueError as exc:
            st.sidebar.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")
        else:
            st.divider()
            st.subheader("Your results")
            st.caption(
                f"Rank {int(rank):,} · {selected_year}, round {selected_round} · "
                f"{selected_quota} quota · {selected_seat_type} · {selected_gender}"
            )

            if results.empty:
                st.warning(
                    "No historical records match this combination of year, round, quota, "
                    "seat type, gender, and optional preferences. Try a broader selection."
                )
            else:
                eligible = results[results["rank_eligible_by_estimate"]].copy()
                if eligible.empty:
                    st.info(
                        "No rows meet your rank under the model's estimated closing ranks. "
                        "The closest historical programs are shown for comparison; they are "
                        "not rank-eligible matches."
                    )
                    displayed = results.sort_values("rank_difference").head(20)
                else:
                    st.success(
                        f"{len(eligible):,} historical program records have a model-estimated "
                        "closing rank at or beyond your entered rank."
                    )
                    displayed = eligible.sort_values("predicted_closing_rank")

                summary_columns = st.columns(3)
                summary_columns[0].metric("Rows displayed", f"{len(displayed):,}")
                summary_columns[1].metric("Historical records checked", f"{len(results):,}")
                summary_columns[2].metric("Source year", str(selected_year))

                table = displayed[
                    [
                        "institute",
                        "academic_program_name",
                        "predicted_closing_rank",
                        "closing_rank",
                        "opening_rank",
                        "year",
                        "round",
                        "quota",
                        "seat_type",
                        "gender",
                    ]
                ].rename(
                    columns={
                        "institute": "Institute",
                        "academic_program_name": "Program",
                        "predicted_closing_rank": "Predicted closing rank (estimate)",
                        "closing_rank": "Historical closing rank",
                        "opening_rank": "Historical opening rank",
                        "year": "Year",
                        "round": "Round",
                        "quota": "Quota",
                        "seat_type": "Seat type",
                        "gender": "Gender category",
                    }
                )
                st.dataframe(
                    table,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Predicted closing rank (estimate)": st.column_config.NumberColumn(format="%.0f"),
                        "Historical closing rank": st.column_config.NumberColumn(format="%d"),
                        "Historical opening rank": st.column_config.NumberColumn(format="%d"),
                    },
                )
                st.caption(
                    "Historical opening and closing ranks are the recorded cutoffs for the selected "
                    "year and round. The predicted rank is the model's estimate for that same row."
                )
else:
    st.divider()
    overview_columns = st.columns(3)
    overview_columns[0].metric("Historical records", f"{len(dataset):,}")
    overview_columns[1].metric("Institutes", f"{dataset['institute'].nunique():,}")
    overview_columns[2].metric("Programs", f"{dataset['academic_program_name'].nunique():,}")
    st.markdown("### Historical coverage")
    st.markdown(
        f"The included file contains records from **{min(years)} to {max(years)}**. "
        "Choose a year and counselling round in the sidebar, then enter your rank to view "
        "the model estimates alongside the observed cutoffs."
    )

st.markdown(
    '<p class="source-line">Source: repository-provided JoSAA CSV and CatBoost model. '
    "The CSV filename mentions 2019–2024, but its records cover 2021–2024. "
    "The repository does not include external source provenance or a current-year forecast.</p>",
    unsafe_allow_html=True,
)