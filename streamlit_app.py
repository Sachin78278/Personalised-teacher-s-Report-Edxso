import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Group Assessment Analysis", layout="wide")
st.title("📊 Group-Level Assessment Analysis")
st.markdown(
    "Upload a pre-assessment CSV to analyze group-level pre scores. "
    "You can also upload a post-assessment CSV for comparison, but only pre is required. "
    "Participant names are not needed; rows are treated as individual responses."
)

ANSWER_KEYS = {
    "pre_assessment": {
        1: "A", 2: "C", 3: "D", 4: "A", 5: "B", 6: "C", 7: "D", 8: "D", 9: "A", 10: "B", 11: "B", 12: "C"
    },
    "post_assessment": {
        1: "B", 2: "D", 3: "B", 4: "A", 5: "B", 6: "C", 7: "B", 8: "A", 9: "D", 10: "C", 11: "C", 12: "A"
    }
}

QUESTION_LABELS = {
    1: "Question 1",
    2: "Question 2",
    3: "Question 3",
    4: "Question 4",
    5: "Question 5",
    6: "Question 6",
    7: "Question 7",
    8: "Question 8",
    9: "Question 9",
    10: "Question 10",
    11: "Question 11",
    12: "Question 12",
}

VALID_ANSWERS = {"A", "B", "C", "D"}


def normalize_answer(answer):
    if pd.isna(answer):
        return ""
    text = str(answer).strip().upper()
    if not text:
        return ""

    if text[0] in VALID_ANSWERS:
        return text[0]

    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", text).strip()
    if cleaned and cleaned[0] in VALID_ANSWERS:
        return cleaned[0]

    cleaned = cleaned.split()
    if cleaned and cleaned[0] in VALID_ANSWERS:
        return cleaned[0]

    return ""


def detect_answer_columns(df):
    columns = list(df.columns)
    if len(columns) == 12:
        return columns

    def is_answer_column(series):
        values = series.dropna().astype(str).str.strip().str.upper()
        values = values[values != ""]
        if values.empty:
            return False
        sample = values.head(20)
        valid_ratio = sample.isin(VALID_ANSWERS).mean()
        return valid_ratio >= 0.75

    label_match_cols = []
    content_match_cols = []

    # Check for question-related keywords in column names
    question_keywords = ["mental health", "stress", "teacher", "student", "classroom", "assessment", "test", "parent", "school", "emotion", "support", "practice", "warning", "sign", "response", "approach", "action", "interpretation", "step", "culture"]

    for col in columns:
        label = str(col).strip().lower()
        if re.search(r"(^|\W)(q|question)\s*\d{1,2}(\W|$)", label):
            label_match_cols.append(col)
            continue

        # Check if column name contains question keywords
        if any(keyword in label for keyword in question_keywords):
            label_match_cols.append(col)
            continue

        if is_answer_column(df[col]):
            content_match_cols.append(col)

    if len(label_match_cols) >= 12:
        return label_match_cols[:12]
    if len(content_match_cols) >= 12:
        return content_match_cols[:12]
    if len(columns) > 12:
        return columns[-12:]
    return columns


def load_group_dataframe(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Unable to read file: {e}")
        return None

    if df.empty:
        st.error("The uploaded file is empty.")
        return None

    df = df.reset_index(drop=True)
    return df


def extract_scores(df, assessment_type):
    answer_cols = detect_answer_columns(df)
    if not answer_cols or len(answer_cols) < 12:
        st.warning("Could not detect 12 answer columns automatically. Using available columns.")

    answer_cols = list(answer_cols[:12])
    selected = df[answer_cols]
    if isinstance(selected, pd.Series):
        selected = selected.to_frame()

    # Use map for element-wise operations (pandas 2.0+ compatible)
    normalized = selected.map(normalize_answer)
    correct_answers = [ANSWER_KEYS[assessment_type].get(i + 1, "") for i in range(len(answer_cols))]

    scores = normalized.eq(pd.Series(correct_answers, index=selected.columns), axis=1).sum(axis=1)
    question_correct = normalized.eq(pd.Series(correct_answers, index=selected.columns), axis=1).sum(axis=0)
    response_counts = normalized.replace("", np.nan).count(axis=0)

    return {
        "answer_cols": answer_cols,
        "scores": scores,
        "question_correct": question_correct,
        "response_counts": response_counts,
        "normalized": normalized
    }


def create_summary_table(result, label):
    scores = result["scores"]
    data = {
        "Metric": [
            "Participants",
            "Average Score",
            "Median Score",
            "Minimum Score",
            "Maximum Score",
            "% Correct Questions (avg)"
        ],
        "Value": [
            len(scores),
            f"{scores.mean():.2f} / 12",
            f"{scores.median():.1f} / 12",
            f"{scores.min()} / 12",
            f"{scores.max()} / 12",
            f"{(result['question_correct'] / len(scores) * 100).mean():.1f}%"
        ]
    }
    df = pd.DataFrame(data)
    df["Value"] = df["Value"].astype(str)  # Ensure string type for Arrow compatibility
    return df


def plot_score_distribution(scores, title):
    fig = px.histogram(scores, nbins=12, labels={"value": "Score", "count": "Participants"}, title=title)
    fig.update_layout(bargap=0.2)
    return fig


def plot_question_correct_rate(result, title):
    rates = (result["question_correct"] / result["response_counts"].replace(0, np.nan) * 100).fillna(0)
    question_labels = [QUESTION_LABELS.get(i + 1, f"Q{i + 1}") for i in range(len(rates))]
    fig = px.bar(
        x=question_labels,
        y=rates,
        labels={"x": "Question", "y": "% Correct"},
        title=title,
    )
    fig.update_layout(yaxis=dict(range=[0, 100]), xaxis_tickangle=-45)
    return fig


def align_group_improvement(pre_result, post_result):
    pre_scores = pre_result["scores"]
    post_scores = post_result["scores"]
    if len(pre_scores) != len(post_scores):
        return None
    improvement = post_scores.values - pre_scores.values
    return pd.Series(improvement, name="Improvement")


pre_file = st.file_uploader("Upload Pre-Assessment CSV (required)", type=["csv", "xls", "xlsx"], key="group_pre")
post_file = st.file_uploader("Upload Post-Assessment CSV (optional)", type=["csv", "xls", "xlsx"], key="group_post")

if pre_file is None:
    if post_file is None:
        st.info("Upload a pre-assessment file to analyze the pre report. Post assessment is optional.")
    else:
        st.warning("Pre-assessment is required first. Upload the pre file to see group-level pre analysis. Post assessment may be added later for comparison.")
else:
    pre_df = load_group_dataframe(pre_file)
    post_df = load_group_dataframe(post_file) if post_file is not None else None

    pre_result = extract_scores(pre_df, "pre_assessment") if pre_df is not None else None
    post_result = extract_scores(post_df, "post_assessment") if post_df is not None else None

    if pre_result is None and post_result is None:
        st.error("No valid assessment data was loaded.")
    else:
        if pre_result is not None:
            st.subheader("📋 Pre-Assessment Group Summary")
            st.dataframe(create_summary_table(pre_result, "Pre-Assessment"), width='stretch')
            st.plotly_chart(plot_score_distribution(pre_result["scores"], "Pre-Assessment Score Distribution"), use_container_width=True)
            st.plotly_chart(plot_question_correct_rate(pre_result, "Pre-Assessment Question Correct Rate"), use_container_width=True)

        if post_result is not None:
            st.subheader("📋 Post-Assessment Group Summary")
            st.dataframe(create_summary_table(post_result, "Post-Assessment"), width='stretch')
            st.plotly_chart(plot_score_distribution(post_result["scores"], "Post-Assessment Score Distribution"), use_container_width=True)
            st.plotly_chart(plot_question_correct_rate(post_result, "Post-Assessment Question Correct Rate"), use_container_width=True)

        if pre_result is not None and post_result is not None:
            st.markdown("---")
            st.subheader("📈 Group Improvement Analysis")
            if len(pre_result["scores"]) == len(post_result["scores"]):
                improvement = align_group_improvement(pre_result, post_result)
                st.write("Assuming pre- and post-rows correspond by participant order.")
                st.write(f"Average improvement: {improvement.mean():.2f} points")
                st.plotly_chart(plot_score_distribution(improvement, "Row-wise Score Improvement"), use_container_width=True)
                st.dataframe(
                    pd.DataFrame({
                        "Pre Score": pre_result["scores"],
                        "Post Score": post_result["scores"],
                        "Improvement": improvement
                    }).reset_index(drop=True),
                    width='stretch',
                )
            else:
                st.warning(
                    "Pre and post files contain a different number of participant rows. "
                    "Row-wise improvement cannot be calculated accurately. "
                    "Review group-level averages and question-level rates instead."
                )

        st.markdown("---")
        st.markdown(
            "### Notes\n"
            "- Rows are treated as individual responses when participant names are missing.\n"
            "- If the same group completed both assessments in the same order, row-wise improvement is shown when row counts match.\n"
            "- If there are extra columns in the files, the app attempts to detect the 12 answer columns automatically."
        )
