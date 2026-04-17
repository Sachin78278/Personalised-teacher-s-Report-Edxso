import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import re
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
from urllib.parse import quote_plus, unquote_plus
import numpy as np

# Configure Streamlit page
st.set_page_config(page_title="Teacher Assessment Analysis with Grading", layout="wide")
st.title("📊 Teacher Assessment Analysis & Grading Report")
st.markdown("Compare pre and post-assessment responses with automatic grading and improvement analysis")

# Answer Keys
ANSWER_KEYS = {
    "pre_assessment": {
        1: "A", 2: "C", 3: "D", 4: "A", 5: "B", 6: "C", 7: "D", 8: "D", 9: "A", 10: "B", 11: "B", 12: "C"
    },
    "post_assessment": {
        1: "B", 2: "D", 3: "B", 4: "A", 5: "B", 6: "C", 7: "B", 8: "A", 9: "D", 10: "C", 11: "C", 12: "A"
    }
}

QUESTIONS_MAPPING = {
    "pre_assessment": {
        1: "Mental health is best understood as:",
        2: "Which is a common source of stress for students today?",
        3: "Which is the clearest early warning sign a teacher may observe?",
        4: "A student who is usually cheerful has become quiet, avoids friends, and has stopped submitting work. What should the teacher do first?",
        5: "Which classroom practice is most likely to improve emotional safety?",
        6: "Before a test, a student says, 'I know I will not do well.' What is the most helpful immediate teacher response?",
        7: "Which approach is most appropriate while talking to parents about a concern?",
        8: "Which action should a teacher avoid when concerned about a student?",
        9: "Which assessment practice is most likely to reduce student stress?",
        10: "One student is irritable, one is withdrawn, and one frequently reports headaches before tests. What is the best interpretation?",
        11: "A teacher notices repeated emotional distress even after classroom support. What is the best next step?",
        12: "Which statement best reflects a mentally healthy school culture?"
    },
    "post_assessment": {
        1: "Good mental health in students is best reflected when they:",
        2: "Which description best matches burnout?",
        3: "Which factor is most closely linked to healthy student development in school?",
        4: "A student who usually participates well has stopped answering and avoids eye contact. What is the most appropriate first response?",
        5: "Which classroom practice best supports student emotional safety?",
        6: "A teacher wants to speak to parents about a student's recent change in behaviour. Which opening is best?",
        7: "Which is the best example of a healthy teacher response to student stress?",
        8: "Which assessment practice best supports student well-being?",
        9: "A teacher notices one student becomes restless before tests, one becomes silent during group work, and one often says, 'I cannot do this.' What is the best next move?",
        10: "A school wants to become more mentally healthy. Which change is likely to have the strongest everyday effect?",
        11: "A teacher has supported a student in class, checked in privately, and still sees persistent distress. What should the teacher conclude?",
        12: "Which statement best reflects the spirit of the workshop?"
    }
}

ANSWER_OPTIONS = {
    "pre_assessment": {
        1: {"A": "emotional, social, and psychological well-being", "B": "freedom from illness alone", "C": "difficulty in managing emotions", "D": "need for professional treatment"},
        2: {"A": "classroom seating, school uniforms, and lunch timing", "B": "hobbies, games, and free periods", "C": "academic demands, peer pressure, and family expectations", "D": "assemblies, announcements, and timetables"},
        3: {"A": "missing one homework task in a week", "B": "giving one incorrect answer in class", "C": "asking for help during a lesson", "D": "showing sudden withdrawal over several days"},
        4: {"A": "watch the pattern and speak privately", "B": "ask the student about it during class", "C": "inform the student's classmates immediately", "D": "send the student out for non-compliance"},
        5: {"A": "correcting errors in front of the class", "B": "using respectful language and encouragement", "C": "comparing weaker students with stronger ones", "D": "increasing warnings before each task"},
        6: {"A": "remind the student about the consequences", "B": "tell the student to stop worrying", "C": "guide the student to begin with familiar questions", "D": "advise the student to try harder next time"},
        7: {"A": "sharing labels and seeking confirmation", "B": "sharing conclusions and asking for action", "C": "sharing concerns and suggesting punishment", "D": "sharing observations and inviting partnership"},
        8: {"A": "noting repeated patterns in behaviour", "B": "discussing specific observations with care", "C": "seeking support within school systems", "D": "deciding on a diagnosis from classroom signs"},
        9: {"A": "offering clear success criteria in advance", "B": "keeping performance criteria flexible", "C": "varying instructions across sections", "D": "changing expectations during the task"},
        10: {"A": "they are showing three unrelated behaviours", "B": "they are showing possible signs of stress", "C": "they are responding to weak discipline", "D": "they are seeking attention from teachers"},
        11: {"A": "continue the same support for a longer period", "B": "move the concern through school support channels", "C": "discuss the concern in front of the class", "D": "ask classmates to keep an eye on the student"},
        12: {"A": "emotional issues should be handled outside class", "B": "classroom discipline matters more than student feeling", "C": "academic progress depends on emotional safety", "D": "strong students cope without extra support"}
    },
    "post_assessment": {
        1: {"A": "remain cheerful in most situations", "B": "manage emotions and function reasonably well", "C": "perform strongly in academic tasks", "D": "respond well to adult instructions"},
        2: {"A": "temporary tiredness after a demanding day", "B": "reduced energy during an examination period", "C": "frustration after receiving difficult feedback", "D": "exhaustion, detachment, and reduced motivation"},
        3: {"A": "strong academic competition", "B": "consistent adult support and connection", "C": "strict correction of weak performance", "D": "regular comparison with peer performance"},
        4: {"A": "observe carefully and check in privately", "B": "ask the student to explain the change in class", "C": "record the issue as poor classroom attitude", "D": "ask another student to encourage participation"},
        5: {"A": "responding quickly to mistakes in public", "B": "acknowledging effort in a respectful way", "C": "changing behaviour expectations by situation", "D": "using firm language to build resilience"},
        6: {"A": "Your child has developed a serious concern.", "B": "You need to address this issue at home.", "C": "We have noticed some changes and want to support together.", "D": "This pattern is affecting classroom discipline."},
        7: {"A": "reducing all demands for the rest of the term", "B": "acknowledging the feeling and offering calm guidance", "C": "advising the student to become mentally stronger", "D": "postponing the conversation until the student settles down"},
        8: {"A": "giving clear criteria and calm instructions", "B": "increasing pressure before important tasks", "C": "reviewing errors without discussing strengths", "D": "using surprise questions to improve readiness"},
        9: {"A": "treat all three situations as behaviour concerns", "B": "wait for the patterns to become more frequent", "C": "ask peers to provide encouragement first", "D": "respond to each pattern with appropriate support"},
        10: {"A": "adding one annual awareness event", "B": "strengthening discipline for emotional outbursts", "C": "combining supportive teaching with referral systems", "D": "assigning student well-being only to counsellors"},
        11: {"A": "the classroom response needs more time", "B": "the student may be resisting feedback", "C": "the concern may need referral support", "D": "the issue is likely to settle on its own"},
        12: {"A": "teachers support mental health through daily practice", "B": "teachers need specialist training before taking action", "C": "teachers should focus on learning, not emotional needs", "D": "teachers should refer most concerns immediately"}
    }
}


def normalize_text(text):
    if text is None:
        return ""
    cleaned = str(text).strip().lower()
    cleaned = re.sub(r'[“”"\r\n]+', ' ', cleaned)
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def find_best_matching_column(question_num, question_text, columns):
    normalized_question = normalize_text(question_text)
    best_match = None
    best_ratio = 0.0

    for col in columns:
        col_norm = normalize_text(col)
        if not col_norm:
            continue

        if normalized_question in col_norm or col_norm in normalized_question:
            return col

        if f'q{question_num}' in col_norm or f'question {question_num}' in col_norm:
            return col

        if re.search(rf'(^|\D){question_num}(\D|$)', col_norm):
            return col

        ratio = SequenceMatcher(None, normalized_question, col_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = col

    if best_ratio >= 0.50:
        return best_match

    return None


def normalize_answer_text(answer_str, question_num, assessment_type):
    if not answer_str or str(answer_str).strip().lower() in ['nan', 'not answered', '']:
        return ""

    text = str(answer_str).strip()
    lower = normalize_text(text)
    if not lower:
        return ""

    if lower[0] in ['a', 'b', 'c', 'd']:
        if lower.startswith(('a.', 'b.', 'c.', 'd.')) or lower.startswith(('a ', 'b ', 'c ', 'd ')):
            return lower[0].upper()
        if len(lower) == 1:
            return lower[0].upper()

    options = ANSWER_OPTIONS.get(assessment_type, {}).get(question_num, {})
    for letter, option_text in options.items():
        opt = normalize_text(option_text)
        if lower == opt or lower.startswith(opt) or (opt.startswith(lower) and len(lower) > 1):
            return letter

    for letter, option_text in options.items():
        opt = normalize_text(option_text)
        if opt in lower or lower in opt:
            return letter

    parts = re.split(r'[;,/]|\band\b', text.lower())
    for part in parts:
        part_norm = normalize_text(part)
        if not part_norm:
            continue
        for letter, option_text in options.items():
            opt = normalize_text(option_text)
            if opt in part_norm or part_norm in opt:
                return letter

    return ""

DATA_DIR = os.path.join(".streamlit", "saved_data")
PRE_SAVED_FILE = os.path.join(DATA_DIR, "pre_assessment.csv")
POST_SAVED_FILE = os.path.join(DATA_DIR, "post_assessment.csv")


def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def save_uploaded_assessment(uploaded_file, save_path):
    ensure_data_dir()
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
            df.to_csv(save_path, index=False)
        else:
            uploaded_file.seek(0)
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.read())
        return True
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return False


def load_saved_assessment(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_excel(path)
        except Exception:
            return None


def clear_saved_assessment_data():
    for path in [PRE_SAVED_FILE, POST_SAVED_FILE]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


def initialize_gemini():
    """Initialize Gemini API with environment variable"""
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY in secrets or environment.")
        return False
    genai.configure(api_key=api_key)
    return True

def load_assessment_file(uploaded_file):
    """Load and CSV or Excel file, trying common encodings."""
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
            return df

        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'utf-16']
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding, sep=None, engine='python')
                return df

        st.error("Error loading file: unable to decode the file. Please save it as UTF-8 CSV or Excel.")
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def extract_questions_and_answers(df, assessment_type):
    """
    Extract questions and answers from CSV based on question text matching
    Returns: dict with teacher names as keys and dict of {question_number: answer} as values
    """
    if df is None or df.shape[0] == 0:
        return None
    
    teacher_col_idx = None
    for col in df.columns:
        if col.lower() in ['full name', 'name', 'teacher name', 'teacher', 'name of participant']:
            teacher_col_idx = df.columns.get_loc(col)
            break
    
    if teacher_col_idx is None:
        teacher_col_idx = 0
    
    question_map = QUESTIONS_MAPPING[assessment_type]
    columns = list(df.columns)
    answer_columns = [col for i, col in enumerate(columns) if i != teacher_col_idx]
    fallback_by_position = len(answer_columns) >= 12

    question_column_map = {}
    used_columns = set()
    for q_num, question_text in question_map.items():
        matching_col = find_best_matching_column(q_num, question_text, columns)
        if matching_col:
            question_column_map[q_num] = matching_col
            used_columns.add(matching_col)

    if fallback_by_position:
        remaining_columns = [col for col in answer_columns if col not in used_columns]
        for q_num in range(1, 13):
            if q_num not in question_column_map and q_num - 1 < len(remaining_columns):
                question_column_map[q_num] = remaining_columns[q_num - 1]

    assessments = {}
    for idx in range(len(df)):
        teacher_name = df.iloc[idx, teacher_col_idx]

        if pd.isna(teacher_name) or teacher_name == '':
            continue

        teacher_name = str(teacher_name).strip()

        qa_dict = {}
        for q_num, matching_col in question_column_map.items():
            if not matching_col:
                continue
            col_idx = df.columns.get_loc(matching_col)
            answer = df.iloc[idx, col_idx]
            answer_str = str(answer).strip() if not pd.isna(answer) else ""
            if answer_str:
                qa_dict[q_num] = answer_str

        if qa_dict:
            assessments[teacher_name] = qa_dict

    return assessments if assessments else None

def detect_assessment_type(df):
    """Detect whether a DataFrame is pre-assessment or post-assessment."""
    if df is None or df.shape[1] == 0:
        return None

    pre_headers = [q.lower() for q in QUESTIONS_MAPPING["pre_assessment"].values()]
    post_headers = [q.lower() for q in QUESTIONS_MAPPING["post_assessment"].values()]

    pre_matches = 0
    post_matches = 0

    for col in df.columns:
        col_text = str(col).lower()
        for question_text in pre_headers:
            if question_text in col_text or col_text in question_text:
                pre_matches += 1
                break
        for question_text in post_headers:
            if question_text in col_text or col_text in question_text:
                post_matches += 1
                break

    if pre_matches > post_matches and pre_matches >= 2:
        return "pre_assessment"
    if post_matches > pre_matches and post_matches >= 2:
        return "post_assessment"
    return None

def calculate_score(answers, assessment_type):
    """Calculate score based on answer key"""
    answer_key = ANSWER_KEYS[assessment_type]
    score = 0
    total = 12
    
    for q_num, correct_answer in answer_key.items():
        teacher_answer = answers.get(q_num, "")
        teacher_letter = normalize_answer_text(teacher_answer, q_num, assessment_type)
        if teacher_letter == correct_answer:
            score += 1
    
    return score, total

def build_answer_match_table(answers, assessment_type):
    """Build a DataFrame showing teacher answers compared to the correct answers."""
    answer_key = ANSWER_KEYS[assessment_type]
    questions = QUESTIONS_MAPPING[assessment_type]
    rows = []
    
    for q_num in range(1, 13):
        raw_answer = answers.get(q_num, "Not answered")
        teacher_letter = normalize_answer_text(raw_answer, q_num, assessment_type)
        correct_answer = answer_key.get(q_num, "")
        rows.append({
            "Question #": q_num,
            "Question": questions.get(q_num, f"Question {q_num}"),
            "Teacher Answer Text": raw_answer,
            "Teacher Answer": teacher_letter if teacher_letter else "Not answered",
            "Correct Answer": correct_answer,
            "Match": "Yes" if teacher_letter == correct_answer and teacher_letter else "No"
        })
    
    return pd.DataFrame(rows)

def generate_analysis_prompt(teacher_name, pre_answers, post_answers, pre_score, post_score):
    """Generate a holistic, comprehensive prompt for Gemini analysis"""
    
    pre_answer_key = ANSWER_KEYS["pre_assessment"]
    post_answer_key = ANSWER_KEYS["post_assessment"]
    pre_questions = QUESTIONS_MAPPING["pre_assessment"]
    post_questions = QUESTIONS_MAPPING["post_assessment"]
    
    prompt = f"""Analyze the teacher's professional development holistically based on their pre and post-webinar assessment responses.

Teacher: {teacher_name}

SCORING CONTEXT:
- Pre-Assessment Score: {pre_score}/12
- Post-Assessment Score: {post_score}/12
- Improvement: {post_score - pre_score} points ({((post_score - pre_score) / 12 * 100):.1f}% increase)

Below is the raw data of their responses. Use this to understand their conceptual journey, but DO NOT output a question-by-question breakdown in your response.

PRE-ASSESSMENT:
"""
    
    # Feed the pre-assessment context
    for q_num in range(1, 13):
        question = pre_questions.get(q_num, f"Question {q_num}")
        teacher_raw_answer = pre_answers.get(q_num, "Not answered")
        teacher_letter = normalize_answer_text(teacher_raw_answer, q_num, "pre_assessment")
        correct_answer = pre_answer_key.get(q_num, "")
        is_correct = "CORRECT" if teacher_letter == correct_answer else f"INCORRECT (Should be {correct_answer})"
        
        prompt += f"Q{q_num}: {question}\nTeacher Answered: {teacher_letter} - {is_correct}\n"
    
    prompt += f"\nPOST-ASSESSMENT:\n"
    
    # Feed the post-assessment context
    for q_num in range(1, 13):
        question = post_questions.get(q_num, f"Question {q_num}")
        teacher_raw_answer = post_answers.get(q_num, "Not answered")
        teacher_letter = normalize_answer_text(teacher_raw_answer, q_num, "post_assessment")
        correct_answer = post_answer_key.get(q_num, "")
        is_correct = "CORRECT" if teacher_letter == correct_answer else f"INCORRECT (Should be {correct_answer})"
        
        prompt += f"Q{q_num}: {question}\nTeacher Answered: {teacher_letter} - {is_correct}\n"
    
    prompt += """

Based on the data above, provide a comprehensive, holistic evaluation of this teacher's professional growth. Structure your report as follows:

1. **Holistic Growth Narrative**: A brief, encouraging summary of how their overall understanding of student mental health and emotional safety evolved after the webinar.
2. **Key Paradigm Shifts**: Identify the major conceptual themes where the teacher showed the most significant change in mindset (e.g., shifting from punitive disciplinary reactions to supportive observation, or improving parent communication strategies). 
3. **Classroom Readiness**: An assessment of how well prepared this teacher currently is to implement trauma-informed and mentally healthy practices in their daily teaching.
4. **Strategic Recommendations**: 2-3 high-level, actionable suggestions for their continuous professional development based on any lingering misconceptions.

Format the response in a supportive, professional narrative style suitable for a 1-on-1 performance feedback session. **Strictly avoid itemizing specific question numbers (e.g., do not say "In Question 4..."). Speak to the concepts instead.**"""
    
    return prompt

def generate_local_report(teacher_name, pre_answers, post_answers, pre_score, post_score):
    report_lines = [
        f"Fallback report for {teacher_name}",
        "----------------------------------------",
        f"Pre-Assessment Score: {pre_score}/12",
        f"Post-Assessment Score: {post_score}/12",
        f"Improvement: {post_score - pre_score} points",
        "",
        "Pre-Assessment correctness:",
    ]

    pre_key = ANSWER_KEYS["pre_assessment"]
    for q_num in range(1, 13):
        raw = pre_answers.get(q_num, "Not answered")
        letter = normalize_answer_text(raw, q_num, "pre_assessment") or "Not answered"
        correct = pre_key[q_num]
        report_lines.append(f"Q{q_num}: {letter} (correct: {correct})")

    report_lines.extend(["", "Post-Assessment correctness:"])
    post_key = ANSWER_KEYS["post_assessment"]
    for q_num in range(1, 13):
        raw = post_answers.get(q_num, "Not answered")
        letter = normalize_answer_text(raw, q_num, "post_assessment") or "Not answered"
        correct = post_key[q_num]
        report_lines.append(f"Q{q_num}: {letter} (correct: {correct})")

    report_lines.extend(["", "Note: Gemini analysis was unavailable due to access issues.", "Please verify your project permissions or contact support."])
    return "\n".join(report_lines)

def generate_gemini_report(teacher_name, pre_answers, post_answers, pre_score, post_score):
    """Generate individual analysis using Gemini API"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = generate_analysis_prompt(teacher_name, pre_answers, post_answers, pre_score, post_score)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_message = str(e)
        if '403' in error_message or 'denied access' in error_message.lower() or 'project has been denied access' in error_message.lower():
            return (
                f"⚠️ Gemini access denied: {error_message}\n\n"
                "The app is using a local fallback report instead. "
                "Please check your Gemini project permissions or contact support.\n\n"
                f"{generate_local_report(teacher_name, pre_answers, post_answers, pre_score, post_score)}"
            )
        return f"Error generating report: {error_message}\n\n{generate_local_report(teacher_name, pre_answers, post_answers, pre_score, post_score)}"

def create_score_comparison_chart(scores_data):
    """Create a visual comparison of pre vs post scores"""
    fig = go.Figure()
    
    teachers = list(scores_data.keys())
    pre_scores = [scores_data[t]["pre_score"] for t in teachers]
    post_scores = [scores_data[t]["post_score"] for t in teachers]
    
    fig.add_trace(go.Bar(
        x=teachers,
        y=pre_scores,
        name='Pre-Assessment',
        marker_color='indianred',
        text=pre_scores,
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        x=teachers,
        y=post_scores,
        name='Post-Assessment',
        marker_color='lightseagreen',
        text=post_scores,
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Pre vs Post-Assessment Scores",
        xaxis_title="Teacher Name",
        yaxis_title="Score (out of 12)",
        barmode='group',
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def create_improvement_chart(scores_data):
    """Create a chart showing improvement"""
    fig = go.Figure()
    
    teachers = list(scores_data.keys())
    improvements = [scores_data[t]["post_score"] - scores_data[t]["pre_score"] for t in teachers]
    colors = ['green' if x > 0 else 'red' for x in improvements]
    
    fig.add_trace(go.Bar(
        x=teachers,
        y=improvements,
        marker_color=colors,
        text=improvements,
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Score Improvement (Post - Pre)",
        xaxis_title="Teacher Name",
        yaxis_title="Improvement (points)",
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def create_performance_gauge(score, total=12):
    """Create a gauge chart for a single score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Score"},
        delta={'reference': total},
        gauge={
            'axis': {'range': [None, total]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, total/3], 'color': "lightgray"},
                {'range': [total/3, 2*total/3], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': total}
        }
    ))
    fig.update_layout(height=400)
    return fig

def create_global_question_analysis_chart(pre_assessments, post_assessments, teachers_both):
    """Create a chart showing the success rate of each question across all teachers."""
    pre_key = ANSWER_KEYS["pre_assessment"]
    post_key = ANSWER_KEYS["post_assessment"]

    pre_correct = {q: 0 for q in range(1, 13)}
    post_correct = {q: 0 for q in range(1, 13)}
    total = len(teachers_both)

    if total == 0:
        return go.Figure()

    for teacher in teachers_both:
        for q_num in range(1, 13):
            ans = normalize_answer_text(pre_assessments[teacher].get(q_num, ""), q_num, "pre_assessment")
            if ans == pre_key.get(q_num):
                pre_correct[q_num] += 1

        for q_num in range(1, 13):
            ans = normalize_answer_text(post_assessments[teacher].get(q_num, ""), q_num, "post_assessment")
            if ans == post_key.get(q_num):
                post_correct[q_num] += 1

    pre_pct = [(pre_correct[q] / total) * 100 for q in range(1, 13)]
    post_pct = [(post_correct[q] / total) * 100 for q in range(1, 13)]
    questions = [f"Q{q}" for q in range(1, 13)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=questions,
        y=pre_pct,
        name='Pre-Webinar (% Correct)',
        marker_color='indianred',
        text=[f"{val:.1f}%" for val in pre_pct],
        textposition='auto',
    ))

    fig.add_trace(go.Bar(
        x=questions,
        y=post_pct,
        name='Post-Webinar (% Correct)',
        marker_color='lightseagreen',
        text=[f"{val:.1f}%" for val in post_pct],
        textposition='auto',
    ))

    fig.update_layout(
        title="Macro Insight: Success Rate per Question",
        xaxis_title="Question Number",
        yaxis_title="Success Rate (%)",
        barmode='group',
        height=500,
        hovermode='x unified',
        template='plotly_white',
        yaxis=dict(range=[0, 115]) 
    )

    return fig

def generate_cohort_analysis_prompt(total_teachers, avg_pre, avg_post, avg_imp, pre_pct, post_pct):
    """Generate a prompt for Gemini to analyze the entire cohort's performance."""
    prompt = f"""You are an expert educational data analyst. Analyze the following pre- and post-webinar assessment data for a cohort of teachers and write a comprehensive Executive Report for a Product Manager.

COHORT OVERVIEW:
- Total Participants: {total_teachers}
- Average Pre-Assessment Score: {avg_pre:.2f}/12
- Average Post-Assessment Score: {avg_post:.2f}/12
- Average Improvement: +{avg_imp:.2f} points

QUESTION-LEVEL SUCCESS RATES (Pre-Webinar vs Post-Webinar):
"""
    for q_num in range(1, 13):
        pre_val = pre_pct[q_num-1]
        post_val = post_pct[q_num-1]
        delta = post_val - pre_val
        prompt += f"Q{q_num}: Pre: {pre_val:.1f}% | Post: {post_val:.1f}% | Change: {delta:+.1f}%\n"

    prompt += """
Based on this data, provide a comprehensive report including:
1. **Executive Summary**: A brief overview of the training's overall impact.
2. **Macro Score Analysis**: Interpretation of the cohort's average score changes.
3. **Areas of Massive Growth (Webinar Successes)**: Identify which questions/topics saw the biggest positive jumps.
4. **Areas of High Retention**: Identify concepts teachers already knew well (high pre and post scores).
5. **Areas of Concern (Regression/Confusion)**: Identify questions where scores dropped or remained very low, indicating a need for module revision.
6. **Recommendations for Next Iteration**: Specific, actionable advice for the Product Manager to improve the next webinar based on the data.

Format the response in a clear, professional, and structured manner using markdown. Keep it punchy and actionable."""
    
    return prompt

def generate_gemini_cohort_report(total_teachers, avg_pre, avg_post, avg_imp, pre_pct, post_pct):
    """Generate the cohort analysis using Gemini API"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = generate_cohort_analysis_prompt(total_teachers, avg_pre, avg_post, avg_imp, pre_pct, post_pct)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error generating cohort report: {str(e)}"

# Main application layout
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Pre-Assessment CSV")
    pre_file = st.file_uploader("Upload Pre-Assessment CSV", key="pre_assessment")

with col2:
    st.subheader("📋 Post-Assessment CSV")
    post_file = st.file_uploader("Upload Post-Assessment CSV", key="post_assessment")

# Process uploaded files
saved_data_message = None
pre_df = None
post_df = None

if pre_file and post_file:
    # Load files
    pre_df = load_assessment_file(pre_file)
    post_df = load_assessment_file(post_file)
    
    if pre_df is not None and post_df is not None:
        save_uploaded_assessment(pre_file, PRE_SAVED_FILE)
        save_uploaded_assessment(post_file, POST_SAVED_FILE)
        saved_data_message = "✅ Uploaded files saved for future shareable report links."
elif os.path.exists(PRE_SAVED_FILE) and os.path.exists(POST_SAVED_FILE):
    pre_df = load_saved_assessment(PRE_SAVED_FILE)
    post_df = load_saved_assessment(POST_SAVED_FILE)
    if pre_df is not None and post_df is not None:
        saved_data_message = "✅ Loaded saved assessment data. Shareable teacher links are now active."

if saved_data_message:
    st.success(saved_data_message)
    if st.button("Clear saved assessment data"):
        clear_saved_assessment_data()
        st.experimental_rerun()

if pre_df is not None and post_df is not None:
    # Detect assessment types from the uploaded file headers
    detected_pre_type = detect_assessment_type(pre_df)
    detected_post_type = detect_assessment_type(post_df)

    if detected_pre_type == "post_assessment" and detected_post_type == "pre_assessment":
        st.warning("It looks like the files were uploaded in reverse order. The app will swap them automatically.")
        pre_df, post_df = post_df, pre_df
        detected_pre_type, detected_post_type = detected_post_type, detected_pre_type

    if detected_pre_type is None or detected_post_type is None:
        st.info("Could not automatically detect one or both assessment types. Using the uploaded order.")
        detected_pre_type = detected_pre_type or "pre_assessment"
        detected_post_type = detected_post_type or "post_assessment"

    # Extract assessments
    pre_assessments = extract_questions_and_answers(pre_df, detected_pre_type)
    post_assessments = extract_questions_and_answers(post_df, detected_post_type)

    if pre_assessments and post_assessments:
        # Find teachers present in both assessments
        pre_teachers = set(pre_assessments.keys())
        post_teachers = set(post_assessments.keys())
        teachers_both = pre_teachers & post_teachers
        teachers_only_pre = pre_teachers - post_teachers
        teachers_only_post = post_teachers - pre_teachers
            
        teacher_param = st.query_params.get("teacher")
        teacher_name = None
        if teacher_param:
            if isinstance(teacher_param, list) and len(teacher_param) > 0:
                teacher_name = teacher_param[0]
            elif isinstance(teacher_param, str):
                teacher_name = teacher_param
            teacher_name = unquote_plus(teacher_name).strip() if teacher_name else None

        if teacher_name and teacher_name in teachers_both:
            # Show single teacher report
            pre_score, _ = calculate_score(pre_assessments[teacher_name], "pre_assessment")
            post_score, _ = calculate_score(post_assessments[teacher_name], "post_assessment")

            st.markdown("---")
            st.subheader(f"📊 Report for {teacher_name}")

            # Show individual scores
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pre-Assessment Score", f"{pre_score}/12")
            with col2:
                st.metric("Post-Assessment Score", f"{post_score}/12")
            with col3:
                st.metric("Improvement", f"{post_score - pre_score} points")

            # Show gauge charts
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_performance_gauge(pre_score), use_container_width=True, key=f"gauge_pre_{teacher_name}")
            with col2:
                st.plotly_chart(create_performance_gauge(post_score), use_container_width=True, key=f"gauge_post_{teacher_name}")

            # Generate AI analysis
            with st.spinner(f"Generating detailed report for {teacher_name}..."):
                report = generate_gemini_report(
                    teacher_name, 
                    pre_assessments[teacher_name], 
                    post_assessments[teacher_name],
                    pre_score,
                    post_score
                )
                st.markdown(report)

            # Option to download report
            download_content = f"Teacher Assessment Report\n{'='*60}\n\nTeacher: {teacher_name}\n"
            download_content += f"Pre-Assessment Score: {pre_score}/12\nPost-Assessment Score: {post_score}/12\n"
            download_content += f"Improvement: {post_score - pre_score} points\n\n"
            download_content += f"Detailed Analysis:\n{report}"

            st.download_button(
                label="📥 Download Report",
                data=download_content,
                file_name=f"{teacher_name}_assessment_report.txt",
                mime="text/plain"
            )

        else:
            # Show full app
            st.markdown("---")
            st.subheader("📊 Assessment Summary")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pre-Assessment Teachers", len(pre_teachers))
            with col2:
                st.metric("Post-Assessment Teachers", len(post_teachers))
            with col3:
                st.metric("Both Assessments", len(teachers_both))
            with col4:
                st.metric("Incomplete", len(teachers_only_pre) + len(teachers_only_post))

            # Calculate scores for all teachers
            scores_data = {}
            for teacher in teachers_both:
                pre_score, _ = calculate_score(pre_assessments[teacher], "pre_assessment")
                post_score, _ = calculate_score(post_assessments[teacher], "post_assessment")
                scores_data[teacher] = {
                    "pre_score": pre_score,
                    "post_score": post_score,
                    "improvement": post_score - pre_score
                }

            st.markdown("---")
            st.subheader("📈 Score Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_score_comparison_chart(scores_data), use_container_width=True, key="score_comparison_chart")
            
            with col2:
                st.plotly_chart(create_improvement_chart(scores_data), use_container_width=True, key="improvement_chart")
            
            # Show Global Question Analysis Chart
            # st.markdown("#### 🔍 Global Question Analysis")
            # global_chart = create_global_question_analysis_chart(pre_assessments, post_assessments, teachers_both)
            # st.plotly_chart(global_chart, use_container_width=True, key="global_question_analysis_chart")
            
            # Summary Statistics
            st.markdown("---")
            st.subheader("📊 Overall Statistics")
            
            pre_scores_list = [scores_data[t]["pre_score"] for t in teachers_both]
            post_scores_list = [scores_data[t]["post_score"] for t in teachers_both]
            improvements_list = [scores_data[t]["improvement"] for t in teachers_both]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Pre-Score", f"{np.mean(pre_scores_list):.2f}/12")
            with col2:
                st.metric("Avg Post-Score", f"{np.mean(post_scores_list):.2f}/12")
            with col3:
                st.metric("Avg Improvement", f"{np.mean(improvements_list):.2f}")
            with col4:
                improved_count = sum(1 for x in improvements_list if x > 0)
                st.metric("Teachers Improved", f"{improved_count}/{len(teachers_both)}")
            
            # Initialize Gemini
            if not initialize_gemini():
                st.stop()
            
            st.markdown("---")
            
            # Generate reports for teachers with both assessments
            if teachers_both:
                st.subheader("✅ Teacher Reports")
                st.write(f"Generating detailed analysis for {len(teachers_both)} teachers...")
                
                # Create tabs for each teacher
                teacher_list = sorted(list(teachers_both))
                tabs = st.tabs(teacher_list)
                
                for tab, teacher_name in zip(tabs, teacher_list):
                    with tab:
                        pre_score, _ = calculate_score(pre_assessments[teacher_name], "pre_assessment")
                        post_score, _ = calculate_score(post_assessments[teacher_name], "post_assessment")
                        
                        # Show individual scores
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Pre-Assessment Score", f"{pre_score}/12")
                        with col2:
                            st.metric("Post-Assessment Score", f"{post_score}/12")
                        with col3:
                            st.metric("Improvement", f"{post_score - pre_score} points")
                        
                        # Show gauge charts
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(create_performance_gauge(pre_score), use_container_width=True, key=f"gauge_pre_{teacher_name}")
                        with col2:
                            st.plotly_chart(create_performance_gauge(post_score), use_container_width=True, key=f"gauge_post_{teacher_name}")
                        
                        # Show answer-key match tables
                        # st.subheader("✅ Answer Key Matches")
                        # st.markdown("**Pre-Assessment answer match:**")
                        # pre_match_df = build_answer_match_table(pre_assessments[teacher_name], "pre_assessment")
                        # st.dataframe(pre_match_df, use_container_width=True)
                        # st.markdown("**Post-Assessment answer match:**")
                        # post_match_df = build_answer_match_table(post_assessments[teacher_name], "post_assessment")
                        # st.dataframe(post_match_df, use_container_width=True)
                        
                        # Generate AI analysis
                        with st.spinner(f"Generating detailed report for {teacher_name}..."):
                            report = generate_gemini_report(
                                teacher_name, 
                                pre_assessments[teacher_name], 
                                post_assessments[teacher_name],
                                pre_score,
                                post_score
                            )
                            st.markdown(report)
                        
                        # Option to download report
                        download_content = f"Teacher Assessment Report\n{'='*60}\n\nTeacher: {teacher_name}\n"
                        download_content += f"Pre-Assessment Score: {pre_score}/12\nPost-Assessment Score: {post_score}/12\n"
                        download_content += f"Improvement: {post_score - pre_score} points\n\n"
                        download_content += f"Detailed Analysis:\n{report}"
                        
                        st.download_button(
                            label="📥 Download Report",
                            data=download_content,
                            file_name=f"{teacher_name}_assessment_report.txt",
                            mime="text/plain"
                        )
                
                # Shareable Links
                st.markdown("---")
                st.subheader("🔗 Shareable Links")
                base_url = st.text_input("Enter your deployed app URL (e.g., https://yourapp.streamlit.app)", value="https://your-app-name.streamlit.app")
                if base_url:
                    st.write("Copy these links to share individual reports:")
                    for teacher in teacher_list:
                        safe_name = quote_plus(teacher)
                        link = f"{base_url.rstrip('/')}/?teacher={safe_name}"
                        st.code(link, language="text")
                
                # Show incomplete assessments
                st.markdown("---")
                st.subheader("⚠️ Incomplete Assessments")
                
                if teachers_only_pre:
                    st.warning(f"**Teachers who completed Pre-Assessment but not Post-Assessment ({len(teachers_only_pre)}):**")
                    for teacher in sorted(teachers_only_pre):
                        st.write(f"• {teacher}")
                
                if teachers_only_post:
                    st.warning(f"**Teachers who completed Post-Assessment but not Pre-Assessment ({len(teachers_only_post)}):**")
                    for teacher in sorted(teachers_only_post):
                        st.write(f"• {teacher}")
    else:
        st.error("Unable to extract data from CSV files. Please ensure the data includes teacher names and 12 question responses.")
elif pre_file or post_file or os.path.exists(PRE_SAVED_FILE) or os.path.exists(POST_SAVED_FILE):
    st.error("Failed to load one or both files.")

else:
    st.info("👉 Please upload both Pre-Assessment and Post-Assessment CSV files to begin analysis.")
    
    # Show expected file format
    with st.expander("📖 Expected CSV Format & Features"):
        st.markdown("""
        **CSV Structure:**
        - **First column or "Full Name"**: Teacher Names
        - **Next 12 columns**: Answers to 12 multiple-choice questions (A, B, C, or D)
        - Other columns like email, date, designation are automatically skipped
        
        **Features:**
        - ✅ Automatic grading based on answer keys
        - ✅ Score comparison visualization (pre vs post)
        - ✅ Improvement tracking for each teacher
        - ✅ Statistical analysis (average improvement, % improved)
        - ✅ Global macro-level analysis (question-by-question success rate)
        - ✅ AI-powered Executive Cohort Report for PMs
        - ✅ Individual teacher reports with scores
        - ✅ Detailed breakdown of correct/incorrect answers
        - ✅ Download reports as text files
        
        **Answer Keys Configured:**
        - Pre-Assessment: 12 questions with correct answers A-D
        - Post-Assessment: 12 questions with correct answers A-D
        """)