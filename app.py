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

# Initialize Gemini API
def initialize_gemini():
    """Initialize Gemini API with environment variable"""
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ Gemini API key not found. Please set GEMINI_API_KEY in secrets or environment.")
        return False
    genai.configure(api_key=api_key)
    return True

# Load and validate assessment files
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
                # try a more permissive separator if necessary
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
    
    # Find teacher name column
    teacher_col_idx = None
    for col in df.columns:
        if col.lower() in ['full name', 'name', 'teacher name', 'teacher', 'name of participant']:
            teacher_col_idx = df.columns.get_loc(col)
            break
    
    if teacher_col_idx is None:
        teacher_col_idx = 0
    
    # Get question mappings for this assessment type
    question_map = QUESTIONS_MAPPING[assessment_type]
    columns = list(df.columns)
    answer_columns = [col for i, col in enumerate(columns) if i != teacher_col_idx]
    fallback_by_position = len(answer_columns) >= 12

    # Build question-to-column map once for the sheet
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
    """Generate a comprehensive prompt for Gemini analysis"""
    
    pre_answer_key = ANSWER_KEYS["pre_assessment"]
    post_answer_key = ANSWER_KEYS["post_assessment"]
    pre_questions = QUESTIONS_MAPPING["pre_assessment"]
    post_questions = QUESTIONS_MAPPING["post_assessment"]
    
    prompt = f"""Analyze the teacher's professional development based on pre and post-asstresessment responses.

Teacher: {teacher_name}

SCORING:
- Pre-Assessment Score: {pre_score}/12
- Post-Assessment Score: {post_score}/12
- Improvement: {post_score - pre_score} points ({((post_score - pre_score) / 12 * 100):.1f}% increase)

PRE-ASSESSMENT (Before Webinar):
Score: {pre_score}/12

"""
    
    for q_num in range(1, 13):
        question = pre_questions.get(q_num, f"Question {q_num}")
        teacher_raw_answer = pre_answers.get(q_num, "Not answered")
        teacher_letter = normalize_answer_text(teacher_raw_answer, q_num, "pre_assessment")
        correct_answer = pre_answer_key.get(q_num, "")
        is_correct = "✓ CORRECT" if teacher_letter == correct_answer else "✗ INCORRECT"
        
        prompt += f"\nQ{q_num}. {question}\n"
        prompt += f"  Your answer: {teacher_raw_answer} ({teacher_letter if teacher_letter else 'Not answered'}) {is_correct}\n"
        prompt += f"  Correct answer: {correct_answer}\n"
    
    prompt += f"\n\nPOST-ASSESSMENT (After Webinar):\nScore: {post_score}/12\n"
    
    for q_num in range(1, 13):
        question = post_questions.get(q_num, f"Question {q_num}")
        teacher_raw_answer = post_answers.get(q_num, "Not answered")
        teacher_letter = normalize_answer_text(teacher_raw_answer, q_num, "post_assessment")
        correct_answer = post_answer_key.get(q_num, "")
        is_correct = "✓ CORRECT" if teacher_letter == correct_answer else "✗ INCORRECT"
        
        prompt += f"\nQ{q_num}. {question}\n"
        prompt += f"  Your answer: {teacher_raw_answer} ({teacher_letter if teacher_letter else 'Not answered'}) {is_correct}\n"
        prompt += f"  Correct answer: {correct_answer}\n"
    
    prompt += f"""

Based on the assessment scores and responses, provide a comprehensive report including:
1. **Score Analysis**: Interpretation of the scores and improvement
2. **Key Areas of Improvement**: Specific topics where the teacher showed growth
3. **Correct Answers in Pre-Assessment**: Knowledge already present
4. **Newly Correct Answers in Post-Assessment**: Knowledge gained from training
5. **Remaining Knowledge Gaps**: Questions still missed in post-assessment
6. **Learning Effectiveness**: How well the teacher retained training concepts
7. **Recommendations**: Specific suggestions for continued professional growth based on remaining gaps
8. **Overall Assessment**: Summary of learning outcomes and readiness for implementation

Format the response in a clear, professional manner suitable for teacher feedback."""
    
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
    """Generate analysis using Gemini API"""
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
if pre_file and post_file:
    # Load files
    pre_df = load_assessment_file(pre_file)
    post_df = load_assessment_file(post_file)
    
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
            
            # Show score comparison visualizations
            st.markdown("---")
            st.subheader("📈 Score Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_score_comparison_chart(scores_data), use_container_width=True, key="score_comparison_chart")
            
            with col2:
                st.plotly_chart(create_improvement_chart(scores_data), use_container_width=True, key="improvement_chart")
            
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
                        st.subheader("✅ Answer Key Matches")
                        st.markdown("**Pre-Assessment answer match:**")
                        pre_match_df = build_answer_match_table(pre_assessments[teacher_name], "pre_assessment")
                        st.dataframe(pre_match_df, use_container_width=True)
                        st.markdown("**Post-Assessment answer match:**")
                        post_match_df = build_answer_match_table(post_assessments[teacher_name], "post_assessment")
                        st.dataframe(post_match_df, use_container_width=True)
                        
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
    else:
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
        - ✅ Individual teacher reports with scores
        - ✅ AI-powered analysis using Gemini 2.5 Flash
        - ✅ Detailed breakdown of correct/incorrect answers
        - ✅ Download reports as text files
        
        **Answer Keys Configured:**
        - Pre-Assessment: 12 questions with correct answers A-D
        - Post-Assessment: 12 questions with correct answers A-D
        """)
