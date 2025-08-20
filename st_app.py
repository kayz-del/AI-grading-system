# st_app.py
import streamlit as st
import os
from PIL import Image
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
import google.generativeai as genai
import json
import easyocr # Use the reliable OCR library

# --- Configuration ---
# NO TESSERACT CONFIGURATION NEEDED
DB_URI = 'sqlite:///instance/exams.db'
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("instance", exist_ok=True)

# Configure the Gemini API using the key from secrets.toml
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("GEMINI_API_KEY not found. Please ensure it is correctly set in your .streamlit/secrets.toml file or Streamlit Cloud settings.")
    st.stop()

# --- Database Setup (unchanged) ---
Base = declarative_base()
engine = create_engine(DB_URI, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)

# --- Database Models (unchanged) ---
class Exam(Base):
    __tablename__ = 'exam'
    id = Column(Integer, primary_key=True)
    title = Column(String(100), nullable=False)
    questions = relationship('Question', backref='exam', cascade="all, delete-orphan")

class Question(Base):
    __tablename__ = 'question'
    id = Column(Integer, primary_key=True)
    exam_id = Column(Integer, ForeignKey('exam.id'), nullable=False)
    question_text = Column(String(500), nullable=False)
    correct_answer = Column(String(1000), nullable=True)
    points = Column(Integer, nullable=False, default=10)

class Submission(Base):
    __tablename__ = 'submission'
    id = Column(Integer, primary_key=True)
    exam_id = Column(Integer, ForeignKey('exam.id'), nullable=False)
    student_name = Column(String(100), nullable=False)
    matric_number = Column(String(50), nullable=False)
    department = Column(String(100), nullable=False)
    final_score = Column(Float, nullable=True)
    total_points = Column(Integer, nullable=True)
    answers = relationship('Answer', backref='submission', cascade="all, delete-orphan")

class Answer(Base):
    __tablename__ = 'answer'
    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey('submission.id'), nullable=False)
    question_id = Column(Integer, ForeignKey('question.id'), nullable=False)
    extracted_text = Column(String, nullable=True)
    awarded_score = Column(Float, nullable=True)
    feedback = Column(String, nullable=True)
    question = relationship("Question")

Base.metadata.create_all(engine)

# --- AI Logic (Gemini and EasyOCR) ---

@st.cache_resource
def load_ocr_model():
    """Load and cache the EasyOCR model."""
    return easyocr.Reader(['en'])

def extract_text_with_easyocr(image_path):
    """Uses EasyOCR to extract text from an image file."""
    try:
        reader = load_ocr_model()
        result = reader.readtext(image_path, detail=0, paragraph=True)
        return " ".join(result)
    except Exception as e:
        st.error(f"Error during OCR with EasyOCR: {e}")
        return ""

@st.cache_data(ttl=600)
def grade_answer_with_gemini(question_text, student_answer, points_for_question):
    """Uses Gemini to grade an answer based on factual correctness and understanding."""
    prompt = f'''
    You are a strict but fair university professor grading an exam. Your task is to evaluate a student's answer based on the provided question.

    **Exam Question:**
    "{question_text}"

    **Student's Answer:**
    "{student_answer}"

    **Instructions:**
    1.  Critically assess if the student's answer is factually and conceptually correct, partially correct, or completely incorrect.
    2.  Provide a brief, one-sentence explanation for your assessment as feedback.
    3.  Provide a numerical score from 0 to {points_for_question}. A fully correct answer gets {points_for_question}. A partially correct answer gets a proportional score. A completely incorrect, irrelevant, or factually wrong answer **must receive 0 points**.

    **Respond ONLY with a JSON object in the following format:**
    {{
        "reasoning": "Your one-sentence explanation here.",
        "score": your_numerical_score_here
    }}
    '''
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        result = json.loads(cleaned_response)
        score = max(0, min(float(result['score']), points_for_question))
        return { "feedback": result['reasoning'], "awarded_score": score }
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return { "feedback": "Could not grade due to an API error.", "awarded_score": 0.0 }

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("✨ AI Exam Grading System (Powered by Gemini)")

page = st.sidebar.selectbox("Choose a page", ["Take Exam", "Create Exam", "View Submissions"])

if page == "Create Exam":
    # This section is unchanged
    st.header("🧑‍🏫 Create a New Exam")
    # ... (code is identical to previous versions)
    with st.form("exam_form"):
        exam_title = st.text_input("Exam Title")
        st.subheader("Add Questions")
        if 'questions' not in st.session_state:
            st.session_state.questions = [{"text": "", "answer": "", "points": 10}]
        for i, q in enumerate(st.session_state.questions):
            st.markdown(f"**Question {i+1}**")
            q['text'] = st.text_area(f"Question Text {i+1}", q['text'])
            q['answer'] = st.text_area(f"Ideal Answer (for human reference only) {i+1}", q['answer'])
            q['points'] = st.number_input(f"Points for Question {i+1}", min_value=1, value=q['points'])
            st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("Add Another Question"):
                st.session_state.questions.append({"text": "", "answer": "", "points": 10})
                st.rerun()
        with col2:
            submitted = st.form_submit_button("Save Exam", type="primary")
        if submitted:
            if not exam_title or not all(q['text'] for q in st.session_state.questions):
                st.error("Please fill out the exam title and all question fields.")
            else:
                session = Session()
                new_exam = Exam(title=exam_title)
                for q_data in st.session_state.questions:
                    new_question = Question(
                        question_text=q_data['text'],
                        correct_answer=q_data['answer'],
                        points=q_data['points']
                    )
                    new_exam.questions.append(new_question)
                session.add(new_exam)
                session.commit()
                session.close()
                st.success(f"Exam '{exam_title}' created!")
                del st.session_state.questions

elif page == "Take Exam":
    st.header("🧑‍🎓 Take an Exam")
    session = Session()
    exams = session.query(Exam).all()
    exam_choices = {exam.title: exam for exam in exams}
    if not exam_choices:
        st.warning("No exams created yet.")
    else:
        selected_exam_title = st.selectbox("Choose an exam", list(exam_choices.keys()))
        selected_exam = exam_choices[selected_exam_title]
        st.subheader("Student Information")
        student_name = st.text_input("Your Full Name")
        matric_number = st.text_input("Matriculation Number")
        department = st.text_input("Department")
        uploaded_files = {}
        for i, question in enumerate(selected_exam.questions):
            st.markdown("---")
            st.subheader(f"Question {i+1} ({question.points} points)")
            st.info(question.question_text)
            uploaded_files[question.id] = st.file_uploader(
                f"Upload answer for Question {i+1}", type=["png", "jpg"], key=f"q_{question.id}"
            )
        if st.button("Submit All Answers for Grading", type="primary"):
            if all(uploaded_files.values()) and student_name and matric_number and department:
                total_score, total_possible_points = 0, sum(q.points for q in selected_exam.questions)
                new_submission = Submission(
                    exam_id=selected_exam.id, student_name=student_name, matric_number=matric_number,
                    department=department, total_points=total_possible_points
                )
                session.add(new_submission)
                session.commit()
                st.subheader("Grading Results")
                for i, question in enumerate(selected_exam.questions):
                    st.markdown(f"---")
                    st.markdown(f"**Processing Question {i+1}...**")
                    uploaded_file = uploaded_files[question.id]
                    with st.spinner(f'AI is reading answer for Question {i+1}...'):
                        image_path = os.path.join(UPLOAD_FOLDER, f"sub{new_submission.id}_q{question.id}_{uploaded_file.name}")
                        with open(image_path, "wb") as f: f.write(uploaded_file.getbuffer())
                        # USE THE NEW RELIABLE OCR FUNCTION
                        extracted_text = extract_text_with_easyocr(image_path)
                    st.markdown("##### Text Extracted by AI:")
                    st.text_area("Extracted Text:", extracted_text if extracted_text else "AI could not read any text.", height=100, disabled=True, key=f"res_text_{question.id}")
                    feedback, awarded_score = "Not graded.", 0.0
                    if extracted_text:
                        with st.spinner(f"Gemini AI is grading answer for Question {i+1}..."):
                           grading_result = grade_answer_with_gemini(question.question_text, extracted_text, question.points)
                           awarded_score = grading_result["awarded_score"]
                           feedback = grading_result["feedback"]
                    st.info(f"**AI Feedback:** {feedback}")
                    st.success(f"**Score for this question:** {awarded_score:.2f} / {question.points}")
                    new_answer = Answer(
                        submission_id=new_submission.id, question_id=question.id,
                        extracted_text=extracted_text, awarded_score=awarded_score, feedback=feedback
                    )
                    session.add(new_answer)
                    total_score += awarded_score
                new_submission.final_score = total_score
                session.commit()
                st.header(f"Final Score: **{total_score:.2f} / {total_possible_points}**")
                st.balloons()
            else:
                st.error("Please fill out all student information and upload an answer for every question.")
    session.close()

elif page == "View Submissions":
    # This section is unchanged
    st.header("📊 View All Submissions")
    # ... (code is identical to previous versions)
    session = Session()
    submissions = session.query(Submission).order_by(Submission.id.desc()).all()
    if not submissions:
        st.warning("No submissions recorded yet.")
    else:
        for sub in submissions:
            score_display = f"{sub.final_score:.2f}" if sub.final_score is not None else "Incomplete"
            expander_title = (
                f"**{sub.student_name}** - Matric: {sub.matric_number}, Dept: {sub.department} "
                f"| Score: **{score_display} / {sub.total_points}**"
            )
            with st.expander(expander_title):
                st.write(f"**Exam ID:** {sub.exam_id}")
                for ans in sub.answers:
                    st.markdown(f"**Question {ans.question.id} ({ans.question.points} points):** *'{ans.question.question_text}'*")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("##### Student's Answer (Extracted Text)")
                        st.text_area("Extracted Text:", ans.extracted_text, height=200, disabled=True, key=f"text_{ans.id}")
                    with col2:
                        st.markdown("##### Ideal Answer (from Engine)")
                        st.text_area("Ideal Answer:", ans.question.correct_answer, height=200, disabled=True, key=f"ideal_{ans.id}")
                    st.info(f"**AI Feedback:** {ans.feedback}")
                    st.success(f"**Score Awarded for this Question:** {ans.awarded_score:.2f}")
                    st.markdown("---")
    session.close()