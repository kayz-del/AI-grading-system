# st_app.py
import streamlit as st
import os
from PIL import Image
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
import google.generativeai as genai
import json

# --- Configuration ---
DB_URI = 'sqlite:///instance/exams.db'
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("instance", exist_ok=True)

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("GEMINI_API_KEY not found. Please ensure it is correctly set in your .streamlit/secrets.toml file or Streamlit Cloud settings.")
    st.stop()

# --- Database Setup ---
Base = declarative_base()
engine = create_engine(DB_URI, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)

# --- Database Models ---
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

# --- AI Logic (Gemini Vision Only) ---

def extract_text_with_gemini_vision(image_path):
    """Uses Gemini's vision capability to extract text from handwritten answers."""
    prompt = '''
    Extract the text from this handwritten answer image as accurately as possible. 
    Preserve the original wording and structure. If the handwriting is unclear, 
    do your best to transcribe what you see without interpretation or correction.
    
    Return ONLY the transcribed text without any additional commentary or formatting.
    '''
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        image = Image.open(image_path)
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        st.error(f"Gemini Vision Text Extraction Error: {e}")
        return ""

def analyze_answer_with_gemini_vision(question_text, image_path, points_for_question):
    """Combined function that extracts text AND grades in one call for better accuracy."""
    prompt = f'''
    You are a strict but fair university professor grading an exam. Your task is to evaluate a student's handwritten answer.

    **Exam Question:**
    "{question_text}"

    **Instructions:**
    1. First, carefully read and transcribe the handwritten answer from the image.
    2. Then, critically assess if the student's answer is factually and conceptually correct, partially correct, or completely incorrect.
    3. Focus on conceptual understanding rather than perfect spelling in handwriting.
    4. Provide a numerical score from 0 to {points_for_question}. A fully correct answer gets {points_for_question}. A partially correct answer gets a proportional score. A completely incorrect, irrelevant, or factually wrong answer MUST receive 0 points.
    5. Be strict - if the answer shows fundamental misunderstandings, award low scores.

    **Respond with a JSON object in this exact format:**
    {{
        "extracted_text": "The transcribed text from the handwritten answer",
        "reasoning": "Your one-sentence evaluation explanation",
        "score": [ACTUAL_SCORE_BASED_ON_QUALITY]
    }}

    **Important:** The "score" must be an actual number between 0 and {points_for_question} based on the answer quality, not the maximum points.
    '''
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        image = Image.open(image_path)
        response = model.generate_content([prompt, image])
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        result = json.loads(cleaned_response)
        
        # Validate and clamp the score
        raw_score = float(result.get('score', 0))
        score = max(0, min(raw_score, points_for_question))
        
        return {
            "extracted_text": result.get('extracted_text', ''),
            "feedback": result.get('reasoning', ''),
            "awarded_score": score
        }
    except Exception as e:
        st.error(f"Gemini Vision Analysis Error: {e}")
        return {
            "extracted_text": "",
            "feedback": "Could not analyze due to an API error.", 
            "awarded_score": 0.0
        }

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("✨ AI Exam Grading System (Powered by Gemini Vision)")

page = st.sidebar.selectbox("Choose a page", ["Take Exam", "Create Exam", "View Submissions", "Admin"])

if page == "Create Exam":
    st.header("🧑‍🏫 Create a New Exam")
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
                f"Upload answer for Question {i+1}", type=["png", "jpg", "jpeg"], key=f"q_{question.id}"
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
                    
                    # Save the uploaded image
                    image_path = os.path.join(UPLOAD_FOLDER, f"sub{new_submission.id}_q{question.id}_{uploaded_file.name}")
                    with open(image_path, "wb") as f: 
                        f.write(uploaded_file.getbuffer())
                    
                    # Display the uploaded image
                    st.image(uploaded_file, caption=f"Your answer for Question {i+1}", use_container_width=True)
                    
                    # Use Gemini Vision to analyze (extract + grade) in one call
                    with st.spinner(f"Gemini AI is analyzing answer for Question {i+1}..."):
                        analysis_result = analyze_answer_with_gemini_vision(question.question_text, image_path, question.points)
                        extracted_text = analysis_result["extracted_text"]
                        awarded_score = analysis_result["awarded_score"]
                        feedback = analysis_result["feedback"]
                    
                    # Show text extracted by Gemini
                    st.markdown("##### Text Extracted by Gemini Vision:")
                    st.text_area("Extracted Text:", extracted_text if extracted_text else "Gemini could not extract any text.", height=100, disabled=True, key=f"res_text_{question.id}")
                    
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
    st.header("📊 View All Submissions")
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
                        st.markdown("##### Student's Answer (Extracted by Gemini)")
                        st.text_area("Extracted Text:", ans.extracted_text, height=200, disabled=True, key=f"text_{ans.id}")
                    with col2:
                        st.markdown("##### Ideal Answer (from Engine)")
                        st.text_area("Ideal Answer:", ans.question.correct_answer, height=200, disabled=True, key=f"ideal_{ans.id}")
                    st.info(f"**AI Feedback:** {ans.feedback}")
                    st.success(f"**Score Awarded for this Question:** {ans.awarded_score:.2f}")
                    st.markdown("---")
    session.close()

# Admin panel uses the original page selectbox above
if page == "Admin":
    st.header("🔧 Admin Panel")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Submissions")
        session = Session()
        submission_count = session.query(Submission).count()
        answer_count = session.query(Answer).count()
        session.close()
        
        st.metric("Total Submissions", submission_count)
        st.metric("Total Answers", answer_count)
        
        if st.button("Clear All Submissions", type="secondary"):
            session = Session()
            try:
                session.query(Answer).delete()
                session.query(Submission).delete()
                session.commit()
                st.success("All submissions cleared successfully!")
                st.rerun()
            except Exception as e:
                session.rollback()
                st.error(f"Error clearing submissions: {e}")
            finally:
                session.close()
    
    with col2:
        st.subheader("Exams")
        session = Session()
        exam_count = session.query(Exam).count()
        question_count = session.query(Question).count()
        session.close()
        
        st.metric("Total Exams", exam_count)
        st.metric("Total Questions", question_count)
        
        if st.button("Clear All Exams", type="secondary"):
            session = Session()
            try:
                session.query(Answer).delete()
                session.query(Submission).delete()
                session.query(Question).delete()
                session.query(Exam).delete()
                session.commit()
                st.success("All exams and submissions cleared successfully!")
                st.rerun()
            except Exception as e:
                session.rollback()
                st.error(f"Error clearing exams: {e}")
            finally:
                session.close()
    
    with col3:
        st.subheader("Uploads Folder")
        if os.path.exists(UPLOAD_FOLDER):
            upload_files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
            st.metric("Uploaded Files", len(upload_files))
            
            if st.button("Clear Uploads Folder", type="secondary"):
                try:
                    for filename in os.listdir(UPLOAD_FOLDER):
                        file_path = os.path.join(UPLOAD_FOLDER, filename)
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    st.success("Uploads folder cleared successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing uploads folder: {e}")
        else:
            st.warning("Uploads folder does not exist")
