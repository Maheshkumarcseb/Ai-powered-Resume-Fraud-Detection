import io
import re
import os
import requests
from flask import Flask, render_template, request, redirect, url_for,flash, session
from flask_login import login_required, current_user 
from sentence_transformers import SentenceTransformer, util
import glob
import fitz 
from PyPDF2 import PdfReader
import docx
import fitz 
import spacy
from bs4 import BeautifulSoup 

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

app.config['SECRET_KEY'] = 'maheshkumar' 

MYSQL_USER = 'root'      
MYSQL_PASSWORD = '12345' 
MYSQL_HOST = 'localhost' 
MYSQL_DB = 'career_toolkit_db'

app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+pymysql://root:12345@localhost/career_toolkit_db"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'
login_manager.login_message_category = "warning"

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

ALL_SKILLS_LIST = [
    "python","web development", "java","spring","spring-boot","c","c++", "C#","SQL","mysql" , "javascript", "sql", "react", "angular", "flask", 
    "django", "aws", "azure", "docker", "kubernetes", "git", "linux", "html", "css",
    "tensorflow", "pytorch", "communication", "leadership", "problem solving", 
    "machine learning", "deep learning", "nlp", "mongodb", "numpy", "pandas", "opencv"
]
ALL_SKILLS = set(ALL_SKILLS_LIST)
TECH_SKILLS_AI = set(ALL_SKILLS_LIST) 


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("WARNING: Spacy model 'en_core_web_sm' not found. Skill extraction will use simple keyword matching.")
    nlp = None


import re
import time
import google.generativeai as genai

API_KEY = "your api key"   # <-- your Gemini API key
MODEL_NAME = "models/gemini-2.5-flash"  

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name=MODEL_NAME)

def extract_numbered_questions(text):
    """Extracts clean numbered questions from Gemini output."""
    pattern = r"(?:^|\n)\s*\d+\s*[\).\-\:]*\s*(.+?)(?=\n\s*\d+\s*[\).\-\:]*\s|\Z)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    cleaned = []
    for m in matches:
        q = m.strip()
        q = re.sub(r"\s+$", "", q)
        cleaned.append(q)
    return cleaned

def safe_generate(prompt, tries=2, delay=1.0):
    """Retry Gemini generation safely."""
    for attempt in range(tries):
        try:
            resp = model.generate_content(prompt)
            text = resp.text if hasattr(resp, "text") else str(resp)
            return text.strip()
        except Exception as e:
            print(f" Gemini generation error (attempt {attempt+1}/{tries}): {e}")
            if attempt + 1 < tries:
                time.sleep(delay)
    return ""

def generate_interview_questions(skills, questions_per_skill=5):
    """
    Generate realistic interview questions using Gemini.
    Returns: dict of skill -> list of questions.
    """
    all_qs = {}
    for skill in skills:
        print(f"\n Generating for skill: {skill}")
        prompt = (
            f"You are an interviewer. Create {questions_per_skill} technical interview "
            f"questions for the skill '{skill}'.\n"
            "- Number them from 1 (easy) to 5 (hard).\n"
            "- Keep each question short and focused on practical / problem-solving.\n"
            "- Do NOT provide answers.\n"
            "- Use concise, professional wording.\n\n"
            "Format:\n1. <question>\n2. <question>\n...\n"
        )

        raw = safe_generate(prompt, tries=3, delay=1.2)
        if not raw:
            print(f"No response for {skill}.")
            all_qs[skill] = []
            continue

        questions = extract_numbered_questions(raw)
        if not questions:
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            heur = [ln for ln in lines if len(ln.split()) > 3]
            questions = heur[:questions_per_skill]
        all_qs[skill] = questions[:questions_per_skill]

        if not all_qs[skill]:
            print(f"⚠️ Could not parse Gemini output for {skill}:\n{raw}\n{'-'*40}")
    return all_qs





def compute_plagiarism_score_mock(resume_text):
    if 'resume template' in resume_text.lower():
        return [{"template_text": "Mock Template 1 (Highly Similar)...", "similarity": 85.2}]
    return []
compute_plagiarism_score = compute_plagiarism_score_mock



def extract_text(file_stream, filename):
    """Extract text from PDF or DOCX file object/stream."""
    text = ""
    filename = filename.lower()
    if filename.endswith('.pdf'):
        try:
            reader = PdfReader(file_stream)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception:
             print("Warning: PDF extraction complexity. Using PyPDF2 fallback.")
    elif filename.endswith('.docx'):
        
        doc = docx.Document(io.BytesIO(file_stream.read()))
        for para in doc.paragraphs:
            text += para.text + " "
    else:
        text = file_stream.read().decode('utf-8', errors='ignore')
        
    return text.strip()


def extract_skills(text, comprehensive=True):
    """Extract skills using Spacy (if available) or simple keyword matching."""
    text = text.lower()
    skill_list_to_use = ALL_SKILLS if comprehensive else TECH_SKILLS_AI
    if nlp and not comprehensive: 
        doc = nlp(text)
        extracted = set()
        for token in doc:
            if token.text in skill_list_to_use:
                extracted.add(token.text)
        return list(extracted)
    else:
        return list(set(skill for skill in skill_list_to_use if skill in text))



def extract_linkedin_url(text):
    pattern = r"https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9\-_]+"
    matches = re.findall(pattern, text)
    return matches[0] if matches else None

def validate_linkedin_url(url):
    try:
        if "linkedin.com/in/" in url:
            return {"valid": True, "title": "Mock Profile Title"}
        return {"valid": False}
    except Exception:
        return {"valid": False}

def compute_profile_score(linkedin_data, resume_skills):
    base_score = 50
    if linkedin_data.get("valid"):
        base_score += 15
    base_score += min(len(resume_skills) * 2, 35)
    return min(int(base_score), 100)

def extract_github_link_from_pdf(file_path):
    """Extract GitHub link from resume PDF"""
    github_pattern = r"(https?://github\.com/[A-Za-z0-9_-]+)"
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text = page.get_text()
            match = re.search(github_pattern, text)
            if match:
                return match.group(1)
    return None

def analyze_github_profile(github_link):
    """Analyze GitHub profile using GitHub REST API"""
    username = github_link.split("github.com/")[-1].strip("/")
    api_url = f"https://api.github.com/users/{username}"
    repos_url = f"https://api.github.com/users/{username}/repos"
    user_data = requests.get(api_url).json()
    repos_data = requests.get(repos_url).json()
    if "message" in user_data and user_data["message"] == "Not Found":
        return {"error": "GitHub profile not found"}
    analysis = {
        "username": user_data.get("login"),
        "name": user_data.get("name"),
        "public_repos": user_data.get("public_repos"),
        "followers": user_data.get("followers"),
        "following": user_data.get("following"),
        "profile_link": user_data.get("html_url"),
        "top_repos": [repo["name"] for repo in sorted(repos_data, key=lambda x: x["stargazers_count"], reverse=True)[:5]]
    }
    return analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    github_link = request.form.get('github_link')
    resume = request.files.get('resume')
    if github_link:
        analysis = analyze_github_profile(github_link)
        return render_template('result2.html', analysis=analysis)
    if resume:
        file_path = f"uploads/{resume.filename}"
        resume.save(file_path)
        extracted_link = extract_github_link_from_pdf(file_path)
        if extracted_link:
            analysis = analyze_github_profile(extracted_link)
            return render_template('result2.html', analysis=analysis)
        else:
            return "No GitHub link found in resume."
    return "Please upload a resume or enter a GitHub link."

@app.route('/result2')
def result():
    return render_template('result2.html')

@app.route('/')
def main_index():
    return render_template('index.html', logged_in=current_user.is_authenticated)

@app.route('/login', methods=['GET','POST'])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('main_index'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('main_index'))
        else:
            flash('Login failed. Check your email and password.', 'danger')
    else:
        if request.args.get('next'):
            flash('Please log in to access this module.', 'info')
    return render_template('login_page.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup_page():
    if current_user.is_authenticated:
        return redirect(url_for('main_index'))
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        user_exists = User.query.filter((User.username == name) | (User.email == email)).first()
        if user_exists:
            flash('Username or Email already exists.', 'danger')
            return render_template('signup_page.html')
        new_user = User(username=name, email=email)
        new_user.set_password(password)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login_page'))
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred during signup: {e}', 'danger')
    return render_template('signup_page.html')

@app.route('/logout')
@login_required
def logout():
    logout_user() 
    flash("You have been logged out successfully!", "info")
    return redirect(url_for('main_index'))  # Redirect to homepage or login page


@app.route('/contact', methods=['GET', 'POST'])
def contact_page():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        if not name or not email or not message:
            flash('All fields are required.', 'danger')
            return render_template('contact_page.html')
        new_message = ContactMessage(name=name, email=email, message=message)
        try:
            db.session.add(new_message)
            db.session.commit()
            flash('Thank you! Your message has been sent successfully.', 'success')
            return redirect(url_for('contact_page')) 
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred: Could not save message. {e}', 'danger') 
    return render_template('contact_page.html')

@app.route('/help')
def help_page():
    return render_template('help_page.html')

@app.route('/about')
def about_page():
    return render_template('about_page.html')


@app.route('/your_profile', methods=['GET', 'POST'])
@login_required
def your_profile():
    """
    Display and optionally update the current user's profile.
    """
    if request.method == 'POST':
       
        new_email = request.form.get('email')
        if new_email:
            current_user.email = new_email
            try:
                db.session.commit()
                flash("Email updated successfully!", "success")
            except Exception as e:
                db.session.rollback()
                flash(f"Failed to update email: {e}", "danger")
        return redirect(url_for('your_profile'))
    return render_template('your_profile.html', user=current_user)

@app.route("/module1/ats-score", methods=["GET", "POST"])
@login_required
def ats_score():
    score, matched, missing, message = None, [], [], ""
    if request.method == "POST":
        job_description = request.form.get("job_description", "")
        resume_file = request.files.get('resume')
        if not job_description or not resume_file or resume_file.filename == '':
            message = "Please provide both a Job Description and a Resume file."
        elif not resume_file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
            message = "Unsupported file format. Upload PDF, DOCX, or TXT."
        else:
            resume_content = resume_file.read()
            resume_text = extract_text(io.BytesIO(resume_content), resume_file.filename)
            job_skills = extract_skills(job_description, comprehensive=True)
            resume_skills = extract_skills(resume_text, comprehensive=True)
            matched = set(resume_skills).intersection(job_skills)
            missing = set(job_skills).difference(resume_skills)
            score = (len(matched) / len(job_skills)) * 100 if job_skills else 0
            score = round(score, 2)
            matched = list(matched)
            missing = list(missing)
    return render_template("index1.html", score=score, matched=matched, missing=missing, message=message)

@app.route('/module2/github-analyzer', methods=['GET', 'POST'])
@login_required
def github_analyzer():
    if request.method == 'GET':
        return render_template('index2.html')
    github_link = request.form.get('github_link')
    resume = request.files.get('resume')
    if github_link:
        analysis = analyze_github_profile(github_link)
        return render_template('result2.html', analysis=analysis)
    if resume and resume.filename != '' and resume.filename.lower().endswith('.pdf'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        resume.save(file_path) 
        extracted_link = extract_github_link_from_pdf(file_path)
        os.remove(file_path)
        if extracted_link:
            analysis = analyze_github_profile(extracted_link)
            return render_template('result2.html', analysis=analysis)
        else:
            return render_template('result2.html', analysis={"error": "No GitHub link found in the PDF resume."})
    return render_template('index2.html', error="Please upload a PDF resume or enter a GitHub link.")

@app.route("/module3/ai-questions", methods=["GET", "POST"])
@login_required
def ai_questions():
    if request.method == "POST":
        resume = request.files.get("resume")
        if not resume or resume.filename == '':
            return render_template("index3.html", error="Please upload a resume file.")
        resume_content = resume.read()
        text = extract_text(io.BytesIO(resume_content), resume.filename)
        if not text.strip():
            return render_template("index3.html", error="Couldn't read text from the file.")
        skills = extract_skills(text, comprehensive=False) 
        if not skills:
            return render_template("index3.html", error="No core technical skills found for question generation.")       
        questions = generate_interview_questions(skills)
        return render_template("index3.html", skills=skills, questions=questions)
    return render_template("index3.html")

@app.route("/module4/dashboard", methods=["GET", "POST"])
@login_required
def dashboard_analyze():
    if request.method == 'GET':
        return render_template('index4.html')
    name = request.form.get('name')
    linkedin_manual = request.form.get('linkedin_url')
    resume = request.files.get('resume')
    if not resume or resume.filename == '':
        return render_template('index4.html', error="Please upload a resume.")
    if not (resume.filename.lower().endswith('.pdf') or resume.filename.lower().endswith('.docx')):
        return render_template('index4.html', error="Only PDF and DOCX files are allowed.")
    resume_content = resume.read()
    text = extract_text(io.BytesIO(resume_content), resume.filename)

    
    linkedin_extracted = extract_linkedin_url(text)
    linkedin_url = linkedin_manual or linkedin_extracted
    linkedin_data = validate_linkedin_url(linkedin_url) if linkedin_url else {"valid": False}

    
    resume_skills = extract_skills(text, comprehensive=False) 
    
    linkedin_skills = ["python", "sql", "html", "css", "flask", "django"] 
    matched = list(set(resume_skills) & set(linkedin_skills))
    missing = list(set(linkedin_skills) - set(resume_skills))

   
    score = compute_profile_score(linkedin_data, resume_skills)

    return render_template('dashboard.html',
                           name=name,
                           linkedin_url=linkedin_url,
                           linkedin_valid=linkedin_data.get("valid"),
                           linkedin_title=linkedin_data.get("title"),
                           resume_skills=resume_skills,
                           linkedin_skills=linkedin_skills,
                           matched=matched,
                           missing=missing,
                           score=score)
 


PLAGIARISM_UPLOAD_FOLDER = "uploads"
PLAGIARISM_DATASET_FOLDER = "resume_folder"
os.makedirs(PLAGIARISM_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLAGIARISM_DATASET_FOLDER, exist_ok=True)


plag_model = SentenceTransformer('all-MiniLM-L6-v2')
print(" Plagiarism detection model loaded successfully!")


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"⚠️ Error reading PDF ({pdf_path}): {e}")
    return text


dataset_resumes, resume_files = [], []
for pdf_file in glob.glob(os.path.join(PLAGIARISM_DATASET_FOLDER, "*.pdf")):
    text = extract_text_from_pdf(pdf_file)
    if len(text.strip()) > 50:
        dataset_resumes.append(text)
        resume_files.append(os.path.basename(pdf_file))
    else:
        print(f" Skipped empty/unreadable file: {pdf_file}")

if dataset_resumes:
    dataset_embeddings = plag_model.encode(dataset_resumes, show_progress_bar=True, convert_to_tensor=True)
    print(f" Loaded and encoded {len(dataset_resumes)} dataset resumes for plagiarism checking.")
else:
    dataset_embeddings = None
    print(" No valid resumes found in 'resume_folder'. Plagiarism detection will not work correctly.")

# Core plagiarism checking function
def check_plagiarism(uploaded_path):
    uploaded_text = extract_text_from_pdf(uploaded_path)
    if len(uploaded_text.strip()) < 50:
        return 0, " The uploaded file contains too little readable text for analysis."

    if dataset_embeddings is None:
        return 0, " No reference resumes available. Please add files to 'resume_folder'."

    uploaded_embedding = plag_model.encode(uploaded_text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(uploaded_embedding, dataset_embeddings)[0]
    max_score = float(cosine_scores.max())
    plagiarism_percent = round(max_score * 100, 2)

    if plagiarism_percent > 80:
        msg = " High plagiarism detected! This resume closely matches existing resumes."
    elif plagiarism_percent > 50:
        msg = " Moderate similarity detected. Review recommended."
    else:
        msg = " Resume appears original."

    return plagiarism_percent, msg


@app.route('/module5/plagiarism', methods=['GET', 'POST'])
@login_required
def plagiarism_checker():
    if request.method == 'POST':
        uploaded_file = request.files.get('resume')
        if not uploaded_file or uploaded_file.filename == '':
            return render_template("index5.html", error="⚠️ No file uploaded.")
        
      
        filepath = os.path.join(PLAGIARISM_UPLOAD_FOLDER, uploaded_file.filename)
        uploaded_file.save(filepath)

     
        plagiarism_percent, msg = check_plagiarism(filepath)

     
        os.remove(filepath)

        
        return render_template('result.html',
                               uploaded=uploaded_file.filename,
                               percentage=plagiarism_percent,
                               message=msg)
    
    
    return render_template('index5.html')


if __name__ == "__main__":
    with app.app_context():
        db.create_all() 
    app.run(debug=True)