# MODULE 0 — CORE SETUP & CONFIG
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
from werkzeug.utils import secure_filename
import time
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sys
sys.stdout.flush()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

app.config['SECRET_KEY'] = 'maheshkumar' 

MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_DB = os.getenv("MYSQL_DB")

app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}"
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
    profile_pic = db.Column(db.String(200), default="profile_pics/default.png")

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

# SKILL & MODEL SETUP

ALL_SKILLS_LIST = [
    "python","web development", "java","spring","spring-boot","c","c++", "C#","SQL","mysql" , 
    "javascript", "sql", "react", "angular", "flask", "django", "aws", "azure", "docker", 
    "kubernetes", "git", "linux", "html", "css", "tensorflow", "pytorch", "communication",
    "leadership", "problem solving", "machine learning", "deep learning", "nlp", "mongodb",
    "numpy", "pandas", "opencv"
]
ALL_SKILLS = set(ALL_SKILLS_LIST)
TECH_SKILLS_AI = set(ALL_SKILLS_LIST)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("WARNING: Spacy model 'en_core_web_sm' not found. Skill extraction will use simple keyword matching.")
    nlp = None

API_KEY = os.getenv("MY_API_KEY")
MODEL_NAME = "models/gemini-2.5-flash"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name=MODEL_NAME)


# COMMON FUNCTIONS USED BY MULTIPLE MODULES
# Gemini, skill extraction, LinkedIn, GitHub, text extraction

def extract_numbered_questions(text):
    pattern = r"(?:^|\n)\s*\d+\s*[\).\-\:]*\s*(.+?)(?=\n\s*\d+\s*[\).\-\:]*\s|\Z)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    cleaned = []
    for m in matches:
        q = m.strip()
        q = re.sub(r"\s+$", "", q)
        cleaned.append(q)
    return cleaned

def safe_generate(prompt, tries=2, delay=1.0):
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
    all_qs = {}
    for skill in skills:
        print(f"\n Generating for skill: {skill}")
        import sys
        sys.stdout.flush()
        prompt = (
            f"You are an interviewer. Create {questions_per_skill} technical interview "
            f"questions for the skill '{skill}'.\n"
            "- Number them from 1 (easy) to 5 (hard).\n"
            "- Keep each question short.\n"
            "- Do NOT provide answers.\n\n"
            "Format:\n1. <question>\n2. <question>\n"
        )
        raw = safe_generate(prompt, tries=3, delay=1.2)
        if not raw:
            all_qs[skill] = []
            continue
        questions = extract_numbered_questions(raw)
        if not questions:
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            heur = [ln for ln in lines if len(ln.split()) > 3]
            questions = heur[:questions_per_skill]
        all_qs[skill] = questions[:questions_per_skill]
    return all_qs

def extract_text(file_stream, filename):
    text = ""
    filename = filename.lower()
    if filename.endswith('.pdf'):
        try:
            reader = PdfReader(file_stream)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception:
            print("Warning: PDF extraction issue.")
    elif filename.endswith('.docx'):
        doc = docx.Document(io.BytesIO(file_stream.read()))
        for para in doc.paragraphs:
            text += para.text + " "
    else:
        text = file_stream.read().decode('utf-8', errors='ignore')
    return text.strip()

def extract_skills(text, comprehensive=True):
    text = text.lower()
    skill_list_to_use = ALL_SKILLS if comprehensive else TECH_SKILLS_AI
    if nlp and not comprehensive:
        doc = nlp(text)
        extracted = set()
        for token in doc:
            if token.text in skill_list_to_use:
                extracted.add(token.text)
        return list(extracted)
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
    github_pattern = r"(https?://github\.com/[A-Za-z0-9_-]+)"
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text = page.get_text()
            match = re.search(github_pattern, text)
            if match:
                return match.group(1)
    return None

def analyze_github_profile(github_link):
    username = github_link.split("github.com/")[-1].strip("/")
    api_url = f"https://api.github.com/users/{username}"
    repos_url = f"https://api.github.com/users/{username}/repos"
    user_data = requests.get(api_url).json()
    repos_data = requests.get(repos_url).json()
    if "Not Found" in str(user_data):
        return {"error": "GitHub profile not found"}
    analysis = {
        "username": user_data.get("login"),
        "name": user_data.get("name"),
        "public_repos": user_data.get("public_repos"),
        "followers": user_data.get("followers"),
        "following": user_data.get("following"),
        "profile_link": user_data.get("html_url"),
        "top_repos": [repo["name"] for repo in sorted(
            repos_data,
            key=lambda x: x["stargazers_count"],
            reverse=True
        )[:5]]
    }
    return analysis


# MODULE 0 — AUTH & GENERAL ROUTES


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
            flash('Login failed. Check email/password.', 'danger')
    return render_template('login_page.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup_page():
    if current_user.is_authenticated:
        return redirect(url_for('main_index'))
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        user_exists = User.query.filter(
            (User.username == name) | (User.email == email)
        ).first()
        if user_exists:
            flash('Username or Email already exists.', 'danger')
            return render_template('signup_page.html')
        new_user = User(username=name, email=email)
        new_user.set_password(password)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Account created! Please login.', 'success')
            return redirect(url_for('login_page'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error during signup: {e}', 'danger')
    return render_template('signup_page.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out!", "info")
    return redirect(url_for('main_index'))

@app.route('/contact', methods=['GET', 'POST'])
def contact_page():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        if not name or not email or not message:
            flash('All fields required.', 'danger')
            return render_template('contact_page.html')
        new_message = ContactMessage(name=name, email=email, message=message)
        try:
            db.session.add(new_message)
            db.session.commit()
            flash('Message sent!', 'success')
            return redirect(url_for('contact_page'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error saving message: {e}', 'danger')
    return render_template('contact_page.html')

@app.route('/help')
def help_page():
    return render_template('help_page.html')

@app.route('/about')
def about_page():
    return render_template('about_page.html')

UPLOAD_FOLDER = "static/profile_pics"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/your_profile', methods=['GET', 'POST'])
@login_required
def your_profile():
    user = current_user
    if request.method == 'POST':
        new_username = request.form.get("username")
        if new_username and new_username != user.username:
            user.username = new_username

        new_email = request.form.get("email")
        if new_email and new_email != user.email:
            user.email = new_email

        if "profile_pic" in request.files:
            image = request.files["profile_pic"]
            if image and image.filename != "":
                filename = secure_filename(image.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                image.save(filepath)
                user.profile_pic = f"profile_pics/{filename}"

        db.session.commit()
        flash("Profile updated!", "success")
        return redirect(url_for("your_profile"))
    login_history = []
    return render_template("your_profile.html", user=user, login_history=login_history)


# MODULE 1 — ATS SCORE


@app.route("/module1/ats-score", methods=["GET", "POST"])
@login_required
def ats_score():
    score, matched, missing, message = None, [], [], ""
    if request.method == "POST":
        job_description = request.form.get("job_description", "")
        resume_file = request.files.get('resume')
        if not job_description or not resume_file or resume_file.filename == '':
            message = "Please provide both Job Description & Resume."
        elif not resume_file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
            message = "Unsupported file format."
        else:
            resume_content = resume_file.read()
            resume_text = extract_text(io.BytesIO(resume_content), resume_file.filename)
            job_skills = extract_skills(job_description, comprehensive=True)
            resume_skills = extract_skills(resume_text, comprehensive=True)
            matched = set(resume_skills).intersection(job_skills)
            missing = set(job_skills).difference(resume_skills)
            score = round((len(matched) / len(job_skills)) * 100, 2) if job_skills else 0
            matched, missing = list(matched), list(missing)
    return render_template("index1.html", score=score, matched=matched, missing=missing, message=message)

# MODULE 2 — GITHUB ANALYZER


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
        return render_template('result2.html', analysis={"error": "No GitHub link found."})
    return render_template('index2.html', error="Upload a PDF or enter a GitHub link.")

@app.route('/analyze', methods=['POST'])
def analyze():
    github_link = request.form.get('github_link')
    resume = request.files.get('resume')
    if github_link:
        return render_template('result2.html', analysis=analyze_github_profile(github_link))
    if resume:
        file_path = f"uploads/{resume.filename}"
        resume.save(file_path)
        extracted_link = extract_github_link_from_pdf(file_path)
        if extracted_link:
            return render_template('result2.html', analysis=analyze_github_profile(extracted_link))
        return "No GitHub link found."
    return "Please upload a resume or enter a GitHub link."

@app.route('/result2')
def result():
    return render_template('result2.html')


# MODULE 3 — AI QUESTIONS


@app.route("/module3/ai-questions", methods=["GET", "POST"])
@login_required
def ai_questions():
    if request.method == "POST":
        resume = request.files.get("resume")
        if not resume or resume.filename == '':
            return render_template("index3.html", error="Please upload a resume.")
        resume_content = resume.read()
        text = extract_text(io.BytesIO(resume_content), resume.filename)
        if not text.strip():
            return render_template("index3.html", error="Couldn't read file.")
        skills = extract_skills(text, comprehensive=False)
        if not skills:
            return render_template("index3.html", error="No core skills found.")
        return render_template("index3.html", skills=skills, questions=generate_interview_questions(skills))
    return render_template("index3.html")


# MODULE 4 — LINKEDIN DASHBOARD

@app.route("/module4/dashboard", methods=["GET", "POST"])
@login_required
def dashboard_analyze():
    if request.method == 'GET':
        return render_template('index4.html')
    name = request.form.get('name')
    linkedin_manual = request.form.get('linkedin_url')
    resume = request.files.get('resume')
    if not resume or resume.filename == '':
        return render_template('index4.html', error="Upload a resume.")
    if not (resume.filename.lower().endswith('.pdf') or resume.filename.lower().endswith('.docx')):
        return render_template('index4.html', error="Only PDF or DOCX allowed.")
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


# MODULE 5 — PLAGIARISM CHECKER

PLAGIARISM_UPLOAD_FOLDER = "uploads"
PLAGIARISM_DATASET_FOLDER = "resume_folder"
os.makedirs(PLAGIARISM_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLAGIARISM_DATASET_FOLDER, exist_ok=True)

plag_model = SentenceTransformer('all-MiniLM-L6-v2')
print(" Plagiarism detection model loaded successfully!")

import pickle

CACHE_FILE = "cached_embeddings.pkl"

if os.path.exists(CACHE_FILE):
    cached = pickle.load(open(CACHE_FILE, "rb"))
    dataset_embeddings = cached["embeddings"]
    resume_files = cached["files"]
    print(f" Loaded cached embeddings ({len(resume_files)} resumes).")
else:
    print("⚠️ Cache file not found. Please run build_cache.py first.")
    dataset_embeddings = None
    resume_files = []

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"⚠️ Error reading PDF ({pdf_path}): {e}")
    return text

def check_plagiarism(uploaded_path):
    uploaded_text = extract_text_from_pdf(uploaded_path)
    if len(uploaded_text.strip()) < 50:
        return 0, " The uploaded file contains too little readable text."
    if dataset_embeddings is None:
        return 0, " No reference resumes available."
    uploaded_embedding = plag_model.encode(uploaded_text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(uploaded_embedding, dataset_embeddings)[0]
    max_score = float(cosine_scores.max())
    plagiarism_percent = round(max_score * 100, 2)
    if plagiarism_percent > 80:
        msg = " High plagiarism detected!"
    elif plagiarism_percent > 50:
        msg = " Moderate similarity detected."
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

# APP RUNNER
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
