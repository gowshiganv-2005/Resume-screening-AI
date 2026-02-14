import os
import re
import pickle
import PyPDF2
import docx2txt
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Folders
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load AI Models
models_dir = os.path.join(os.path.dirname(__file__), 'models')
tfidf = pickle.load(open(os.path.join(models_dir, 'tfidf.pkl'), 'rb'))
clf = pickle.load(open(os.path.join(models_dir, 'clf.pkl'), 'rb'))
encoder = pickle.load(open(os.path.join(models_dir, 'encoder.pkl'), 'rb'))

TARGET_ROLES_TEXT = {
    "Data Science": "Machine Learning, Python, SQL, Deep Learning, Statistics, Data Analysis, NLP, R, Neural Networks",
    "Web Designing": "HTML, CSS, JavaScript, Figma, UI/UX, Adobe XD, Responsive Design, Bootstrap, Tailwind, Front-end",
    "Java Developer": "Java, Spring Boot, Hibernate, Microservices, SQL, J2EE, Maven, Backend development",
    "HR": "Recruitment, Employee Relations, Talent Management, Payroll, Performance Appraisal, Corporate Communication",
    "Sales": "Business Development, Sales, Client Relationships, Market Research, B2B, Lead Generation, CRM",
    "Mechanical Engineer": "CAD, SolidWorks, Manufacturing, Thermodynamics, Automotive, Robotics, Structural Analysis",
    "Python Developer": "Python, Django, Flask, REST API, SQL, Backend, Algorithms, Scripting, Automation"
}

def clean_resume(resume_text):
    resume_text = re.sub(r'http\S+\s*', ' ', resume_text)
    resume_text = re.sub(r'RT|cc', ' ', resume_text)
    resume_text = re.sub(r'#\S+', '', resume_text)
    resume_text = re.sub(r'@\S+', '  ', resume_text)
    resume_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text)
    resume_text = re.sub(r'\s+', ' ', resume_text)
    return resume_text.strip().lower()

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == '.pdf':
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
    elif ext == '.docx':
        text = docx2txt.process(file_path)
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    return text

def get_gemma_feedback(resume_text, predicted_role, match_percentage):
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("GEMMA_MODEL")
    
    prompt = f"""
    Analyze the following resume text and provide a brief feedback. 
    The system predicted the role as: {predicted_role} with a match percentage of {match_percentage}%.
    
    Resume Text: {resume_text[:1000]}... (truncated)
    
    Provide:
    1. Key Strengths found in the resume.
    2. Missing Keywords for the role of {predicted_role}.
    3. Personalized advice to improve the resume for this role.
    
    Format: Keep it clear and professional. Use bullet points.
    """
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"AI Feedback currently unavailable: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Extract and clean text
    raw_text = extract_text_from_file(file_path)
    cleaned_text = clean_resume(raw_text)
    
    # Predict Category
    input_features = tfidf.transform([cleaned_text])
    prediction_id = clf.predict(input_features)[0]
    predicted_role = encoder.inverse_transform([prediction_id])[0]
    
    # Calculate Match Percentage
    # We compare the resume against the sample keywords for the predicted role
    role_keywords = TARGET_ROLES_TEXT.get(predicted_role, "")
    role_vec = tfidf.transform([clean_resume(role_keywords)])
    similarity = cosine_similarity(input_features, role_vec)[0][0]
    
    # Scale similarity to percentage (simple heuristic)
    match_percentage = round(similarity * 100, 2)
    # Ensure it's not too low for relevant resumes
    match_percentage = min(100, match_percentage * 3) # Simple multiplier for demonstration
    match_percentage = max(10, match_percentage) # Floor
    match_percentage = round(match_percentage, 2)

    # Get AI Feedback from Gemma
    ai_feedback = get_gemma_feedback(cleaned_text, predicted_role, match_percentage)
    
    return jsonify({
        "role": predicted_role,
        "match_percentage": match_percentage,
        "ai_feedback": ai_feedback,
        "filename": file.filename
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
