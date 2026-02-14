import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle
import os

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')

def clean_resume(resume_text):
    resume_text = re.sub(r'http\S+\s*', ' ', resume_text)  # remove URLs
    resume_text = re.sub(r'RT|cc', ' ', resume_text)  # remove RT and cc
    resume_text = re.sub(r'#\S+', '', resume_text)  # remove hashtags
    resume_text = re.sub(r'@\S+', '  ', resume_text)  # remove mentions
    resume_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)  # remove punctuations
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text) 
    resume_text = re.sub(r'\s+', ' ', resume_text)  # remove extra whitespace
    return resume_text.lower()

# Synthetic Dataset for initial training
data = {
    'Category': [
        'Data Science', 'Data Science', 'Data Science',
        'Web Designing', 'Web Designing', 'Web Designing',
        'Java Developer', 'Java Developer', 'Java Developer',
        'HR', 'HR', 'HR',
        'Sales', 'Sales', 'Sales',
        'Mechanical Engineer', 'Mechanical Engineer', 'Mechanical Engineer',
        'Python Developer', 'Python Developer', 'Python Developer'
    ],
    'Resume': [
        'Data scientist with experience in machine learning, python, sql, and deep learning. Worked on NLP projects.',
        'Experienced data analyst skilled in python, pandas, numpy and scikit-learn. Expertise in data visualization.',
        'Machine learning engineer with focus on computer vision and neural networks using pytorch and tensorflow.',
        'Creative web designer with skills in HTML, CSS, JavaScript, and Figma. UI/UX design experience.',
        'Front-end developer proficient in React, Tailwind CSS, and web page performance optimization.',
        'Web designer specializing in responsive layouts, bootstrap, and modern graphic design tools.',
        'Java developer with expertise in Spring Boot, Hibernate, and Microservices architecture.',
        'Senior backend engineer skilled in Java, SQL, and enterprise application development.',
        'Software developer proficient in Java, J2EE, and Maven for high-scale applications.',
        'HR professional with experience in recruitment, employee relations, and talent management.',
        'Human resources manager skilled in payroll, performance appraisal, and corporate communications.',
        'Talent acquisition specialist with focus on technical hiring and employee onboarding.',
        'Sales executive with a proven track record in business development and client relationship management.',
        'Passionate sales manager skilled in market research, strategic planning, and revenue growth.',
        'Business development associate with expertise in B2B sales and lead generation.',
        'Mechanical engineer experienced in CAD, SolidWorks, and manufacturing processes.',
        'Design engineer specializing in thermodynamics, automotive systems, and structural analysis.',
        'Mechanical maintenance engineer with focus on industrial machinery and robotics.',
        'Python developer skilled in Django, Flask, and REST APIs. Strong algorithmic background.',
        'Full stack python developer proficient in scripting, automation, and database management.',
        'Backend developer with expertise in Python, SQLAlchemy, and cloud deployment.'
    ]
}

df = pd.DataFrame(data)
df['Resume'] = df['Resume'].apply(lambda x: clean_resume(x))

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(df['Resume'])
X = tfidf.transform(df['Resume'])

# Training
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X, df['Category'])

# Save models
models_path = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.exists(models_path):
    os.makedirs(models_path)

pickle.dump(tfidf, open(os.path.join(models_path, 'tfidf.pkl'), 'wb'))
pickle.dump(clf, open(os.path.join(models_path, 'clf.pkl'), 'wb'))
pickle.dump(le, open(os.path.join(models_path, 'encoder.pkl'), 'wb'))

print("Model training complete and saved to models/ folder.")
