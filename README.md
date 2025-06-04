# iWork Job Portal

iWork is a job portal built using HTML, CSS, and Django, featuring a cutting-edge AI-powered job recommender system. The recommender, **HGCNL-JRec**, leverages both Natural Language Processing (NLP) and Graph Neural Networks (GNN) to provide highly relevant job recommendations to users based on their resumes and job interactions.

---

## 🚀 Key Features

- User authentication and profile management (Employer/Employee roles)
- Job postings, search, and application workflows
- **AI-powered job recommendation system (HGCNL-JRec)**
- Resume parsing and skill extraction using NLP
- Admin dashboard (Django admin)
- Responsive web UI with custom templates

---

## 🧠 AI Recommender System: HGCNL-JRec

### Overview

The heart of this portal is the **HGCNL-JRec** recommender, a hybrid model that combines:
- **LDA (Latent Dirichlet Allocation)** for topic modeling (via `gensim`)
- **Graph Neural Networks (GNN)** for learning from user-job-application relationships (via `PyTorch`)
- **spaCy** with custom entity rulers for extracting skills, degrees, and majors from resumes and job descriptions

### How It Works

1. **Resume & Job Parsing:**
   - When an employee uploads a PDF resume, it is parsed using `pdfminer` and cleaned with NLP techniques.
   - Skills, degrees, and majors are extracted using spaCy and custom entity rulers (`website/recommend/recFiles/`).
   - Job descriptions are similarly processed.

2. **Feature Embedding:**
   - Extracted features are embedded into a unified representation.
   - LDA assigns a topic to each resume/job for semantic grouping.

3. **Graph Construction:**
   - Users, jobs, and their interactions (applications) are represented as nodes and edges in a heterogeneous graph.
   - The graph is stored and updated in `website/recommend/recFiles/web_graph.pt`.

4. **Recommendation:**
   - The GNN model (HGCNL-JRec) learns from the graph structure and content features.
   - For a given user, the model recommends jobs they are most likely to be interested in, excluding those already applied to.
   - The top-K recommendations are returned and displayed in the UI.

### Code & Artifacts
- **Model code:** `website/recommend/recommeder.py`, `graph.py`, `preprocess.py`
- **Pretrained models and rulers:** `website/recommend/recFiles/`
- **Global variables:** `website/recommend/global_vars.py` (paths to models, rulers, and graph)

### Dependencies
- `torch`, `gensim`, `spacy`, `pdfminer`, `nltk`, and more (see `requirements.txt`)
- Pretrained spaCy entity rulers and LDA models are included in the repo for immediate use

### Customization & Extension
- You can retrain the LDA or GNN models with your own data
- Entity rulers for skills, degrees, and majors can be extended for new domains
- The recommender logic is modular and can be adapted for other recommendation tasks

---

## 🗂️ Project Structure

- `website/` - Main Django app (views, models, forms, templates, static files)
- `website/recommend/` - Recommender system code and models
- `employee_resumes/` - Uploaded resumes (PDF)
- `Notebooks/` - Jupyter notebooks for data processing and experiments
- `job_portal/` - Django project settings and configuration
- `requirements.txt` - All dependencies (Django, PyTorch, gensim, spaCy, etc.)

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/benbarekfatima/JobPortal.git
    cd JobPortal
    ```
2. **Create and activate a virtual environment:**
    ```bash
    python -m venv job_env
    # On Windows:
    job_env\Scripts\activate
    # On Unix/Mac:
    source job_env/bin/activate
    ```
3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Download NLTK stopwords (if not auto-downloaded):**
    ```python
    python -c "import nltk; nltk.download('stopwords')"
    ```
5. **Apply migrations and create a superuser:**
    ```bash
    python manage.py migrate
    python manage.py createsuperuser
    ```
6. **Run the development server:**
    ```bash
    python manage.py runserver
    ```
7. **Access the app:**  
   Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

---

## 🌐 Deployment

- **Vercel:** Configured via `vercel.json` to deploy using Python 3.9 and the Django WSGI entry point.

---

## 📄 Notes

- **Resume Uploads:** Only PDF files are accepted.
- **Skills/majors/degrees extraction:** Uses custom spaCy entity rulers (see `website/recommend/recFiles/`).
- **Admin:** Access Django admin at `/admin/` after creating a superuser.
- **Customization:** You can extend the recommender, add new features, or modify the UI via the templates.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
