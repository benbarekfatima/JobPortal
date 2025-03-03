# iWork Job Portal

iWork is a job portal built using HTML, CSS, and Django. It includes a novel recommender model, HGCNL-JRec, for job recommendations. The recommender model uses LDA (gensim) and GNN (pytorch).

## Features

- User authentication and profile management
- Job postings and search functionality
- Advanced job recommendation system using HGCNL-JRec

## Prerequisites

Make sure you have the following installed on your system:

- Python 3.x
- Django
- pip (Python package installer)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/benbarekfatima/JobPortal.git
    cd JobPortal
    ```

2. Activate the virtual environment:
    ```bash
    source job_env/bin/activate  # On Windows use `job_env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1. Apply database migrations:
    ```bash
    python manage.py migrate
    ```

2. Create a superuser:
    ```bash
    python manage.py createsuperuser
    ```

3. Run the development server:
    ```bash
    python manage.py runserver
    ```

4. Open your web browser and go to `http://127.0.0.1:8000/` to access the application.
