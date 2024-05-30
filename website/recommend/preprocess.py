from pdfminer.high_level import extract_text
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
import spacy
from . import global_vars



def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def remove_html_tags(text):
    pattern = r'<(.*?)>'
    return re.sub(pattern, ' ', text)

def remove_html_entities(text):
    pattern = r'&\w+'
    return re.sub(pattern, ' ', text)

def replace_special_characters(text):
    pattern = r'[;:]|(\\r)|(\\n)'
    return re.sub(pattern, ' ', text)

def remove_extra_spaces(text):
    pattern = r'\s\s+?(?=\S)'
    return re.sub(pattern, ' ', text)

def replace_punctuation(text):
    punctuation = '!"#$%&\'()*,-./:;<=>?@[\\]^_`{|}~'
    return text.translate(str.maketrans(punctuation, ' ' * len(punctuation)))

def remove_non_ascii(text):
    return ''.join(char if ord(char) < 128 else ' ' for char in text)

def clean_text(text):
    text = remove_html_tags(text)
    text = remove_html_entities(text)
    text = replace_special_characters(text)
    text = replace_punctuation(text)
    text = remove_non_ascii(text)
    text = remove_extra_spaces(text)
    text = text.lower()
    return text

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def get_skills(text):
    nlp = spacy.load("en_core_web_lg")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.from_disk(global_vars.skill_path)
    doc = nlp(text)
    skills = []
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            skills.append(ent.text)
    return skills

def unique_skills(x):
    return list(set(x))

def get_degree(text):
    nlp = spacy.load("en_core_web_lg")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.from_disk(global_vars.degree_path)
    doc = nlp(text)
    degrees = []
    for ent in doc.ents:
        if ent.label_ == "DEGREE":
            degrees.append(ent.text)
    return degrees

def get_majors(text):
    nlp = spacy.load("en_core_web_lg")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.from_disk(global_vars.major_path)
    doc = nlp(text)
    majors = []
    for ent in doc.ents:
        if ent.label_ == "MAJOR":
            majors.append(ent.text)
    return majors


def classify_text(lda_model, text):
    # Convert the new text to a bag-of-words vector
    new_text_bow = lda_model.id2word.doc2bow(text)

    # Get topic probabilities for the new text
    topic_probs = lda_model.get_document_topics(new_text_bow)  # List of tuples (topic ID, probability)

    # Sort topics by probability and extract the most dominant topic
    dominant_topic = max(topic_probs, key=lambda x: x[1])[0]

    return dominant_topic



def concatenate_features(skills, majors, degrees):
    degree_str = "degree: " + " ".join(degrees) + " . " if degrees else ""
    major_str = "majors: " + " ".join(majors) + " . " if majors else ""
    skills_str = "skills: " + " ".join(skills) + " . " if skills else ""
    return f"{degree_str} {major_str} {skills_str}".strip()



def preprocess_text(text):
    cleantext= clean_text(text)
    cleaner_text= remove_stopwords(cleantext)
    skills=get_skills(cleaner_text)
    majors=get_majors(cleaner_text)
    degrees=get_degree(cleaner_text)
    topic=classify_text(global_vars.LDA_model, skills)
    text_emb=concatenate_features(skills,majors,degrees)
    return text_emb,topic


def preprocess_resume(pdf_path):
    text= extract_text_from_pdf(pdf_path)
    text_emb,topic = preprocess_text(text)
    return text_emb,topic







