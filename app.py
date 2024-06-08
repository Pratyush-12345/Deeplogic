import pdfminer
import pytesseract
from PIL import Image
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from transformers import pipeline
import streamlit as st
import pdfminer.high_level
import pdfminer.layout
import io
import fitz  # PyMuPDF


def extract_text_and_images(file_contents):
    text = ""
    images = []
    with io.BytesIO(file_contents) as file:
        pdf_document = fitz.open("pdf", file)
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            text += page.get_text()
            for img_index, img in enumerate(page.get_images(full=True)):
                img_bytes = pdf_document.extract_image(img[0])['image']
                images.append(Image.open(io.BytesIO(img_bytes)))
    return text, images



# OCR Integration for Images
def perform_ocr_for_images(images):
    image_text = []
    for image in images:
        text = pytesseract.image_to_string(image)
        image_text.append(text)
    return image_text

# Preprocessing
def preprocess_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    sentences = sent_tokenize(cleaned_text)
    tokens = [word_tokenize(sentence) for sentence in sentences]
    return tokens

# LLM Integration
def load_llm_model():
    return pipeline("text-generation", model="gpt2")

# Question Answering
def answer_question(context, question):
    qa_model = pipeline("question-answering")
    result = qa_model(question=question, context=context)
    return result['answer']

# Information Extraction (Using NER as an example)
def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Streamlit UI code
def main():
    st.title("Document Processing Chatbot")
    uploaded_file = st.file_uploader("Upload document", type=['pdf'])
    if uploaded_file is not None:
        file_contents = uploaded_file.read()
        st.write("Document uploaded!")
        st.write("Extracting text and images...")
        text, images = extract_text_and_images(file_contents)
        st.write("Performing OCR for images...")
        image_text = perform_ocr_for_images(images)
        text += " ".join(image_text)
        st.write("Preprocessing text...")
        preprocessed_text = preprocess_text(text)
        st.write("Extracting entities...")
        entities = extract_entities(text)
        st.write("Entities:", entities)
        question = st.text_input("Ask a question about the document:")
        if question:
            st.write("Answering question...")
            answer = answer_question(text, question)
            st.write("Answer:", answer)

if __name__ == "__main__":
    llm_model = load_llm_model()
    main()
