import streamlit as st
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from googletrans import Translator
from dotenv import load_dotenv
import tempfile

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR'  # Update this path based on your Tesseract installation

translator = Translator()

# Set the path to the Poppler binaries directory
poppler_path = r'C:\Program Files\poppler-0.68.0\bin'  # Update this with the correct path

# Add the Poppler directory to the system PATH variable
os.environ["PATH"] += os.pathsep + poppler_path

def translate_text(text, dest_lang='en'):
    translated = translator.translate(text, dest=dest_lang)
    return translated.text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf.read())
            temp_file_path = temp_file.name

        # Extract text using PyPDF2
        pdf_reader = PyPDF2.PdfReader(temp_file_path)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

        # Convert PDF to images for OCR
        images = convert_from_path(temp_file_path)
        for image in images:
            text += pytesseract.image_to_string(image)
        
        # Remove the temporary file
        os.remove(temp_file_path)
            
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer 
    is not in the provided context, just say, "Answer is not available in the given context". Don't provide a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def extract_information(text):
    prompt = f"Extract entities (names, dates, locations, organizations) and summarize the following text:\n\n{text}\n\nEntities and Summary:"
    response = genai.text(prompt=prompt, model="gemini-pro")
    return response.result

def classify_document(text):
    prompt = f"Classify the following document into predefined categories based on its content:\n\n{text}\n\nCategory:"
    response = genai.text(prompt=prompt, model="gemini-pro")
    return response.result

def translate_internal(text, target_language='en'):
    prompt = f"Translate the following text into {target_language}:\n\n{text}"
    response = genai.text(prompt=prompt, model="gemini-pro")
    return response.result

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("REPLY: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat With Multiple PDFs 📃")
    st.header("Chat With PDFs 📃 Using Gemini 🕵️‍♀️")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and click on the Submit and process", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    translated_text = translate_internal(raw_text, target_language='en')
                    st.text_area("Translated Text", translated_text, height=300)
                    
                    extracted_info = extract_information(translated_text)
                    st.text_area("Extracted Information", extracted_info, height=300)
                    
                    document_category = classify_document(translated_text)
                    st.write(f"Document Category: {document_category}")
                    
                    text_chunks = get_text_chunks(translated_text)
                    get_vector_store(text_chunks)
                    st.success("Processing Done")

    st.subheader("Ask Questions About Your Documents")
    user_question = st.text_input("ASK a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    st.subheader("Extract Specific Information")
    info_type = st.selectbox("Select Information Type", ["Entities and Summary", "Document Category", "Custom"])
    if info_type == "Custom":
        custom_prompt = st.text_area("Enter Your Custom Extraction Prompt")
        if st.button("Extract Information"):
            custom_response = genai.text(prompt=custom_prompt, model="gemini-pro").result
            st.text_area("Custom Extracted Information", custom_response, height=300)
    elif st.button("Extract Information"):
        if pdf_docs:
            raw_text = get_pdf_text(pdf_docs)
            translated_text = translate_internal(raw_text, target_language='en')
            if info_type == "Entities and Summary":
                extracted_info = extract_information(translated_text)
                st.text_area("Extracted Information", extracted_info, height=300)
            elif info_type == "Document Category":
                document_category = classify_document(translated_text)
                st.write(f"Document Category: {document_category}")

if __name__ == "__main__":
    main()
