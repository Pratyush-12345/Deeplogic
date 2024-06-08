# Deeplogic
# Chat With PDFs 📃 Using Gemini 🕵️‍♀️

This project provides a Streamlit-based application to interact with multiple PDF documents using Google Generative AI. Users can upload PDF documents, ask questions, request specific information to be extracted, and review the extracted data.

## Features

1. **Upload Documents for Processing**: Upload multiple PDF files for text extraction and processing.
2. **Ask Questions About Document Content**: Input questions related to the content of the uploaded PDFs.
3. **Request Specific Information to Be Extracted**: Extract entities like names, dates, locations, organizations, and summaries from the text.
4. **Review Extracted Data and Provide Feedback**: Review the extracted information and provide custom prompts for information extraction.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/chat-with-pdfs.git
    cd chat-with-pdfs
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install Tesseract OCR:
    - **Windows**: Download the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
    

4. Set up your Google Generative AI API key:
    - Create a `.env` file in the project root and add your API key:
        ```env
        GOOGLE_API_KEY=your_google_api_key_here
        ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Use the sidebar to upload PDF files and process them.

3. Interact with the main interface to:
    - Ask questions about the PDF content.
    - Extract specific information like entities and summaries.
    - Review extracted data and provide custom extraction prompts.

## Notes

- Ensure that Tesseract OCR is correctly installed and configured on your system.
- The application uses Google Generative AI's "gemini-pro" model for advanced natural language processing tasks.

## License

This project is licensed under the MIT License.
