# main_ai_models


Gemini AI Models - Python Implementation

Welcome to the Gemini AI Models repository, where we provide a Python implementation of various AI models, leveraging advanced tools like Google Generative AI, OpenAI, LangChain, and more. This project offers multiple capabilities for working with text, images, audio, and documents such as PDFs, Word files, and CSVs, all wrapped in easy-to-use classes.

Features

1. Gemini Models
Text Model: Utilizes Google's gemini-1.5-pro model to generate detailed text responses. The model is highly flexible and can be used in different scenarios, including summarization, content generation, and more.
Video Response: This feature allows you to upload video files, and the model generates a text-based response based on video content.
Image Response: Upload images, and the model will interpret and provide a text-based response.
Audio Response: Allows you to upload audio files and convert them into text using advanced AI models.
2. OpenAI Integration
Text Generation: Supports OpenAI's GPT-4o model to generate high-quality text responses based on input questions.
Document Processing: Allows for reading and processing Word documents using the OpenAI model, providing context-aware answers based on document content.
3. Document Handling
Word Documents: Extracts and processes text from Word documents (.docx) and provides intelligent responses based on the document content.
PDF Documents: Processes PDF files and chunks the text for effective querying, using embeddings for efficient document retrieval and QA chain responses.
CSV Files: Uses the create_csv_agent method to answer questions based on CSV file content.
4. AI-Powered Data Processing
PandasAI Integration: Easily analyze Excel files by loading them into a DataFrame and querying the data with natural language prompts.
CSV Analysis: Leverage the LangChain CSVLoader to load, process, and analyze CSV files.
Document Splitting: Uses RecursiveCharacterTextSplitter for splitting large text documents into manageable chunks for better processing.
5. Embeddings and Vector Stores
Integrates Google Generative AI Embeddings for embedding large texts and storing them in FAISS for effective document retrieval.
Supports vector searches for fast information retrieval, especially when working with large text documents such as PDFs or Word files.
6. Image Creation with OpenAI
Uses OpenAI's DALL-E model for generating images based on textual prompts.
7. LLM Integration
LangChain and LlamaIndex: Supports integration with LangChain and LlamaIndex for easy-to-use workflows for text and document processing.
Anthropic Claude Models: Provides access to Claude's advanced AI models for text processing, supporting complex queries and document comprehension tasks.
Requirements

Before you can use this repository, you need to install the following dependencies:

bash
Copy code
pip install pandasai pandas langchain PyPDF2 docx openai llama-index dotenv
Additionally, make sure to have the following API keys in your .env file:

GOOGLE_API_KEY: For Google Generative AI
OPENAI_API_KEY: For OpenAI integration
PANDASAI_API_KEY: For PandasAI
ANTHROPIC_API_KEY: For Anthropic API
Getting Started

1. Clone the Repository
Clone the project repository to your local machine:

bash
Copy code
git clone https://github.com/halefcobn/GeminiModels.git
cd GeminiModels
2. Set Up the Environment
Create a .env file and configure it with your API keys:

makefile
Copy code
GOOGLE_API_KEY=your-google-api-key
OPENAI_API_KEY=your-openai-api-key
PANDASAI_API_KEY=your-pandasai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
3. Run the AI Models
You can run the AI models through various classes provided. Below are examples of how to use them:

Text Model

python
Copy code
from GeminiModels import GeminiTextModels
gemini = GeminiTextModels()
response = gemini.get_answer("What is the weather today?")
print(response)
PDF Document Processing

python
Copy code
from GeminiModels import PDFDocument
pdf_processor = PDFDocument()
response = pdf_processor.get_response_pdf("Summarize the document", "sample.pdf")
print(response)
CSV File Processing

python
Copy code
from GeminiModels import CSVFile
csv_processor = CSVFile()
response = csv_processor.get_csv_response("data.csv", "What is the total revenue?")
print(response)
4. Test the Models
You can modify the examples and input different prompts or document files to see how each model performs in real scenarios. Make sure your .env file is properly set up, and your API keys are valid for full functionality.

Folder Structure

The repository is organized into the following folders:

models/: Contains model classes and methods for text, image, and document processing.
utilities/: Includes utility functions such as file handling and model-specific operations.
examples/: Sample scripts demonstrating how to use each model.
Contributing

If you'd like to contribute to this project:

Fork the repository
Create a new branch (git checkout -b feature-branch)
Make your changes
Push your changes (git push origin feature-branch)
Create a pull request
License

This project is licensed under the MIT License. See the LICENSE file for more details.

Contact

For any questions or issues, feel free to reach out or open an issue on GitHub. We appreciate your contributions and feedback!

