import os
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

PROMPT_TEMPLATE = """
You have to answer the question from the provided context properly.
Context: {context}
Question: {question}

Answer is:
"""

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv("GEMINI_API_KEY"), model='models/embedding-001')
model = ChatGoogleGenerativeAI(google_api_key=os.getenv("GEMINI_API_KEY"), model="models/gemini-1.5-pro", temperature=.9)

def get_model_response(file, query):
    # split context into chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    context = '\n\n'.join(str(p.page_content) for p in file)
    data = text_splitter.split_text(context)

    # generate embedding
    searcher = Chroma.from_texts(data, embeddings).as_retriever()
    records = searcher.get_relevant_documents(query)
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=['context', 'question'])
    
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    response = chain.invoke(
        {
            'input_documents': records,
            'question': query
        },
        return_only_outputs=True
    )

    return response['output_text']

