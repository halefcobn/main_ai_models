import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
import pandas as pd
from pandasai import Agent
from dotenv import load_dotenv
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
from google.generativeai import GenerativeModel, configure
import google.generativeai as genai
import time
from openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.llms.anthropic import Anthropic
from langchain_openai import ChatOpenAI
load_dotenv()
from llama_index.llms.gemini import Gemini
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import openai
from docx import Document
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from llama_index.core.llms import ChatMessage
from llama_index.llms.gemini import Gemini
import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class GeminiModels:
    def __init__(self):
        # Load environment variables
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key is not loaded. Check your .env file.")
        genai.configure(api_key=self.api_key)
        
        # Initialize configuration for the text model
        self.text_model = genai.GenerativeModel(model_name="gemini-1.5-pro", system_instruction=["Helpful assistant"],)

    
    @staticmethod
    def gemini_video_response(prompt,video_path, sys_video_command):
        try:
            video_file = genai.upload_file(path=video_path)            
            video_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro", system_instruction=sys_video_command)
            response = video_model.generate_content([prompt,video_file])
            
            video_text = response.text
            return video_text
        except Exception as e:
            return str(e)
    
    def gemini_image_response(self, image_path, sys_image_command):
        try:
            print(f"Uploading image: {image_path}")
            image_file = genai.upload_file(path=image_path)
            time.sleep(5)  # Adjust this sleep time as necessary for your use case
            print(f"Uploaded image file: {image_file}")
            
            image_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro", system_instruction=sys_image_command)
            response = image_model.generate_content([image_file])
            
            image_text = response.text
            return image_text
        except Exception as e:
            return str(e)
    
    @staticmethod
    def gemini_audio_response(audio_path, sys_video_command):
        try:
            audio_file = genai.upload_file(path=audio_path) # Adjust this sleep time as necessary for your use case        
            auido_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro", system_instruction=sys_video_command)
            response = auido_model.generate_content(["Convert auido to the texts properly", audio_file])
            
            audio_text = response.text
            return audio_text
        except Exception as e:
                return str(e)
    
class GeminiTextModels:

    def __init__(self):
        self.messages = []  # Initialize an empty list to store chat history


    def get_answer(self, message):

        # Add the user message to the chat history
        self.messages.append(ChatMessage(role="assistant", content="act as python developer expert only"))
        self.messages.append(ChatMessage(role="user", content=message))
        
        # Send the chat history to the model
        resp = Gemini(api_key=os.getenv("GOOGLE_API_KEY")).chat(self.messages)
        
        # Add the whatmodel's response to the chat history
        self.messages.append(ChatMessage(role="assistant", content=resp))
        
        # Print the model's response
        return resp.message.content.strip()

class OpenAIModels:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.messages = [("system","AI assistant",)] 
    def set_model_assistant(self, assistant):
        self.messages = [("system", assistant)]
        
    def response_openai_text(self, user_message: str) -> str:
        if not user_message:  
            return "No message given"
        self.messages.append(("human", user_message) )
        ai_msg = self.llm.invoke(self.messages)
        self.messages.append(("assistant", ai_msg.content))
        return ai_msg.content
    

class WORDDocument:
    
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        load_dotenv()
        self.messages = [("system", "Act as an helpfu AI assistant on word documents.")]
    # Function to read the content of a Word document
    def read_docx(self,file_path):
        document = Document(file_path)
        full_text = []
        for para in document.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)


    def get_docx_response(self, question, word_file_path):
        # Ask for the path to the Word document
        file_path = word_file_path

        try:
            # Read the document content
            document_text = self.read_docx(file_path)

            # Get the user's question
            user_question = question
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.3,
                max_tokens=1500,
                timeout=60,
                max_retries=2,
                api_key=openai.api_key
            )

            self.messages.append(("human", f"The following is the content of a word document:\n{document_text}\n\nQuestion: {user_question}\nAnswer:"))
            ai_msg = llm.invoke(self.messages)
            self.messages.append(("assistant", ai_msg.content))
            return ai_msg.content

        except Exception as e:
            print(f"An error occurred: {e}")


from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
import os

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

class PDFDocument:
    def __init__(self):
        load_dotenv()
    
    def get_pdf_text(self, pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(self,text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(self, text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")


    def get_conversational_chain(self):
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just act as a Arcadia, MBSE and Agile expert.

        Context: {context}
        Question: {question}

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.3)
        
        retriever = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True).as_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        return qa_chain
    def user_input(self, user_question):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = self.get_conversational_chain()
        response = chain.invoke({"input_documents": docs, "query": user_question}, return_only_outputs=True)
#        output_text = response['output_text'].strip()
        return response['result']

    def get_response_pdf(self, message, pdf_path):
        user_question = message
        pdf_docs = [pdf_path]  # Modify as needed
        raw_text = self.get_pdf_text(pdf_docs)
        text_chunks = self.get_text_chunks(raw_text)
        self.get_vector_store(text_chunks)
        if user_question:
            res = self.user_input(user_question)
            return res
        
class CSVFile:
    def __init__(self):
        load_dotenv()
    
    def get_csv_response(self, csv_file_path,question) -> str :
        agent = create_csv_agent(ChatOpenAI(temperature=0, model="gpt-4o",api_key=os.getenv("OPENAI_API_KEY")),
            csv_file_path,
            verbose=False,
            agent_type=AgentType.OPENAI_FUNCTIONS, allow_dangerous_code=True
            
            
        )

        response = agent.invoke(question)
        return response['output']
    
class PandasAI:

    def get_file_path_and_question(path, question):
        
        df = pd.read_excel(path)

        # By default, unless you choose a different LLM, it will use BambooLLM.
        # You can get your free API key signing up at https://pandabi.ai (you can also configure it in your .env file)
        os.environ["PANDASAI_API_KEY"] = os.getenv("PANDASAI_API_KEY")

        agent = Agent([df])
        return agent.chat(question)

class Anthropic:
    def get_answer(prompt, message):
        
        
        os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

        messages = [
            ChatMessage(
                role="system", content=prompt
            ),
            ChatMessage(role="user", content=message),
        ]
        resp = Anthropic(model="claude-3-5-sonnet-20240620").chat(messages)
        return resp
    
class CreateImageOpenAI:
    def __init__(self) -> None:
        self.api_key= os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

    def generate_image(self,message):
        response = self.client.images.generate(
            model="dall-e-2",
            prompt=message,
            size="512x512",
            quality="hd",
            n=1,
        )
        return response.data[0].url
    

from utilities import get_model_response
warnings.filterwarnings("ignore", category=DeprecationWarning)

class GeminiDataAnalysis:
    @staticmethod
    def get_answer(question, file_path):
        csv_loader = CSVLoader(file_path, encoding="utf-8", csv_args={'delimiter': ','})
        data = csv_loader.load()

        response = get_model_response(data, question)
        # Add the user message to the chat history
        return response
    
            

if __name__ == "__main__":
    data_response=GeminiDataAnalysis()
    
    analysis=data_response.get_answer(question= "Explain the data",file_path="pupulation.csv")
    print(analysis)
