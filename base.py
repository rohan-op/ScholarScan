from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
from langchain import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory

class Base:
    def __init__(self):
        load_dotenv()

    def get_summary_text(self,pdf_docs):
        generate_summary_llm = HuggingFaceHub(repo_id="facebook/bart-large-cnn", model_kwargs={"temperature": 0.9, "max_length":1000, "min_length":300})

        summary = ''
        for i,pdf in enumerate(pdf_docs):
            text = ''
            pdf_reader = PdfReader(pdf)
            for pages in pdf_reader.pages:
                text += pages.extract_text()
            summary += "\nResearch Paper " + str(i) + ':\n\n'
            summary += generate_summary_llm(text) + '\n\n'
        return summary


    def get_text_chunks(self,raw_text):
        text_splitter = CharacterTextSplitter(
            separator = '\n',
            chunk_size = 700,
            chunk_overlap = 50,
            length_function = len
        )
        chunks = text_splitter.split_text(raw_text)
        return chunks


    def get_summary(self, pdf_docs):
        llm = OpenAI(temperature=1)
        summary = ''
        prompt_template = """Write a 400 word long summary of the following research paper, also mention the title and author of the research paper: {text}"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        for i,doc in enumerate(pdf_docs):
            # Get the file name and extension
            file_name = doc.name
            file_extension = os.path.splitext(file_name)[1].lower()

            # Process only text files (e.g., .txt)
            if file_extension == ".txt":
                # Load the text file using TextLoader
                loader = TextLoader(file_name)
                documents = loader.load()

                # Split the documents using RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                #chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
                summary += "\n\nRESEARCH PAPER "+ str(i) +":\n\n"
                summary += chain.run(texts)
        return summary


    def get_chat_history(self):
        chat = ChatOpenAI(temperature=1)
        history = ChatMessageHistory()
        return chat, history