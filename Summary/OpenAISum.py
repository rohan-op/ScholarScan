from base import Base
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI

class OAISumChain(Base):

    def __init__(self):
        self.llm = ChatOpenAI(temperature=1, model='gpt-3.5-turbo-16k')

    def get_summary(self, document, text_splitter,chain_type='stuff'):
        
        prompt_template = """Write a concise 1000 word long summary of the research paper specified below, Strictly follow the format provided below. 

        SUMMARY FORMAT = 
        TITLE: [Title of the research paper]
        [NEXT LINE]
        AUTHORS: [Authors of the research paper]
        [NEXT LINE]
        YEAR: [year of publication]
        [NEXT LINE]
        JOURNAL NAME: [name of the journal]
        [NEXT LINE]
        VOLUME AND PAGE NUMBER: [volume number and page]
        [NEXT LINE]
        SUMMARY: [summary of the research paper]
        [NEXT LINE]
        
        
        CONTENT TO SUMMARIZE: 
        {text}"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        text = text_splitter.split_documents(document)
        chain = load_summarize_chain(self.llm, chain_type=chain_type, prompt=PROMPT)
        summary = chain.run(text)
        return summary


    def get_summaries(self, pdf_docs):
        #llm = OpenAI(temperature=1)
        
        summaries = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)

        for i,pdf in enumerate(pdf_docs):
            text = ''
            print("ith Iteration Summary:"+ str(i))
            pdf_reader = PdfReader(pdf)
            for pages in pdf_reader.pages:
                text += pages.extract_text()
            document = [Document(page_content=text)]
            # summaries.append("\n\nRESEARCH PAPER "+ str(i) +":\n\n"+self.get_summary(document,llm,text_splitter))
            summaries.append(self.get_summary(document,text_splitter)+"\n\n")
        return summaries
