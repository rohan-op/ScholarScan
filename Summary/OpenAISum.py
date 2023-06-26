from base import Base
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI

class OAISumChain(Base):

    def get_summary(self, document, llm, text_splitter,chain_type='stuff'):
        prompt_template = """Write a concise 600 word long summary of the following research paper, also mention the title and author of the research paper: {text}"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        texts = text_splitter.split_documents(document)
        chain = load_summarize_chain(llm, chain_type=chain_type, prompt=PROMPT)
        summary = chain.run(texts)
        return summary


    def get_summaries(self, pdf_docs):
        #llm = OpenAI(temperature=1)
        llm = ChatOpenAI(temperature=1, model='gpt-3.5-turbo-16k')
        summaries = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for i,pdf in enumerate(pdf_docs):
            text = ''
            pdf_reader = PdfReader(pdf)
            for pages in pdf_reader.pages:
                text += pages.extract_text()
            document = [Document(page_content=text)]
            summaries.append("\n\nRESEARCH PAPER "+ str(i) +":\n\n"+self.get_summary(document,llm,text_splitter))
        return summaries
