from base import Base

from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os

class OpenAISummary(Base):

    def get_summary_text(self, pdf_docs):
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
