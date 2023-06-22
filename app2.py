import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os

from htmlTemplates import css, bot_template, user_template
from langchain import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.docstore.document import Document


from langchain.prompts import PromptTemplate
from langchain.schema import (SystemMessage, HumanMessage)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory


def get_summary_text(pdf_docs):
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


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator = '\n',
        chunk_size = 700,
        chunk_overlap = 50,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_summary(pdf_docs):
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


def get_chat_history():
    chat = ChatOpenAI(temperature=1)
    history = ChatMessageHistory()
    return chat, history


def main():
    #load api keys
    load_dotenv()

    # streamlit initialize page
    st.set_page_config(page_title="Review Generator", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Review Generator :books:")

    # uninitialized session variables 
    if "summary" not in st.session_state:
        st.session_state.summary = None

    # streamlit navbar
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFS here and click on process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get summaries of all the research papers
                st.session_state.summary = get_summary(pdf_docs)

    prompt = PromptTemplate(
        input_variables= ["sections", "numwords"],
        template = "Write the {sections} for this academic review paper. The {sections} should be {numwords} atleast words long"
    )
    sections_list = ["Title","Introduction","Literature Review", "Comparison", "Conclusion", "References", "Abstract"]
    numwords_list = ["4","2000","2000","2000","1000","200","400"]

    
    # sections_list = ["Title","Introduction","Literature Review"]
    # numwords_list = ["4","2000","2000"]
    
    # Initialise the chatbot with history
    chat, history = get_chat_history()

    # Setup Intructions for the chatbot
    messages = [
        SystemMessage(content='''
        You are given summaries of multiple research papers below. Your job is to write different sections of an academic review paper. For example the Introduction, Abstract, Literature Review, Comparision, Conclusion, Acknowledgements, References etc. The user will provide the section that is to be wrriten and how long it should be. All the sections should be unique and should not be same as other sections of the paper. If you are using direct/same sentences from the summary of any research paper, mention the reference in the text. All the sections should be written in an academic manner. 

        NOTE: When writing an Introduction, do not mention what the summaries of the research paper or their content. Rather focus on the context and topic of the academic paper and what we plan to achieve from this review.

        SUMMARIES OF ALL THE RESEARCH PAPERS: 
        '''+str(st.session_state.summary))
    ]
    st.write(bot_template.replace("{{MSG}}",str(st.session_state.summary)),unsafe_allow_html=True)

    # Generate Sections of Academic Review Paper:
    for section, numword in zip(sections_list, numwords_list):
        messages.append(HumanMessage(content=prompt.format(sections=section,numwords=numword)))
        response = chat(messages)
        st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)

if __name__ == '__main__':
    main()