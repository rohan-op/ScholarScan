import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from htmlTemplates import css, bot_template, user_template
from langchain import HuggingFaceHub

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
                #st.session_state.summary = get_summary_text(pdf_docs)
                st.session_state.summary = '''
                
                '''

    prompt = PromptTemplate(
        input_variables= ["sections", "numwords"],
        template = "Write the {sections} for this academic review paper. The {sections} should be {numwords} atleast words long"
    )
    # sections_list = ["Title","Introduction","Literature Review", "Comparison", "Conclusion", "References"]
    # numwords_list = ["4","2000","2000","2000","1000","200"]

    
    sections_list = ["Title","Introduction","Literature Review"]
    numwords_list = ["4","2000","2000"]
    
    # Initialise the chatbot with history
    chat, history = get_chat_history()

    # Setup Intructions for the chatbot
    history.add_system_message("""
    You are given summaries of multiple research papers below. Your job is to write different sections of an academic review paper. For example the Introduction, Abstract, Literature Review, Comparision, Conclusion, Acknowledgements, References etc. The user will provide the section that is to be wrriten and how long it should be. All the sections should be unique and should not be same as other sections of the paper. If you are using direct/same sentences from the summary of any research paper, mention the reference in the text. All the sections should be written in an academic manner.

    SUMMARIES OF ALL THE RESEARCH PAPERS: 
    """+st.session_state.summary)


    # Generate Sections of Academic Review Paper:
    for section, numword in sections_list, numwords_list:
        history.add_user_message(prompt.format(sections=section,numwords=numword))
        response = chat(history.messages)
        st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)
        history.add_ai_message(response.content)

if __name__ == '__main__':
    main()