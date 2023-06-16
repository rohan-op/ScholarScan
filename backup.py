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
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "summary" not in st.session_state:
        st.session_state.summary = None

    # streamlit navbar
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFS here and click on process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get summaries of all the research papers
                st.session_state.summary = get_summary_text(pdf_docs)

    prompt = PromptTemplate(
        input_variables= ["section", "numwords"],
        template = "Write the {section} for this academic review paper. The {section} should be {numwords} atleast words long"
    )

    sections = ["Title","Introduction","Literature Review", "Comparison", "Conclusion", "References"]
    #chat = ChatOpenAI(temperature=1)        
    chat, history = get_chat_history()
    history.add_system_message

    messages = [
        SystemMessage(content='''
        You are given summaries of multiple research papers below. Your job is to write different sections of an academic review paper. For example the Introduction, Abstract, Literature Review, Comparision, Conclusion, Acknowledgements, References etc. The user will provide the section that is to be wrriten and how long it should be. All the sections should be unique and should not be same as other sections of the paper. If you are using direct/same sentences from the summary of any research paper, mention the reference in the text. All the sections should be written in an academic manner.

        SUMMARIES OF ALL THE RESEARCH PAPERS: 
        '''+st.session_state.summary)
    ]

    # Generate Sections of Academic Review Paper:

    messages.append(HumanMessage(content=prompt.format(section="Title",numwords="4")))
    response = chat(messages)
    st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)

    messages.append(HumanMessage(content=prompt.format(section="Abstract",numwords="1000")))
    response = chat(messages)
    st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)

    messages.append(HumanMessage(content=prompt.format(section="Introduction",numwords="2000")))
    response = chat(messages)
    st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)

    messages.append(HumanMessage(content=prompt.format(section="Comparison",numwords="2000")))
    response = chat(messages)
    st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)

    messages.append(HumanMessage(content=prompt.format(section="Conclusion",numwords="1000")))
    response = chat(messages)
    st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)

    messages.append(HumanMessage(content=prompt.format(section="References",numwords="100")))
    response = chat(messages)
    st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)

if __name__ == '__main__':
    main()