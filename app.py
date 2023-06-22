import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate
from langchain.schema import (SystemMessage, HumanMessage)
from base import Base

def main():
    #load api keys
    load_dotenv()
    base = Base()

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
                st.session_state.summary = base.get_summary_text(pdf_docs)

    prompt = PromptTemplate(
        input_variables= ["sections", "numwords"],
        template = "Write the {sections} for this academic review paper. The {sections} should be {numwords} atleast words long"
    )
    sections_list = ["Title","Introduction","Literature Review", "Comparison", "Conclusion", "References", "Abstract"]
    numwords_list = ["4","2000","2000","2000","1000","200","400"]

    
    # sections_list = ["Title","Introduction","Literature Review"]
    # numwords_list = ["4","2000","2000"]
    
    # Initialise the chatbot with history
    chat, history = base.get_chat_history()

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