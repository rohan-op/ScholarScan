import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
#from base import Base
from Summary.OpenAISum import OAISumChain as Base

def main():
    #load api keys
    load_dotenv()
    base = Base()
    delimiter = " "

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
        introduction_docs = st.file_uploader("Upload your PDFS here and click on process", accept_multiple_files=True)

        # st.subheader("Your documents")
        # litreview_docs = st.file_uploader("Upload your PDFS here and click on process", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                # Get summaries of all the research papers
                st.session_state.summary = base.get_summaries(introduction_docs)

    # Paper Generation
    if st.session_state.summary:
        # Reference Section
        reference_chat = base.get_ChatOpenAImodel()
        reference_messages = [
            SystemMessage(content=base.reference_apa_instruction+delimiter.join(st.session_state.summary)),
            HumanMessage(content="Write the reference section")
        ]
        response = reference_chat(reference_messages)
        st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)
        references = response.content

        # Introduction Section
        introduction_chat = base.get_ChatOpenAImodel()
        intro_messages = [
            SystemMessage(content=base.introduction_instruction.format(title="A survey on sentiment analysis methods, applications,and challenges", summary=st.session_state.summary, references=references)),
            HumanMessage(content="Write the Introduction section")
        ]
        response = introduction_chat(intro_messages)
        st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)
        introduction = response.content  

if __name__ == '__main__':
    main()