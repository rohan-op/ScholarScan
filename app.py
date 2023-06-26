import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
#from base import Base
from Metric.word2vec import Word2Vec
from Summary.OpenAISum import OAISumChain as Base

def main():
    #load api keys
    load_dotenv()
    base = Base()
    word2vec = Word2Vec()

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
                st.session_state.summary = base.get_summaries(pdf_docs)

    prompt = PromptTemplate(
        input_variables= ["sections", "numwords"],
        template = "Write the {sections} for this academic review paper. The {sections} should be {numwords} atleast words long"
    )

    sections_list = ["Title","Introduction","Literature Review", "Comparison", "Conclusion", "References", "Abstract"]
    numwords_list = ["4","2000","2000","2000","1000","200","400"]
    # sections_list = ["Title","Introduction","Literature Review"]
    # numwords_list = ["4","2000","2000"]
    
    # Initialize the chatbot with history 
    chat = base.get_ChatOpenAImodel()
    delimiter = " "
    # Setup Intructions for the chatbot
    if st.session_state.summary:
        messages = [
            SystemMessage(content=base.instruction+delimiter.join(st.session_state.summary))
        ]
        st.write(bot_template.replace("{{MSG}}",delimiter.join(st.session_state.summary)),unsafe_allow_html=True)

        scores = word2vec.get_similarity_scores(pdf_docs,st.session_state.summary)
        for score in scores:
            st.write(bot_template.replace("{{MSG}}",str(score)),unsafe_allow_html=True)

        # Generate Sections of Academic Review Paper:
        for section, numword in zip(sections_list, numwords_list):
            messages.append(HumanMessage(content=prompt.format(sections=section,numwords=numword)))
            response = chat(messages)
            messages.append(AIMessage(content=response.content))
            st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)

if __name__ == '__main__':
    main()