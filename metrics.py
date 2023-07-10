import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from base import Base as Base
from Summary.OpenAISum import OAISumChain as Base2
from Metric.rouge import Metric

def main():
    #load api keys
    load_dotenv()
    sum_gen_1 = Base()
    sum_gen_2 = Base2()
    metric = Metric()
    delimiter = " "

    # streamlit initialize page
    st.set_page_config(page_title="Review Generator", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Review Generator :books:")

    # uninitialized session variables 
    if "summary" not in st.session_state:
        st.session_state.sum_gen_1 = None
        st.session_state.sum_gen_2 = None

    # streamlit navbar
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFS here and click on process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get summaries of all the research papers
                st.session_state.sum_gen_1 = sum_gen_1.get_summaries(pdf_docs)
                st.session_state.sum_gen_2 = sum_gen_2.get_summaries(pdf_docs)

    # Metric Calculation
    if st.session_state.sum_gen_2:
        st.write(user_template.replace("{{MSG}}","Summary gen 1"),unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}",delimiter.join(st.session_state.sum_gen_1)),unsafe_allow_html=True)
        st.write(user_template.replace("{{MSG}}","Summary gen 2"),unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}",delimiter.join(st.session_state.sum_gen_2)),unsafe_allow_html=True)

        for i in range(len(st.session_state.sum_gen_1)):
            rouge = metric.rouge(st.session_state.sum_gen_1[i], st.session_state.sum_gen_2[i])
            print("Sr No: "+str(i)+" Rouge Score: "+rouge)

if __name__ == '__main__':
    main()