import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template
from Summary.OpenAISum import OAISumChain as Summary
from Sections.Introduction import Introduction
from Sections.Reference import Reference
from Sections.LiteratureReview import LiteratureReview

def main():
    load_dotenv()
    summary = Summary()
    introduction = Introduction()
    reference = Reference()
    literature = LiteratureReview()

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

        if st.button("Process"):
            with st.spinner("Processing"):
                # Get summaries of all the research papers
                st.session_state.summary = summary.get_summaries(introduction_docs)

    # Paper Generation
    if st.session_state.summary:
        st.write(bot_template.replace("{{MSG}}","LISTING SUMMARIES"),unsafe_allow_html=True)
        for summs in st.session_state.summary:
            st.write(bot_template.replace("{{MSG}}",summs),unsafe_allow_html=True)

        # Reference Section
        reference_section = reference.combine(st.session_state.summary)
        st.write(bot_template.replace("{{MSG}}","LISTING REFERENCES"),unsafe_allow_html=True)
        for references in reference_section:
            st.write(bot_template.replace("{{MSG}}",references),unsafe_allow_html=True)

        # # Introduction Section
        # introduction_section = introduction.latex(references=reference_section, summaries=st.session_state.summary, title="A survey on sentiment analysis methods, applications,and challenges")
        # st.write(bot_template.replace("{{MSG}}","LISTING INTRODUCTIONS"),unsafe_allow_html=True)        
        # st.write(bot_template.replace("{{MSG}}",introduction_section),unsafe_allow_html=True)

        # Literature Review Section
        initial_para,literature_section = literature.refine(references=reference_section, summaries=st.session_state.summary, title="A survey on sentiment analysis methods, applications,and challenges")
        st.write(bot_template.replace("{{MSG}}","LISTING LITERATURE REVIEW"),unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}",initial_para),unsafe_allow_html=True)
        for literatureReview in literature_section:
            st.write(bot_template.replace("{{MSG}}",literatureReview),unsafe_allow_html=True)

        
if __name__ == '__main__':
    main()