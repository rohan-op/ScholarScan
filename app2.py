import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template
from Summary.OpenAISum import OAISumChain as Summary
from Sections.Introduction import Introduction
from Sections.Reference import Reference
from Sections.LiteratureReview import LiteratureReview
from Sections.Challenges import Challenges
from Sections.Proposal import Proposal
from Sections.FutureWork import FutureWork
from Sections.Conclusion import Conclusion
from Sections. Abstract import Abstract

def main():
    load_dotenv()
    summary = Summary()
    introduction = Introduction()
    reference = Reference()
    literature = LiteratureReview()
    challenge = Challenges()
    proposal = Proposal()
    future = FutureWork()
    conclusion = Conclusion()
    abstract = Abstract()

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
        introduction_section = introduction.refine(references=reference_section, summaries=st.session_state.summary, title="A survey on sentiment analysis methods, applications,and challenges")
        st.write(bot_template.replace("{{MSG}}","LISTING INTRODUCTIONS"),unsafe_allow_html=True)        
        st.write(bot_template.replace("{{MSG}}",introduction_section),unsafe_allow_html=True)

        # Literature Review Section
        initial_para,literature_section = literature.refine(references=reference_section, summaries=st.session_state.summary, title="A survey on sentiment analysis methods, applications,and challenges")
        st.write(bot_template.replace("{{MSG}}","LISTING LITERATURE REVIEW"),unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}",initial_para),unsafe_allow_html=True)
        for literatureReview in literature_section:
            st.write(bot_template.replace("{{MSG}}",literatureReview),unsafe_allow_html=True)

        # Challenges Section
        challenges_section = challenge.refine(references=reference_section, literatures=literature_section)
        st.write(bot_template.replace("{{MSG}}","LISTING CHALLENGES SECTION"),unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}",challenges_section),unsafe_allow_html=True)

        # Propose Methodology Section
        methodology_section = proposal.mini(challenges=challenges_section, literature=literature_section)
        st.write(bot_template.replace("{{MSG}}","LISTING METHODOLOGY SECTION"),unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}",methodology_section),unsafe_allow_html=True)

        # Future Work Section
        future_section = future.mini(challenges=challenges_section, proposal=methodology_section)
        st.write(bot_template.replace("{{MSG}}","LISTING FUTURE WORK SECTION"),unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}",future_section),unsafe_allow_html=True)

        # Conclusion Section
        conclusion_section = conclusion.mini(literature=literature_section, future=future_section, challenges=challenges_section, proposal=methodology_section)
        st.write(bot_template.replace("{{MSG}}","LISTING CONCLUSION SECTION"),unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}",conclusion_section),unsafe_allow_html=True)

        # Abstract Section
        abstract_section = abstract.mini(introduction=introduction_section,future=future_section,conclusion=conclusion_section,challenges=challenges_section, proposal=methodology_section)
        st.write(bot_template.replace("{{MSG}}","LISTING ABSTRACT SECTION"),unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}",abstract_section),unsafe_allow_html=True)
        
if __name__ == '__main__':
    main()