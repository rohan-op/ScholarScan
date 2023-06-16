import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain import HuggingFaceHub

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


def get_conversation_chain():
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, memory=memory)
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question+st.session_state.summary})
    st.session_state.chat_history = response['chat_history']

    for i,message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Review Generator", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    summary = ''
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "summary" not in st.session_state:
        st.session_state.summary = None

    st.header("Review Generator :books:")
    user_question = st.text_input("Generate the Introduction, Abstract, Literature Review etc")
    # if user_question:
        # handle_userinput(user_question)


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFS here and click on process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                st.session_state.summary = get_summary_text(pdf_docs)
                

    # if summary != '':
    #     st.write(summary)
    chat = ChatOpenAI(temperature=1)
                
    messages = [
        SystemMessage(content="You are nice assistant")
    ]

    #   if user_question:
    st.write(user_template.replace("{{MSG}}","Abstract"),unsafe_allow_html=True)
    messages.append(HumanMessage(content="Can you write a 600 to 700 word Abstract for an academic review paper, considering the information mentioned below about several research papers?\n\n"+st.session_state.summary))
    response = chat(messages)
    st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)

    st.write(user_template.replace("{{MSG}}","Introduction"),unsafe_allow_html=True)
    messages.append(HumanMessage(content="Can you write a 600 to 700 word Introduction for an academic review paper, considering the information mentioned below about several research papers?\n\n"+st.session_state.summary))
    response = chat(messages)
    st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)

    st.write(user_template.replace("{{MSG}}","Literature Review"),unsafe_allow_html=True)
    messages.append(HumanMessage(content="Can you write a 600 to 700 word Literature Review for an academic review paper, considering the information mentioned below about several research papers?\n\n"+st.session_state.summary))
    response = chat(messages)
    st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)

    st.write(user_template.replace("{{MSG}}","Results"),unsafe_allow_html=True)
    messages.append(HumanMessage(content="Can you write a 600 to 700 word Results for an academic review paper, considering the information mentioned below about several research papers?\n\n"+st.session_state.summary))
    response = chat(messages)
    st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)

    st.write(user_template.replace("{{MSG}}","Discussion"),unsafe_allow_html=True)
    messages.append(HumanMessage(content="Can you write a 600 to 700 word Discussion for an academic review paper, considering the information mentioned below about several research papers?\n\n"+st.session_state.summary))
    response = chat(messages)
    st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)

    st.write(user_template.replace("{{MSG}}","Conclusion"),unsafe_allow_html=True)
    messages.append(HumanMessage(content="Can you write a 600 to 700 word Conclusion for an academic review paper, considering the information mentioned below about several research papers?\n\n"+st.session_state.summary))
    response = chat(messages)
    st.write(bot_template.replace("{{MSG}}",response.content),unsafe_allow_html=True)

if __name__ == '__main__':
    main()