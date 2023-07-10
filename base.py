from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.prompts import PromptTemplate

class Base:
    instruction = '''
        You are given summaries of multiple research papers below. Your job is to write different sections of an academic review paper. For example the Introduction, Abstract, Literature Review, Comparision, Conclusion, Acknowledgements, References etc. The user will provide the section that is to be wrriten and how long it should be. All the sections should be unique and should not be same as other sections of the paper. If you are using direct/same sentences from the summary of any research paper, mention the reference in the text. All the sections should be written in an academic manner. 

        NOTE: When writing an Introduction, do not mention what the summaries of the research paper or their content. Rather focus on the context and topic of the academic paper and what we plan to achieve from this review.

        SUMMARIES OF ALL THE RESEARCH PAPERS: 
        '''
    
    reference_apa_instruction = """
        Based on the information provided below, your job is to create a reference section for an academic review paper.
        Make sure that you are only referencing the papers that are mention below. The reference section should be in APA format.

        CONTENT:
    """

    introduction_instruction = PromptTemplate(
        input_variables= ["title", "summary", "references"],
        template = """
        Based on the title of the academic review paper, your job is to write an introduction for this academic review paper. You are provided summary of research papers for reference. Do in-text citation if you are using ideas from the research paper. Reference section is also provided that is in APA format. The writing style and note should be academic in nature. Use bullet points when required.

        TITLE: {title}

        FORMAT OF THE INTRODUCTION:
        1) Elaborate on the topic of dicussion.
        2) Mention previously done work and their view of topic in discussion.
        3) Mention the real-world applications of technology that is being discussed.
        4) Set the stage by mentioning the structure of academic review paper i.e Introduction, Literature Review, Comparison of Methodologies, Challenges, Future work, and Conclusion in the same order.
        5) No need to separately conclude the introduction. End the introduction after mentioning the structure of the review paper.

        SUMMARIES OF REFERENCE RESEARCH PAPERS: {summary}

        REFERENCES SECTION: {references}
    """
    )

    background_instruction = """"""

    improvement_instruction = """"""

    performance_evaluation = """"""

    applications_instruction = """"""

    challenges_instruction = """"""

    conclusion_instruction = """"""



    def __init__(self):
        load_dotenv()


    def get_HFmodel(self, repo_id="facebook/bart-large-cnn", model_kwargs={"temperature": 0.9, "max_length":1000, "min_length":300}):
        return HuggingFaceHub(repo_id=repo_id, model_kwargs=model_kwargs)


    def get_ChatOpenAImodel(self,temperature = 1, model='gpt-3.5-turbo-16k'):
        return ChatOpenAI(temperature=temperature, model=model)


    def get_summaries(self,pdf_docs):
        generate_summary_llm = self.get_HFmodel()

        summary = []
        for i,pdf in enumerate(pdf_docs):
            text = ''
            pdf_reader = PdfReader(pdf)
            for pages in pdf_reader.pages:
                text += pages.extract_text()
            summary.append("\nResearch Paper " + str(i) + ':\n\n' + generate_summary_llm(text) + '\n\n')
        return summary


    def get_chat_history(self):
        chat = self.get_ChatOpenAImodel()
        history = ChatMessageHistory()
        return chat, history