from base import Base
from langchain.prompts import PromptTemplate
from langchain.schema import (SystemMessage, HumanMessage)

class LiteratureReview(Base):

    literature_mini_instruction = PromptTemplate(
        input_variables= ["summaries", "references"],
        template = """
        Your job is to write literature review for this academic review paper. You are provided summary of reference research papers for literature review. Do in-text citation if you are using ideas from the research paper. Reference section is also provided that is in APA format. The writing style and note should be academic in nature. There is no need for conclusion paragraph.

        FORMAT OF THE LITERATURE REVIEW:
        1) Mention the aim/goal of the reference research paper
        2) Mention the methodologies used to meet the goals.
        3) Mention previously done work and their view of topic in discussion.
        4) Mention the challenges faced.

        NOTE: Do not start paragraphs with sentences like "In conclusion,", "Overall," or similar.
        
        SUMMARIES OF REFERENCE RESEARCH PAPERS: {summaries}

        REFERENCES SECTION: {references}
    """)

    literature_refine_instruction = PromptTemplate(
        input_variables= ["title", "literature"],
        template = """
        You are provided with an literature review section of an academic review paper, your job is to the first paragraph for literature review based on the information provided. Try to elaborate on the idea, motive and importance of topic in hand. The writing style and note should be strictly academic in nature. 

        TITLE: {title}
        
        INTRODUCTION SECTION: {literature}
    """)

    introduction_latex_instruction = PromptTemplate(
        input_variables= ["title", "introduction"],
        template = """
        You are provided with an introduction section of an academic review paper, your job is to convert this introduction text into Latex. The provided introduction section will have in-text citation. Use IEEE latex template. The output LATEX of your job should not have anything after the introduction section of the paper.

        TITLE: {title}
        
        INTRODUCTION SECTION: {introduction}
    """)

    def __init__(self):
        self.model = self.get_ChatOpenAImodel()

    def mini(self, references, summaries):
        introduction_messages = [
            SystemMessage(content=self.literature_mini_instruction.format(summaries=summaries, references=references)),
            HumanMessage(content="Write the Literature Review section")
        ]
        response = self.model(introduction_messages)
        return response.content
    
    def multi_mini(self,references, summaries):
        minis = []
        if len(references) != len(summaries):
            raise ValueError("References and summaries must have the same number of items.")
        odd_number_of_items = len(references) % 2 != 0

        for i in range(0, len(summaries),2):
            print("ith Iteration Literature:"+ str(i))
            temp_ref = references[i:i+2]
            temp_summ = summaries[i:i+2]
            minis.append(self.mini(references=temp_ref, summaries=temp_summ))

        if odd_number_of_items:
            temp_ref = [references[-1]]
            temp_summ = [summaries[-1]]
            minis.append(self.mini(references=temp_ref, summaries=temp_summ))

        return minis

    def refine(self, references, summaries, title):
        literature = self.multi_mini(references=references, summaries=summaries)
        introduction_messages = [
                SystemMessage(content=self.literature_refine_instruction.format(title=title, literature=literature)),
                HumanMessage(content="Write the initial paragraph")
            ]
        print("\nEnhancing the Literature review\n")
        response = self.model(introduction_messages)
        return response.content, literature
    
    def latex(self, references, summaries, title):
        introduction = self.refine(references=references, summaries=summaries, title=title)
        introduction_messages = [
                SystemMessage(content=self.introduction_latex_instruction.format(title=title, introduction=introduction)),
                HumanMessage(content="Write the Introduction section in Latex")
            ]
        print("Writing Introduction in Latex")
        response = self.model(introduction_messages)
        return response.content
    
    def create(self, references, summaries, title):
        return 0