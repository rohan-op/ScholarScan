from base import Base
from langchain.prompts import PromptTemplate
from langchain.schema import (SystemMessage, HumanMessage)

class Introduction(Base):

    introduction_mini_instruction = PromptTemplate(
        input_variables= ["title", "summaries", "references"],
        template = """
        Based on the title of the academic review paper, your job is to write an introduction for this academic review paper. You are provided summary of research papers for reference. Do in-text citation if you are using ideas from the research paper. Reference section is also provided that is in APA format. The writing style and note should be academic in nature.

        TITLE: {title}

        FORMAT OF THE INTRODUCTION:
        1) Elaborate on the topic of dicussion.
        2) Mention previously done work and their view of topic in discussion.
        3) Mention the real-world applications of technology that is being discussed.
        
        SUMMARIES OF REFERENCE RESEARCH PAPERS: {summaries}

        REFERENCES SECTION: {references}
    """)

    introduction_combine_instruction = PromptTemplate(
        input_variables= ["title", "introduction1", "introduction2", "references"],
        template = """
        You will be provided with two unique introduction sections, your job is to combine and rewrite the introduction section for an academic review paper. Do in-text citation if you are using ideas from the research paper. Reference section is also provided that is in APA format. The writing style and note should be academic in nature.

        TITLE: {title}

        FORMAT OF THE INTRODUCTION:
        1) Elaborate on the topic of dicussion.
        2) Mention previously done work and their view of topic in discussion.
        3) Mention the real-world applications of technology that is being discussed.
        
        INTRODUCTION 1: {introduction1}

        INTRODUCTION 2: {introduction2}

        REFERENCES SECTION: {references}
    """)

    introduction_refine_instruction = PromptTemplate(
        input_variables= ["title", "introduction"],
        template = """
        You are provided with an introduction section of an academic review paper, your job is to enhance this introduction. Try to elaborate on the idea, motive and importance of topic in hand. The ideas in the introduction should be cohesive. The writing style and note should be strictly academic in nature. The provided introduction section will have in-text citation. 

        TITLE: {title}

        FORMAT OF THE INTRODUCTION:
        1) Elaborate on the topic of dicussion.
        2) Mention previously done work and their view of topic in discussion.
        3) Mention the real-world applications of technology that is being discussed.
        4) Set the stage by mentioning the structure of academic review paper i.e Introduction, Literature Review, Comparison of Methodologies, Challenges, Future work, and Conclusion in the same order.
        
        INTRODUCTION SECTION: {introduction}

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

    def mini(self, references, summaries, title):
        introduction_messages = [
            SystemMessage(content=self.introduction_mini_instruction.format(title=title, summaries=summaries, references=references)),
            HumanMessage(content="Write the Introduction section")
        ]
        response = self.model(introduction_messages)
        return response.content
    
    def multi_mini(self,references, summaries, title):
        minis = []
        if len(references) != len(summaries):
            raise ValueError("References and summaries must have the same number of items.")
        odd_number_of_items = len(references) % 2 != 0

        for i in range(0, len(summaries),2):
            print("ith Iteration Introduction:"+ str(i))
            temp_ref = references[i:i+2]
            temp_summ = summaries[i:i+2]
            minis.append(self.mini(references=temp_ref, summaries=temp_summ, title= title))

        if odd_number_of_items:
            temp_ref = [references[-1]]
            temp_summ = [summaries[-1]]
            minis.append(self.mini(references=temp_ref, summaries=temp_summ, title= title))

        return minis
    
    def combine(self, references, summaries, title):
        minis = self.multi_mini(references=references, summaries=summaries, title=title)
        
        while(len(minis) > 1):
            print("Introduction Combining length: "+str(len(minis)))
            introduction1 = minis.pop(0)
            introduction2 = minis.pop(0)
            introduction_messages = [
                SystemMessage(content=self.introduction_combine_instruction.format(title=title, introduction1=introduction1, introduction2=introduction2, references=references)),
                HumanMessage(content="Write the Introduction section")
            ]
            response = self.model(introduction_messages)
            print("\nCombined Output:"+response.content+"\n")
            minis.insert(0, response.content)
        return response.content   

    def refine(self, references, summaries, title):
        introduction = self.combine(references=references, summaries=summaries, title=title)
        introduction_messages = [
                SystemMessage(content=self.introduction_refine_instruction.format(title=title, introduction=introduction)),
                HumanMessage(content="Write the Introduction section")
            ]
        print("\nEnhancing the Introduction\n")
        response = self.model(introduction_messages)
        print("ENHANCED INTRODUCTION:"+ response.content+"\n")
        return response.content
    
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