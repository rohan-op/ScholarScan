from base import Base
from langchain.prompts import PromptTemplate
from langchain.schema import (SystemMessage, HumanMessage)

class Challenges(Base):

    challenges_mini_instruction = PromptTemplate(
        input_variables= ["literature", "references"],
        template = """
        Your job is to write challenges section for this academic review paper. You are provided the literature review section. Do in-text citation if you are using ideas from the research paper. Reference section is also provided that is in APA format. The writing style and note should be academic in nature. There is no need for conclusion paragraph.

        FORMAT OF THE CHALLENGES SECTION:
        1) Mention and elaborate the challenge/problems
        2) Mention and elaborate the methodologies/techonology/techniques where this problem occurs.

        NOTE: Strictly follow the above format and do not start paragraphs with sentences like "In conclusion,", "Overall," or similar.
        
        LITERATURE REVIEW: {literature}

        REFERENCES SECTION: {references}
    """)

    challenges_combine_instruction = PromptTemplate(
        input_variables= ["challenges1", "challenges2", "references"],
        template = """
        You will be provided with two unique challenges sections, your job is to combine and rewrite the challenges section for an academic review paper. Make sure that the challenges are not mentioned twice. The final challenges section should consist of unique challenges only. Do in-text citation if you are using ideas from the research paper. Reference section is also provided that is in APA format. The writing style and note should be academic in nature.

        FORMAT OF THE CHALLENGES SECTION:
        1) Mention and elaborate the challenge/problems
        2) Mention and elaborate the methodologies/techonology/techniques where this problem occurs.
        
        CHALLENGES 1: {challenges1}

        CHALLENGES 2: {challenges2}

        REFERENCES SECTION: {references}
    """)

    challenges_refine_instruction = PromptTemplate(
        input_variables= ["title", "challenges"],
        template = """
        You are provided with a challenges section of an academic review paper, your job is to enhance this challenges section. All the challenges mentioned in this section are suppose to be unique. Challenges mentioned in The writing style and note should be strictly academic in nature. The provided introduction section will have in-text citation. 

        FORMAT OF THE CHALLENGES SECTION:
        1) Mention and elaborate the challenge/problems
        2) Mention and elaborate the methodologies/techonology/techniques where this problem occurs.

        NOTE: Strictly follow the above format and do not start paragraphs with sentences like "In conclusion,", "Overall," or similar.
        
        CHALLENGES SECTION: {challenges}
    """)

    def __init__(self):
        self.model = self.get_ChatOpenAImodel()

    def mini(self, references, literature):
        introduction_messages = [
            SystemMessage(content=self.challenges_mini_instruction.format(literature=literature, references=references)),
            HumanMessage(content="Write the Challenges section")
        ]
        response = self.model(introduction_messages)
        return response.content
    
    def multi_mini(self,references, literatures):
        minis = []
        for literature in literatures:
            minis.append(self.mini(references=references,literature=literature))

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