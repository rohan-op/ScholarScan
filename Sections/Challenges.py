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
        input_variables= ["challenges"],
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
        challenges_messages = [
            SystemMessage(content=self.challenges_mini_instruction.format(literature=literature, references=references)),
            HumanMessage(content="Write the Challenges section")
        ]
        response = self.model(challenges_messages)
        return response.content
    
    def multi_mini(self,references, literatures):
        minis = []
        print("\nChallenges multimini\n")
        for literature in literatures:
            minis.append(self.mini(references=references,literature=literature))

        return minis
    
    def combine(self, references, literatures):
        minis = self.multi_mini(references=references, literatures=literatures)
        
        while(len(minis) > 1):
            print("Challenges Combining length: "+str(len(minis)))
            challenges1 = minis.pop(0)
            challenges2 = minis.pop(0)
            challenges_messages = [
                SystemMessage(content=self.challenges_combine_instruction.format( challenges1=challenges1, challenges2=challenges2, references=references)),
                HumanMessage(content="Write the Challenges section")
            ]
            response = self.model(challenges_messages)
            #print("\nChallenges Combined Output:"+response.content+"\n")
            minis.insert(0, response.content)
        return response.content

    def refine(self, references, literatures):
        challenges = self.combine(references=references, literatures=literatures)
        challenges_messages = [
                SystemMessage(content=self.challenges_refine_instruction.format(challenges=challenges)),
                HumanMessage(content="Write the Challenges section")
            ]
        print("\nEnhancing the Challenges section\n")
        response = self.model(challenges_messages)
        return response.content