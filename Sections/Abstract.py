from base import Base
from langchain.prompts import PromptTemplate
from langchain.schema import (SystemMessage, HumanMessage)

class Abstract(Base):

    abstract_mini_instruction = PromptTemplate(
        input_variables= ["introduction","proposal", "challenges","future", "conclusion"],
        template = """
        You are provided the introduction, methodology, challenges, future work and conclusion section of an academic review paper. Your job is write abstract section for this academic review paper based on the information provided. Do in-text citation if you are using ideas from the research paper. Reference section is also provided that is in APA format. The writing style and note should be academic in nature. There is no need for conclusion paragraph.
        
        INTRODUCTION SECTION: {introduction}

        PROPOSED METHODOLOGY: {proposal}

        CHALLENGES SECTION: {challenges}

        FUTURE WORK SECTION: {future}

        CONCLUSION SECTION: {conclusion}

    """)

    def __init__(self):
        self.model = self.get_ChatOpenAImodel()

    def mini(self, introduction, challenges, proposal, future, conclusion):
        abstract_messages = [
            SystemMessage(content=self.abstract_mini_instruction.format(introduction=introduction,proposal=proposal, challenges=challenges, future=future, conclusion=conclusion)),
            HumanMessage(content="Write the Abstract section")
        ]
        response = self.model(abstract_messages)
        return response.content