from base import Base
from langchain.prompts import PromptTemplate
from langchain.schema import (SystemMessage, HumanMessage)

class Conclusion(Base):

    conclusion_mini_instruction = PromptTemplate(
        input_variables= ["proposal", "challenges","future", "literature"],
        template = """
        You are provided the future work, literature review, challenges and methodology section of an academic review paper. Your job is write conclusion section for this academic review paper based on the information provided. Do in-text citation if you are using ideas from the research paper. Reference section is also provided that is in APA format. The writing style and note should be academic in nature. There is no need for conclusion paragraph.
        
        PROPOSED METHODOLOGY: {proposal}

        CHALLENGES SECTION: {challenges}

        FUTURE WORK SECTION: {future}

        LITERATURE REVIEW: {literature}
    """)

    def __init__(self):
        self.model = self.get_ChatOpenAImodel()

    def mini(self, challenges, proposal, future, literature):
        conclusion_messages = [
            SystemMessage(content=self.conclusion_mini_instruction.format(proposal=proposal, challenges=challenges, future=future, literature=literature)),
            HumanMessage(content="Write the Conclusion section")
        ]
        response = self.model(conclusion_messages)
        return response.content