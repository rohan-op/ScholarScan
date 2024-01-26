from base import Base
from langchain.prompts import PromptTemplate
from langchain.schema import (SystemMessage, HumanMessage)

class Proposal(Base):

    proposal_mini_instruction = PromptTemplate(
        input_variables= ["literature", "challenges"],
        template = """
        You are provided the literature review and challenges section. Your job is to propose a methodology section for this academic review paper. Do in-text citation if you are using ideas from the research paper. Reference section is also provided that is in APA format. The writing style and note should be academic in nature. There is no need for conclusion paragraph.

        FORMAT OF THE PROPOSED METHODOLOGY SECTION:
        1) Propose a Methodology different from what others have done in Literature Review
        2) This Methodology should highlight how the challenges or current shortcomings are resolved

        NOTE: Strictly follow the above format and do not start paragraphs with sentences like "In conclusion,", "Overall," or similar.
        
        LITERATURE REVIEW: {literature}

        CHALLENGES SECTION: {challenges}
    """)

    def __init__(self):
        self.model = self.get_ChatOpenAImodel()

    def mini(self, challenges, literature):
        proposal_messages = [
            SystemMessage(content=self.proposal_mini_instruction.format(literature=literature, challenges=challenges)),
            HumanMessage(content="Write the Proposal section")
        ]
        response = self.model(proposal_messages)
        return response.content