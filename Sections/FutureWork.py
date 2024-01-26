from base import Base
from langchain.prompts import PromptTemplate
from langchain.schema import (SystemMessage, HumanMessage)

class FutureWork(Base):

    future_mini_instruction = PromptTemplate(
        input_variables= ["proposal", "challenges"],
        template = """
        You are provided the literature review and challenges and methodology section. Your job is write future work section for this academic review paper. Do in-text citation if you are using ideas from the research paper. Reference section is also provided that is in APA format. The writing style and note should be academic in nature. There is no need for conclusion paragraph.

        FORMAT OF THE FUTURE WORK SECTION:
        1) Based on proposed methodology, suggest future improvements
        2) Explain how these future improvements will help resolve challenges that the current methodology is not able to resolve.

        NOTE: Strictly follow the above format and do not start paragraphs with sentences like "In conclusion,", "Overall," or similar.
        
        PROPOSED METHODOLOGY: {proposal}

        CHALLENGES SECTION: {challenges}
    """)

    def __init__(self):
        self.model = self.get_ChatOpenAImodel()

    def mini(self, challenges, proposal):
        future_messages = [
            SystemMessage(content=self.future_mini_instruction.format(proposal=proposal, challenges=challenges)),
            HumanMessage(content="Write the Future Work section")
        ]
        response = self.model(future_messages)
        return response.content