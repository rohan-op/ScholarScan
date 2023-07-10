from base import Base
from langchain.schema import (SystemMessage, HumanMessage)
import time
class Reference(Base):

    reference_apa_mini_instruction = """
        Based on the information provided below, your job is to create a reference section for an academic review paper. Make sure that you are only referencing the papers that are mention below. The reference section should be in APA format.

        CONTENT:
    """

    delimiter = " "

    def __init__(self):
        self.model = self.get_ChatOpenAImodel()

    def mini(self,summary):
    
        reference_messages = [
            SystemMessage(content=self.reference_apa_mini_instruction+summary),
            HumanMessage(content="Write the reference section")
        ]
        response = self.model(reference_messages)
        return response.content
    
    def combine(self, summaries):
        references = []
        for summary in summaries:
            print("Combining references...")
            time.sleep(20)
            references.append(self.mini(summary=summary))
        return references
    
    def refine():
        return 0
    
    def latex():
        return 0