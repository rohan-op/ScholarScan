from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI

loader = TextLoader('R1.txt')
documents = loader.load()

llm = OpenAI(temperature=1,openai_api_key="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")
# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split your docs into texts
texts = text_splitter.split_documents(documents)

# There is a lot of complexity hidden in this one line. I encourage you to check out the video above for more detail
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
summary = chain.run(texts)
print(summary)