from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
    ("human", "Generate 5 interesting facts about {topic}")
])

parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({"topic": "cricket"}))
