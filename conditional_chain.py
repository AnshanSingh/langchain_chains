from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# ------------------ MODEL ------------------

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0
)

# ------------------ OUTPUT SCHEMA ------------------

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback"
    )

parser = PydanticOutputParser(pydantic_object=Feedback)

# ------------------ CLASSIFIER PROMPT ------------------

prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback as positive or negative.

{format_instructions}

Feedback:
{feedback}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

classifier_chain = prompt1 | llm | parser

# ------------------ RESPONSE PROMPTS ------------------

positive_prompt = PromptTemplate(
    template="Write a polite response to this positive feedback:\n{feedback}",
    input_variables=["feedback"]
)

negative_prompt = PromptTemplate(
    template="Write a polite apology and solution for this negative feedback:\n{feedback}",
    input_variables=["feedback"]
)

text_parser = StrOutputParser()

# ------------------ BRANCH ------------------

branch_chain = RunnableBranch(
    (lambda x: x["sentiment"] == "positive", positive_prompt | llm | text_parser),
    (lambda x: x["sentiment"] == "negative", negative_prompt | llm | text_parser),
    RunnableLambda(lambda _: "Could not determine sentiment")
)

# ------------------ FINAL CHAIN ------------------

chain = classifier_chain | branch_chain

# ------------------ RUN ------------------

print(chain.invoke({"feedback": "This is a terrible phone"}))
