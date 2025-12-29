from langchain_huggingface import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain.schema.runnable import RunnableParallel
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# ------------------ MODELS ------------------

llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    task="text-generation"
)

model = ChatGoogleGenerativeAI(
    model="gemini-pro"
)

# ------------------ PROMPTS ------------------

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the text:\n{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short questions from the text:\n{text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="""
Merge the notes and quiz into a single document.

NOTES:
{notes}

QUIZ:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

# ------------------ CHAINS ------------------

parallel_chain = RunnableParallel({
    "notes": prompt1 | llm | parser,
    "quiz": prompt2 | model | parser
})

merge_chain = prompt3 | llm | parser

chain = parallel_chain | merge_chain

# ------------------ RUN ------------------

text = """
Artificial Intelligence is the simulation of human intelligence
in machines that are programmed to think and learn.
"""

result = chain.invoke({"text": text})
print(result)


chain.get_graph().print_ascii()