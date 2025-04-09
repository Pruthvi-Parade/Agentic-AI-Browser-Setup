import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Instantiate OpenAI with the API key
llm = OpenAI(temperature=0.7)

prompt = PromptTemplate(
    input_variables=["task", "tone", "summary"],
    template="Task:{task}\nTone:{tone}\nSummary:{summary}"
)

chain = prompt | llm

result = chain.invoke({"task": "Summarize the provided text", "tone": "scarastic", "summary": """What is LangChain?

LangChain is a framework designed to simplify the development of applications powered by large language models (LLMs). It provides tools and integrations that allow developers to harness the capabilities of LLMs effectively, enabling tasks such as natural language understanding, text generation, and more. ​
GitHub


Key Components of LangChain:

Language Models (LLMs): At its core, LangChain integrates with various LLMs, allowing applications to process and generate human-like text. These models can perform tasks like translation, summarization, and question-answering.​

Prompt Templates: These are predefined structures that guide the LLM's responses. By providing a consistent format, prompt templates ensure that the model's outputs are tailored to specific tasks or applications.​

Chains: LangChain allows the linking of multiple components (like LLMs and prompt templates) in a sequence, known as a "chain." This enables the creation of more complex workflows where the output of one component becomes the input for another.​


Agents: Agents are advanced components that can make decisions about which actions to take based on user input. They can interact with external tools, retrieve information, and use LLMs to generate responses, making applications more dynamic and interactive.​
"""})

print(result)