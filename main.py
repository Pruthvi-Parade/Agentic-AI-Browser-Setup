from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

template = "Translate the following English text to French: {text}"
prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

chain = prompt | llm | StrOutputParser()

answer = chain.invoke({"text": "I love french"})
print(answer)

