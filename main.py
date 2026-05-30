from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# from langchain_anthropic import ChatAnthropic
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain.agents import create_tool_calling_agent, AgentExecutor
# from tools import search_tool, wiki_tool, save_tool

load_dotenv()

llm = ChatGroq(model="qwen/qwen3-32b",)
response = llm.invoke("What is the oldest religion in the world?")
print(response.content)