from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools import search_tool, wiki_tool, save_tool

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.3)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research assistant.

Your job:
- Research the user's query thoroughly using tools.
- If the user asks to "save it to a file" or "save to a separate file", you MUST use the save_text_to_file tool with a proper filename.
- At the end, always respond with ONLY the JSON in the exact format specified. No extra text.

{format_instructions}"""),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())


tools = [search_tool, wiki_tool, save_tool]

agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8,
)


# ====================== RUN ======================
if __name__ == "__main__":
    query = input("What can I help you research? ")
    
    raw_response = agent_executor.invoke({"query": query})

    output = raw_response.get("output")

    try:
        if isinstance(output, list):
            text = output[0].get("text", str(output))
        else:
            text = str(output)

        structured_response = parser.parse(text)
        print("\n=== Research Completed ===\n")
        print(structured_response)

        # Optional: Auto-save if user mentioned saving
        if any(word in query.lower() for word in ["save", "file", "txt"]):
            from tools import save_to_txt
            save_result = save_to_txt(
                data={
                    "topic": structured_response.topic,
                    "summary": structured_response.summary,
                    "sources": structured_response.sources
                },
                filename="research_output.txt"   
            )
            print(save_result)

    except Exception as e:
        print("Error parsing response:", e)
        print("\nRaw Output:\n", output)