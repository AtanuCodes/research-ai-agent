from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.tools import Tool
from datetime import datetime
import os


def save_to_txt(data: dict | str, filename: str = "research_output.txt"):
    """
    Saves research data to a text file.
    """
    try:
        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(data, dict):
            content = f"""Topic: {data.get('topic', 'N/A')}
Summary:
{data.get('summary', '')}

Sources:
{chr(10).join(['- ' + s for s in data.get('sources', [])])}
"""
        else:
            content = str(data)

        formatted_text = f"""--- Research Output ---
Timestamp: {timestamp}
{content}
{'='*80}

"""

        with open(filename, "a", encoding="utf-8") as f:
            f.write(formatted_text)

        return f"✅ Successfully saved to: {filename}"

    except Exception as e:
        return f"❌ Failed to save file: {str(e)}"


# ==================== TOOLS ====================

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="""Useful when user asks to save the research to a file.
    Example: g:/Ai-Agent/ml_dl_research.txt""",
)


search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for latest or general information.",
)



api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=800)

def safe_wiki_search(query: str) -> str:
    try:
        return api_wrapper.run(query)
    except Exception as e:
        return f"Wikipedia error: {str(e)}. Try using search tool instead."

wiki_tool = Tool(
    name="wikipedia",
    func=safe_wiki_search,
    description="Search Wikipedia for reliable factual information.",
)