from langchain_core.tools import tool
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def search_web(query: str) -> str:
    """
    Internet pe kuch bhi search karne ke liye.
    Latest news, current events, ya koi bhi topic dhundhne ke liye use karo.
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        results = client.search(
            query=query,
            max_results=5,
            search_depth="advanced"
        )
        
        # Results ko clean format mein convert karo
        output = f"Web search results for: '{query}'\n\n"
        
        for i, result in enumerate(results['results'], 1):
            output += f"Source {i}: {result['title']}\n"
            output += f"URL: {result['url']}\n"
            output += f"Content: {result['content']}\n\n"
        
        return output
        
    except Exception as e:
        return f"Search mein error aaya: {str(e)}"