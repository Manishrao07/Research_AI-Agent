from langchain_core.tools import tool
import requests

@tool
def search_wikipedia(topic: str) -> str:
    """
    Wikipedia se kisi bhi topic ki detailed background information lene ke liye.
    History, definitions, aur established facts ke liye use karo.
    """
    try:
        # Wikipedia API directly use karo (library ki jagah)
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_")
        
        response = requests.get(url, headers={"User-Agent": "ai-research-agent/1.0"})
        
        if response.status_code == 200:
            data = response.json()
            output = f"Wikipedia: {data['title']}\n"
            output += f"URL: {data['content_urls']['desktop']['page']}\n\n"
            output += f"Summary:\n{data['extract']}\n"
            return output
        else:
            # Fallback — search endpoint use karo
            search_url = f"https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": topic,
                "format": "json",
                "srlimit": 1
            }
            r = requests.get(search_url, params=params, headers={"User-Agent": "ai-research-agent/1.0"})
            data = r.json()
            title = data["query"]["search"][0]["title"]
            
            # Ab us title ka summary lo
            url2 = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
            r2 = requests.get(url2, headers={"User-Agent": "ai-research-agent/1.0"})
            data2 = r2.json()
            
            output = f"Wikipedia: {data2['title']}\n"
            output += f"URL: {data2['content_urls']['desktop']['page']}\n\n"
            output += f"Summary:\n{data2['extract']}\n"
            return output
            
    except Exception as e:
        return f"Wikipedia search mein error: {str(e)}"