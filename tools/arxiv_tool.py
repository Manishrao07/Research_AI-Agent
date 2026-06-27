from langchain_core.tools import tool
import requests
import re


@tool
def search_arxiv(query: str) -> str:
    """
    Arxiv.org se academic research papers search karne ke liye.
    Scientific, technical, AI/ML, physics, math jaise topics ke liye use karo
    jahan peer-reviewed ya preprint research papers relevant hon.
    """
    try:
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 4,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return f"Arxiv search mein error aaya: status code {response.status_code}"

        xml_text = response.text

        # Simple regex-based parsing (XML library use karne ki zaroorat nahi
        # itne simple structure ke liye, aur extra dependency bachti hai)
        entries = re.findall(r'<entry>(.*?)</entry>', xml_text, re.DOTALL)

        if not entries:
            return f"Arxiv par '{query}' ke liye koi paper nahi mila."

        output = f"Arxiv research papers for: '{query}'\n\n"
        for i, entry in enumerate(entries, 1):
            title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            link_match = re.search(r'<id>(.*?)</id>', entry, re.DOTALL)
            published_match = re.search(r'<published>(.*?)</published>', entry, re.DOTALL)

            title = title_match.group(1).strip().replace('\n', ' ') if title_match else "Unknown title"
            summary = summary_match.group(1).strip().replace('\n', ' ')[:300] if summary_match else ""
            link = link_match.group(1).strip() if link_match else ""
            published = published_match.group(1).strip()[:10] if published_match else ""

            output += f"Paper {i}: {title}\n"
            output += f"Published: {published}\n"
            output += f"URL: {link}\n"
            output += f"Abstract: {summary}...\n\n"

        return output
    except Exception as e:
        return f"Arxiv search mein error aaya: {str(e)}"