from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def QueryUnderstandingTool(query: str) -> bool:
    """Return True if the query seems to request a visualisation based on keywords."""
    # Use LLM to understand intent instead of keyword matching
    messages = [
        {"role": "system", "content": "You are a smart assistant that determines if a query is requesting a data visualization. "
            "Return only 'true' if the query implies or explicitly asks for a plot, chart, trend over time, "
            "comparison between values, or any visual representation of data. Return 'false' if the query is asking only for text or numbers."},
        {"role": "user", "content": query}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.1,
        max_tokens=5  
    )
    
    # Extract the response and convert to boolean
    intent_response = response.choices[0].message.content.strip().lower()
    return intent_response == "true"