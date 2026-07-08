import os
import json
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
api_key=os.getenv("GROQ_API_KEY") 
#print(f"GROQ_API_KEY: {api_key}")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

client = Groq(api_key=api_key)

system_prompt = """
You are a helpful assistant.
Always respond only in valid JSON.
"""

user_prompt = """
Give details about Langchain in JSON format with:topic, definition, advantages, disadvantages
"""

raw_response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=200,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
)

raw_output = raw_response.choices[0].message.content

llm=ChatGroq(model="llama-3.3-70b-versatile",
               api_key=api_key,
                temperature=0.9, 
                max_tokens=100
)


langchain_response = llm.invoke(
    [
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_prompt)
    ]
)

langchain_output = langchain_response.content

result = {
    "raw_sdk_response": raw_output,
    "langchain_response": langchain_output
}

print(json.dumps(result, indent=4))