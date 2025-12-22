#here we will add a rag tool in our chatbot


# backend.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import requests
import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# -------------------
# 1. LLM
# -------------------
# llm = ChatOpenAI()
# llm=ChatGoogleGenerativeAI(model='gemini-2.0-flash')
llm=ChatGroq(model='Llama-3.3-70b-versatile')


def pdf_loader(pdf_path):
    
    loader=PyPDFLoader(pdf_path)
    documents=loader.load()
    #now we got the documents 
    return documents 

#now we have to split it and store in teh vector db 
def chunk_storage(docs):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=text_splitter.split_documents(docs)
    #now we required a embedding model for stroing all the chunks as embedding in the vector db 
    
    embedding_model=GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
    vector_db=Chroma.from_docments(texts=chunks,embedding=embedding_model) 
    #lets use it as a retriever 
    
    retriever=vector_db.as_retriever(search_type='similarity',kwargs={'k':4})
    #lets store these chunks into the vector db 
    return retriever 

 
    


#lets add a rag tool here so that our chatbot can use it when the user upload a document 
@tool 
def rag_tool(query: str)-> dict :
    """This tool is for retrieving the related content from teh pdf document when the 
    user ask any question related to the document .
    and if it return not related content please tell i don't know"""
    docs=pdf_loader('sample.pdf') #for loading
    retriever=chunk_storage(docs) #for storing teh chunks 
    related_chunks=retriever.invoke(query) 
    context=[doc.page_content for doc in related_chunks]
    metadata=[doc.metadata for doc in related_chunks]
    return {'query':query,'context':context,'metadata':metadata}
    
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}




@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

#now lets add another tool named weather api 

@tool 
def get_weather(location: str)-> dict:
    """"Fetch the current weather for a given city using the openweather api """
    api_key=os.environ['WEATHER_API_KEY']
    url = f"https://api.weatherapi.com/v1/current.json"
    params = {
        "key": api_key,
        "q": location,
        "aqi": "no"  # Optional: disables air quality data
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

tools = [search_tool, get_stock_price, calculator,get_weather,rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')
thread_id='33'
chatbot = graph.compile(checkpointer=checkpointer)
config={'configurable':{'thread_id':thread_id}}
initial_state={'messages':[HumanMessage(content="what is machine learning")]}
final_state=chatbot.invoke(initial_state,config=config)
print(final_state['messages'][-1].content)