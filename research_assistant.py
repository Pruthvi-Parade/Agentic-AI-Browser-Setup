# research_assistant.py

import os
from typing import TypedDict, Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.agents import create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json

from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()

# 1. Define Tools
search_tool = TavilySearchResults(max_results=5)

@tool
def summarize_tool(documents: list[str]) -> str:
    """Summarize the given documents."""
    docs = [Document(page_content=doc) for doc in documents]
    chain = load_summarize_chain(llm, chain_type="stuff")
    return chain.run(docs)

@tool
def qa_tool(question: str, documents: list[str]) -> str:
    """Answer a question based on the provided documents."""
    docs = [Document(page_content=doc) for doc in documents]
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain.run(question)

@tool
def recommend_tool(summary: str) -> str:
    """Generate recommendations based on the summary."""
    prompt = f"Based on the following summary, provide recommendations:\n{summary}"
    response = llm.invoke(prompt)
    return response.content

# Combine all tools
tools = [search_tool, summarize_tool, qa_tool, recommend_tool]

# 2. Define the State
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    research_data: list[str]                # Search results or documents
    summary: str                            # Summarized text
    intermediate_steps: list                # Required for agent execution

# 3. Define the Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Use the tools to help the user with their requests. Maintain context from previous interactions."),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)

# 4. Define Custom Tool Node
def tools_node(state: ResearchState):
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls if hasattr(last_message, 'tool_calls') else []
    
    for tool_call in tool_calls:
        tool_name = tool_call["function"]["name"]
        tool_input = json.loads(tool_call["function"]["arguments"])
        
        if tool_name == "tavily_search_results_json":
            results = search_tool._run(**tool_input)
            state["research_data"].extend([r["content"] for r in results])
            state["messages"].append(ToolMessage(content=str(results), tool_call_id=tool_call["id"], name="tavily_search_results_json"))
        elif tool_name == "summarize_tool":
            summary = summarize_tool._run(documents=state["research_data"])
            state["summary"] = summary
            state["messages"].append(ToolMessage(content=summary, tool_call_id=tool_call["id"], name="summarize_tool"))
        elif tool_name == "qa_tool":
            answer = qa_tool._run(question=tool_input["question"], documents=state["research_data"])
            state["messages"].append(ToolMessage(content=answer, tool_call_id=tool_call["id"], name="qa_tool"))
        elif tool_name == "recommend_tool":
            recommendation = recommend_tool._run(summary=state["summary"])
            state["messages"].append(ToolMessage(content=recommendation, tool_call_id=tool_call["id"], name="recommend_tool"))
    
    return state

# 5. Build the Graph
graph = StateGraph(ResearchState)

# Add nodes
graph.add_node("agent", agent)
graph.add_node("tools", tools_node)

# Set entry point
graph.set_entry_point("agent")

# Conditional edges
def should_continue(state):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

# Compile the graph
app = graph.compile()

def main():
    # Initial state
    state = ResearchState(messages=[], research_data=[], summary="", intermediate_steps=[])
    
    print("Research Assistant initialized. Type 'quit' to exit.")
    print("Example queries:")
    print("- Search for recent advances in renewable energy")
    print("- Summarize the findings")
    print("- What are the benefits of solar energy?")
    print("- Recommend strategies for adopting renewable energy")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
            
        state["messages"].append(HumanMessage(content=user_input))
        result = app.invoke(state)
        print("Result: ",result)


if __name__ == "__main__":
    main()
        # for msg in result['messages']:
        #     if isinstance(msg, AIMessage):
        #         arguments_json = msg.additional_kwargs['function_call']['arguments']
        #         arguments = json.loads(arguments_json)
        #         summary = arguments.get('summary')
        #         print("Summary:\n", summary)
        #         break