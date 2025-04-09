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

# Load environment variables from .env file
load_dotenv()

# Step 1: Initialize the Language Model
# We're using OpenAI's GPT-4o-mini model with temperature=0 (more deterministic responses)
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()  # For converting text to vector representations

# Step 2: Define Tools for the Assistant
# Search tool - uses Tavily API to search the web
search_tool = TavilySearchResults(max_results=5)

# Summarization tool - takes a list of documents and creates a summary
@tool
def summarize_tool(documents: list[str]) -> str:
    """Summarize the given documents."""
    docs = [Document(page_content=doc) for doc in documents]
    chain = load_summarize_chain(llm, chain_type="stuff")
    return chain.run(docs)

# Question-answering tool - answers questions based on provided documents
@tool
def qa_tool(question: str, documents: list[str]) -> str:
    """Answer a question based on the provided documents."""
    docs = [Document(page_content=doc) for doc in documents]
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain.run(question)

# Recommendation tool - generates recommendations based on a summary
@tool
def recommend_tool(summary: str) -> str:
    """Generate recommendations based on the summary."""
    prompt = f"Based on the following summary, provide recommendations:\n{summary}"
    response = llm.invoke(prompt)
    return response.content

# Combine all tools into a list
tools = [search_tool, summarize_tool, qa_tool, recommend_tool]

# Step 3: Define the State Structure
# This defines what data we'll track during the conversation
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    research_data: list[str]                # Search results or documents
    summary: str                            # Summarized text
    intermediate_steps: list                # Required for agent execution

# Step 4: Create the Agent with a Prompt
# The prompt defines how the assistant should behave
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Use the tools to help the user with their requests. Maintain context from previous interactions."),
    MessagesPlaceholder(variable_name="messages"),  # Where user messages go
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # For agent's thinking process
])

# Create the agent with our LLM, tools, and prompt
agent = create_openai_functions_agent(llm, tools, prompt)

# Step 5: Define How Tools Are Executed
# This function handles the actual execution of tools when the agent decides to use them
def tools_node(state: ResearchState):
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls if hasattr(last_message, 'tool_calls') else []
    
    for tool_call in tool_calls:
        tool_name = tool_call["function"]["name"]
        tool_input = json.loads(tool_call["function"]["arguments"])
        
        # Handle search tool
        if tool_name == "tavily_search_results_json":
            results = search_tool._run(**tool_input)
            state["research_data"].extend([r["content"] for r in results])
            state["messages"].append(ToolMessage(content=str(results), tool_call_id=tool_call["id"], name="tavily_search_results_json"))
        # Handle summarize tool
        elif tool_name == "summarize_tool":
            summary = summarize_tool._run(documents=state["research_data"])
            state["summary"] = summary
            state["messages"].append(ToolMessage(content=summary, tool_call_id=tool_call["id"], name="summarize_tool"))
        # Handle Q&A tool
        elif tool_name == "qa_tool":
            answer = qa_tool._run(question=tool_input["question"], documents=state["research_data"])
            state["messages"].append(ToolMessage(content=answer, tool_call_id=tool_call["id"], name="qa_tool"))
        # Handle recommendation tool
        elif tool_name == "recommend_tool":
            recommendation = recommend_tool._run(summary=state["summary"])
            state["messages"].append(ToolMessage(content=recommendation, tool_call_id=tool_call["id"], name="recommend_tool"))
    
    return state

# Step 6: Build the Conversation Flow Graph
# This defines how the conversation flows between the agent and tools
graph = StateGraph(ResearchState)

# Add the main components as nodes in the graph
graph.add_node("agent", agent)
graph.add_node("tools", tools_node)

# Set where the conversation starts
graph.set_entry_point("agent")

# Define when to use tools vs. when to end the conversation
def should_continue(state):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"  # If agent wants to use a tool, go to tools node
    return END  # Otherwise, end this round of conversation

# Connect the nodes with edges
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")  # After using tools, go back to agent

# Compile the graph into a runnable application
app = graph.compile()

# Step 7: Main Function to Run the Assistant
def main():
    # Initialize the conversation state
    state = ResearchState(messages=[], research_data=[], summary="", intermediate_steps=[])
    
    # Print welcome message and examples
    print("Research Assistant initialized. Type 'quit' to exit.")
    print("Example queries:")
    print("- Search for recent advances in renewable energy")
    print("- Summarize the findings")
    print("- What are the benefits of solar energy?")
    print("- Recommend strategies for adopting renewable energy")
    
    # Main conversation loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
            
        # Add user message to state and run the conversation
        state["messages"].append(HumanMessage(content=user_input))
        result = app.invoke(state)
        print("Result: ", result)


# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
        # Commented out code for extracting summary from function calls
        # for msg in result['messages']:
        #     if isinstance(msg, AIMessage):
        #         arguments_json = msg.additional_kwargs['function_call']['arguments']
        #         arguments = json.loads(arguments_json)
        #         summary = arguments.get('summary')
        #         print("Summary:\n", summary)
        #         break