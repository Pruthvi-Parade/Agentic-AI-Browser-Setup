# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o")

server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["../servers/math_server.py"],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "what's weather in Pune?"})
            
            # Format the agent response in a readable way
            print("Agent Response:")
            print("---------------")
            for message in agent_response["messages"]:
                if hasattr(message, "content") and message.content:
                    print(f"{message.__class__.__name__}: {message.content}")
                if hasattr(message, "tool_calls") and message.tool_calls:
                    print(f"Tool Calls: {[call['name'] for call in message.tool_calls]}")
                if hasattr(message, "name") and message.name:
                    print(f"Tool '{message.name}' returned: {message.content}")
            
            # Print the final answer
            final_message = agent_response["messages"][-1]
            if hasattr(final_message, "content") and final_message.content:
                print("\nFinal Answer:")
                print(final_message.content)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())