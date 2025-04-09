from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
def get_weather_by_city(city: str) -> str:
    """Get the weather for a given city"""
    return f"The weather in {city} is sunny."

if __name__ == "__main__":
    mcp.run(transport="stdio")