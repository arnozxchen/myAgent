from colorama import Fore, Style
import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from tools import search_tool, financial_tools

load_dotenv()


class Response(BaseModel):
    # Concept: Briefly state the topic
    concept: str = Field(
        description="The specific financial or mathematical concept being addressed, e.g., 'Present Value of Annuity' or 'Bond Valuation'."
    )

    answer: str = Field(
        description="The answer of the user's questions, could be simple numerical values, the selection of MCQ or a statement."
    )
    
    # Formula: The algebraic formula
    formula: str = Field(
        description="The theoretical algebraic formula used for the calculation, formatted in plain text."
    )
    
    # Substitute Formula: The plug-in step (Critical for your requirement)
    substitute_formula: str = Field(
        description="The mathematical equation showing the actual numerical values plugged into the formula, formatted in plain text. Do not just give the final answer here, show the substitution step. Example: '50 \\times \\frac{1 - (1 + 0.05)^{-3}}{0.05}'"
    )
    
    # Human Response: The final conversational output
    human_response: str = Field(
        description="A friendly, natural language summary of the final answer addressed directly to the user."
    )

    # Tools Used: Tracking
    tools_used: list[str] = Field(
        description="A list of the exact names of the tools you called to solve this problem. If no tools were used, return an empty list."
    )


def get_chat_model(provider="rapid-mlx", model_name="default"):
    # Unified configuration (Unified Configuration)
    configs = {
        "rapid-mlx": {
            "base_url": "http://localhost:8000/v1",
            "api_key": "not-needed"
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "not-needed"
        }
    }
    
    if provider not in configs:
        raise ValueError(f"Unsupported provider: {provider}")
        
    return ChatOpenAI(
        base_url=configs[provider]["base_url"],
        api_key=configs[provider]["api_key"],
        model=model_name
    )

llm = get_chat_model(provider="rapid-mlx", model_name="qwen3.5-4b")
# llm = get_chat_model(provider="ollama", model_name="llama3.2")

parser = PydanticOutputParser(pydantic_object = Response)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a learning tutor that will help generate the answer of questions.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, *financial_tools]
agent = create_tool_calling_agent(
    llm = llm,
    prompt = prompt,
    tools = tools
)

agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)
print("What can I help you?")
query = input("Question: ")
raw_response = agent_executor.invoke({"query": query})

print(raw_response)
print(Fore.GREEN + "=" * 40 + " Answer " + "="*40 + Style.RESET_ALL)



try:
    raw_output_str = raw_response.get("output").strip()
    
    # Parse to Python object
    try:
        parsed_data = json.loads(raw_output_str)
        
        if isinstance(parsed_data, list) and len(parsed_data) > 0:
            parsed_data = parsed_data[0] # Extract the first dictionary (Extract dict)
            
        if isinstance(parsed_data, dict):
            structured_response = Response(**parsed_data)
        else:
            raise ValueError("Parsed JSON is neither a list of dicts nor a dict.")
            
    except json.JSONDecodeError:
        # Fallback: Let LangChain's parser try if standard JSON parsing fails
        structured_response = parser.parse(raw_output_str)

    print(Fore.CYAN + "Answer: " + Style.RESET_ALL + structured_response.answer)
    print(Fore.CYAN + "Concept: " + Style.RESET_ALL + structured_response.concept)
    print(Fore.CYAN + "Calculation: " + Style.RESET_ALL + structured_response.substitute_formula)
    print(Fore.CYAN + "Tutor says: " + Style.RESET_ALL + structured_response.human_response)

except Exception as e:
    print("Error parsing response:", e)
    print("Raw Response Output -", raw_response.get("output"))