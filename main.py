# Python AI Agent using LangChain

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic_core.core_schema import model_field
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from todoist_api_python.api import TodoistAPI

load_dotenv()

todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

todoist = TodoistAPI(todoist_api_key)

@tool
def add_task(task:str, desc:str=None):
    '''
    Adds a task to the todo list
    Args:
        task (str): The task to add to the todo list
        desc (str): Description of the task
    '''
    todoist.add_task(content=task, description=desc)

# Functions to be used in llm
tools = [add_task]

# Configures Google LLM Gemini for use
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=gemini_key,
    temperature=0.3
)

# Default prompt for how the LLM will answer
system_prompt = "You are a helpful assistant."
# system_prompt = "You are a helpful assistant. You will help the user add tasks."

# {input} dynamically saves space for input to come later in the code
# Prepares chat prompt
prompt = ChatPromptTemplate([
    ("system", system_prompt), 
    MessagesPlaceholder("history"),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

# Creates specialized scope llm to answer user_input based on system_prompt
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# List of history of user and system messages
history = []

# Continously runs
while True:
    user_input = input("Enter a task: ")

    # Exits program
    if (user_input == "exit"):
        break

    response = agent_executor.invoke({"input":user_input, "history":history})
    print(response['output'])
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response['output']))

