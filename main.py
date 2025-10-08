# Python AI Agent using LangChain

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
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
    Adds a task to the todo list.
    Args:
        task (str): The task to add to the todo list.
        desc (str): Description of the task.
    '''
    todoist.add_task(content=task, description=desc)

@tool
def show_tasks():
    '''
    Shows all tasks from todo list.
    Use this tool when the user wants to see their tasks.
    '''
    results_paginator = todoist.get_tasks()
    tasks = list()
    for task_list in results_paginator:
        for task in task_list:
            tasks.append(task)
    return tasks

# Tool functions to be used in llm
tools = [add_task, show_tasks]

# Configures Google LLM Gemini for use
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=gemini_key,
    temperature=0.3
)

# Prompt for how the LLM will answer
system_prompt = """You are a helpful assistant. 
You will help the user add tasks.
You will help the user show existing tasks.
If the user asks to show the tasks: for example, 'show me the tasks'
print out the tasks to the user. Print them in a bullet list format."""

# {input} dynamically saves space for input to be set later in the code
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

# List of history of user and AI messages
history = list()

# Continuously runs
while True:
    user_input = input("Enter a task: ")

    # Exits program
    if (user_input == "exit"):
        break

    response = agent_executor.invoke({"input":user_input, "history":history})
    print(response['output'])
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response['output']))
