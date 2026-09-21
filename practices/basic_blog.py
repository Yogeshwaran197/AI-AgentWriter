from langchain_core.runnables import add
from pathlib import Path

import operator
from typing import TypedDict, List, Annotated

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

load_dotenv()

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env")

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Groq API key missing, Please add API key in .env")


llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    api_key= GROQ_API_KEY
)

class task(BaseModel):
    id : int
    title : str
    brief : str = Field(..., description="what to cover")


class Plan(BaseModel):
    blog_title : str
    tasks : List[task]


class state(TypedDict):
    topic : str
    plan : Plan
    section : Annotated[List[str], operator.add]
    final : str


def orchestrator(state: state) -> dict:

    topic =  state['topic']
    llm_with_schema =  llm.with_structured_output(Plan)
    plan = llm_with_schema.invoke([
        SystemMessage(content = "Create a blog plan with 5-7 sections on the following topic."),
        HumanMessage(content = f"{topic}")
    ])

    return {"plan": plan}


def fanout(state : state):

    return [
        Send("worker" , {"task":t, "topic":state["topic"], "plan":state["plan"]})
    for t in state["plan"].tasks
    ]


def worker(payload : dict) ->  dict:

    task = payload['task']
    topic = payload['topic']
    plan = payload['plan']

    blog_title = plan.blog_title
    
    section_md =  llm.invoke([
        SystemMessage(content="Write one clean Markdown section."),
        HumanMessage(content=f"""
                Blog: {blog_title}
                Topic: {topic}

                Section: {task.title}
                Brief: {task.brief}

                Return only section content in markdown
            """)
    ]).content.strip()

    return {"section" : [section_md]}

def reducer(state : state) -> dict:

    title = state['topic']
    body = "\n\n".join(state['section']).strip()

    final_md = f"# {title}\n\n{body}"
    filename = title.lower().replace(" ", "_") +".md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")

    return {"final" : final_md}

graph = StateGraph(state)

graph.add_node("orchestrator", orchestrator)
graph.add_node("worker", worker)
graph.add_node("reducer", reducer)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges("orchestrator", fanout , ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)


database_url = get_database_url()

conn = psycopg.connect(
    database_url,
    autocommit=True,
    row_factory= dict_row
)

checkpointer = PostgresSaver(
    conn
)

checkpointer.setup()

app =  graph.compile(checkpointer=checkpointer)

config = {"configurable" : {"thread_id" : "yogi05"}}

response = app.invoke({"topic": "Write a blog on Self Attention", "sections": []}, config=config)

print("="*60)
print("BLOG")
print("="*60)

print(f"\n\nTopic : {response["topic"]}")
print(f"\n\nFinal : {response["final"]}")






