import os
import operator
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict, List, Literal, Annotated
from pydantic import Field, BaseModel
from langgraph.types import Send
from langgraph.graph import START ,END , StateGraph

import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY :
    raise ValueError("GROQ API key missing, add GROQ API key in .env")
    
llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    api_key= GROQ_API_KEY
)


class Task(BaseModel):

    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3–5 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(
        ...,
        description="Target word count for this section (120–450).",
    )
    section_type: Literal[
        "intro", "core", "examples", "checklist", "common_mistakes", "conclusion"
    ] = Field(
        ...,
        description="Use 'common_mistakes' exactly once in the plan.",
    )


class Plan(BaseModel):

    blog_title: str
    audience: str = Field(..., description="Who this blog is for.")
    tone : str = Field(..., description="Writing tone (e.g., practical, crisp).")
    task : List[Task]


class AgentState(TypedDict):

    topic : str
    plan : Plan
    section : Annotated[List[str], operator.add]
    final : str


def orchestrator(state: AgentState) -> dict:

    topic = state['topic']
    llm_with_plan = llm.with_structured_output(Plan)
    
    result =  llm_with_plan.invoke([
        SystemMessage(content="You are a senior technical writer and developer advocate. Your job is to produce a "
                    "highly actionable outline for a technical blog post.\n\n"
                    "Hard requirements:\n"
                    "- Create 5–7 sections (tasks) that fit a technical blog.\n"
                    "- Each section must include:\n"
                    "  1) goal (1 sentence: what the reader can do/understand after the section)\n"
                    "  2) 3–5 bullets that are concrete, specific, and non-overlapping\n"
                    "  3) target word count (120–450)\n"
                    "- Include EXACTLY ONE section with section_type='common_mistakes'.\n\n"
                    "Make it technical (not generic):\n"
                    "- Assume the reader is a developer; use correct terminology.\n"
                    "- Prefer design/engineering structure: problem → intuition → approach → implementation → "
                    "trade-offs → testing/observability → conclusion.\n"
                    "- Bullets must be actionable and testable (e.g., 'Show a minimal code snippet for X', "
                    "'Explain why Y fails under Z condition', 'Add a checklist for production readiness').\n"
                    "- Explicitly include at least ONE of the following somewhere in the plan (as bullets):\n"
                    "  * a minimal working example (MWE) or code sketch\n"
                    "  * edge cases / failure modes\n"
                    "  * performance/cost considerations\n"
                    "  * security/privacy considerations (if relevant)\n"
                    "  * debugging tips / observability (logs, metrics, traces)\n"
                    "- Avoid vague bullets like 'Explain X' or 'Discuss Y'. Every bullet should state what "
                    "to build/compare/measure/verify.\n\n"
                    "Ordering guidance:\n"
                    "- Start with a crisp intro and problem framing.\n"
                    "- Build core concepts before advanced details.\n"
                    "- Include one section for common mistakes and how to avoid them.\n"
                    "- End with a practical summary/checklist and next steps.\n\n"
                    "Output must strictly match the Plan schema."),
            HumanMessage(content=f"{topic}")
    ])

    return {"plan":result}


def fanout(state:AgentState):

    return [
        Send(
            "worker",
            {"task":t, "topic":state['topic'], "plan":state['plan']},
        ) for t in state["plan"].task
    ]


def worker(payload: dict) -> dict:

    topic = payload["topic"]
    task = payload["task"]
    plan = payload["plan"]

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    section_md = llm.invoke(
        [
            SystemMessage(
    content=(
        "You are a senior technical writer and developer advocate. Write ONE section of a technical blog post in Markdown.\n\n"
        "Hard constraints:\n"
        "- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).\n"
        "- Stay close to the Target words (±15%).\n"
        "- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).\n\n"
        "Technical quality bar:\n"
        "- Be precise and implementation-oriented (developers should be able to apply it).\n"
        "- Prefer concrete details over abstractions: APIs, data structures, protocols, and exact terms.\n"
        "- When relevant, include at least one of:\n"
        "  * a small code snippet (minimal, correct, and idiomatic)\n"
        "  * a tiny example input/output\n"
        "  * a checklist of steps\n"
        "  * a diagram described in text (e.g., 'Flow: A -> B -> C')\n"
        "- Explain trade-offs briefly (performance, cost, complexity, reliability).\n"
        "- Call out edge cases / failure modes and what to do about them.\n"
        "- If you mention a best practice, add the 'why' in one sentence.\n\n"
        "Markdown style:\n"
        "- Start with a '## <Section Title>' heading.\n"
        "- Use short paragraphs, bullet lists where helpful, and code fences for code.\n"
        "- Avoid fluff. Avoid marketing language.\n"
        "- If you include code, keep it focused on the bullet being addressed.\n"
    )
)
,
            HumanMessage(
                content=(f"""
                    Blog: {plan.blog_title}\n
                    Audience: {plan.audience}\n
                    Tone: {plan.tone}\n"
                    Topic: {topic}\n\n"
                    Section: {task.title}\n"
                    Section type: {task.section_type}\n"
                    Goal: {task.goal}\n"
                    Target words: {task.target_words}\n"
                    Bullets:{bullets_text}\n
                    """
                )
            ),
        ]
    ).content.strip()


    return {"section" : [section_md]}


def reducer(state: AgentState) -> dict:

    title = state["plan"].blog_title
    body = "\n\n".join(state['section'])

    final_md = f"{title}\n\n{body}"

    filename = "".join(c if c.isalnum() or c in (" ", "_", "-") else "" for c in title)
    filename = filename.strip().lower().replace(" ", "_") + ".md"
    Path(filename).write_text(final_md, encoding="utf-8")

    return {"final" : final_md}

graph = StateGraph(AgentState)

graph.add_node("orchestrator", orchestrator)
graph.add_node("worker", worker)
graph.add_node("reducer",reducer)

graph.add_edge(START , "orchestrator")
graph.add_conditional_edges("orchestrator", fanout, ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)


database_url = get_database_url()
conn = psycopg.connect(
    database_url,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(conn)
checkpointer.setup()

Blog = graph.compile(checkpointer=checkpointer)

config = {"configurable" : {"thread_id" : "jana"}}

response = Blog.invoke({"topic": "Write a blog on Student who passinate about AI Engineer", "sections": []}, config=config)

print("="*60)
print("BLOG")
print("="*60)

print(f"\n\nTopic : {response["topic"]}")
print(f"\n\nFinal : {response["final"]}")










    







