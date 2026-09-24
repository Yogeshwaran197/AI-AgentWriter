from pathlib import Path
from datetime import date, timedelta

import operator
import re
import uuid
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import os
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from groq import BadRequestError

import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

load_dotenv()

OUT_DIR = Path(__file__).resolve().parents[1]


def safe_filename(title: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title.strip())
    return name + ".md"


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
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    max_retries=6,  # retries with backoff on transient errors (incl. Groq's 429 rate_limit_exceeded)
                    # at the client level, so `llm` stays a real ChatGroq (with_structured_output etc.
                    # still work) instead of being wrapped in a generic RunnableRetry
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
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class RouterDecision(BaseModel):
    needs_research: bool
    # NOTE: values now use underscores to match ROUTER_SYSTEM / ORCH_SYSTEM / WORKER_SYSTEM prompt text.
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem]


class ImageSpec(BaseModel):
    filename: str = Field(..., description="save under images / eg qkv.flow.png")
    alt: str
    caption : str = Field(..., description="Caption about the image")
    anchor : str = Field(..., description="A short phrase that appears verbatim in the blog markdown preview. The image is inserted right after the paragraph containing it. Use an exact heading or sentence fragment.")
    prompt : str = Field(..., description="prompt to create the image")
    size : Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list)


class AgentState(TypedDict):

    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]  # was List[Plan] — a single Plan object is stored, not a list

    # workers/ reducer
    sections: Annotated[List[tuple], operator.add]
    merged_md : str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str



ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES."""


def retry_on_bad_json(runnable):
    return runnable.with_retry(
        retry_if_exception_type=(BadRequestError,),
        stop_after_attempt=3,
    )


def router(state: AgentState) -> dict:

    topic = state["topic"]
    router_llm = retry_on_bad_json(llm.with_structured_output(RouterDecision))  # renamed to avoid shadowing the node fn name

    decision = router_llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic : {topic}"),
    ])

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }


def router_next(state: AgentState) -> str:
    # was "reserach" (typo) — didn't match the "research" key in the conditional-edge map, so the
    # graph would raise a KeyError/invalid-node error whenever needs_research was True.
    return "research" if state["needs_research"] else "orchestrator"


SNIPPET_CHAR_LIMIT = 400  # keep each result small so the extractor call stays under the TPM limit


def tavily_search(query: str, max_results: int = 4) -> list[dict]:
    # was: def tavily_search(queries: list, max_results): TavilySearchResults(queries, max_results)
    #      -> wrong constructor signature (positional list+int isn't valid), and
    #      tool.invoke({"queries": queries}) used the wrong key/shape for a single query.
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []

    for r in results:
        raw_snippet = r.get("snippet") or r.get("content") or ""  # was "snippent" (typo)
        normalized.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "snippet": raw_snippet[:SNIPPET_CHAR_LIMIT],  # Tavily's "content" can be huge; truncate it
            "published_at": r.get("published_at") or r.get("published_date"),
            "source": r.get("source"),
        })

    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""


def research_node(state: AgentState) -> dict:

    queries = state.get("queries", []) or []
    top_results = 4  # fewer results per query keeps the extractor prompt small

    raw_results: list[dict] = []

    for q in queries:  # was `for q in queries():` — queries is a list, not callable, so this raised TypeError
        raw_results.extend(tavily_search(q, top_results))

    if not raw_results:
        return {"evidence": []}

    # Cap the total number of results fed into a single structured-output call so the request
    # stays under the model's tokens-per-minute limit (this is what caused the 413 rate_limit_exceeded).
    MAX_ITEMS_PER_EXTRACT_CALL = 15
    raw_results = raw_results[:MAX_ITEMS_PER_EXTRACT_CALL]

    extractor = retry_on_bad_json(llm.with_structured_output(EvidencePack))
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Raw results: {raw_results}"),
    ])

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}


ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120–550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""


def orchestrator(state: AgentState) -> dict:

    planner = retry_on_bad_json(llm.with_structured_output(Plan))

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=f"""
            "topic": {state["topic"]},
            "mode": {mode},
            "evidence(only use when new claims)": {[e.model_dump() for e in evidence][:16]}
        """),
    ])

    return {"plan": plan}


def fanout(state: AgentState):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""


def worker_node(payload: dict) -> dict:

    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}


#Reducer SubGraph

def merge_content(state : AgentState) -> dict:

    plan = state["plan"]
    order_session = [md for _, md in sorted(state['sections'], key = lambda x : x[0])]
    body = "\n\n".join(order_session).strip()
    merged_md = f"{plan.blog_kind}\n{body}"

    return {"merged_md" : merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- The anchor MUST be a verbatim (exact) phrase taken from the markdown preview below — an exact heading or a short sentence. Never invent, shorten, or paraphrase the anchor.
- If no images needed, return images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan containing only the images list.
"""

def decide_images(state: AgentState) -> dict:

    planner = retry_on_bad_json(llm.with_structured_output(GlobalImagePlan))
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    # Send only a compact preview so the request stays under Groq's TPM limit.
    # Lines stay intact so the model's anchors match the full markdown verbatim.
    preview_lines = []
    used = 0
    for line in merged_md.splitlines():
        if used + len(line) > 3000 and preview_lines:
            break
        preview_lines.append(line)
        used += len(line) + 1
    preview = "\n".join(preview_lines)

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Markdown preview (anchors must match this verbatim):\n\n"
                    f"{preview}"
                )
            ),
        ]
    )

    return {
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _insert_after(md: str, anchor: str, block: str) -> str:
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if anchor and anchor in line:
            after = i + 1
            while after < len(lines) and not lines[after].strip():
                after += 1
            lines.insert(after, "")
            lines.insert(after + 1, block)
            return "\n".join(lines)
    return md + "\n\n" + block



def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def generate_and_place_images(state: AgentState) -> dict:

    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    output_md = OUT_DIR / safe_filename(plan.blog_title)

    # If no images requested, just write merged markdown
    if not image_specs:
        output_md.write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = OUT_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        filename = spec["filename"]
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = _insert_after(md, spec.get("anchor", ""), prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = _insert_after(md, spec.get("anchor", ""), img_md)

    output_md.write_text(md, encoding="utf-8")
    return {"final": md}


r =  StateGraph(AgentState)

r.add_node("merger_content", merge_content)
r.add_node("decide_images",  decide_images)
r.add_node("generate_and_place_images", generate_and_place_images)

r.add_edge(START , "merger_content")
r.add_edge("merger_content" , "decide_images")
r.add_edge("decide_images", "generate_and_place_images")
r.add_edge("generate_and_place_images", END)

reducer = r.compile()

g = StateGraph(AgentState)
g.add_node("router", router)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer)

g.add_edge(START, "router")
g.add_conditional_edges("router", router_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)


database_url = get_database_url()
conn = psycopg.connect(
    database_url,
    row_factory=dict_row,
    autocommit =True,
)

checkpointer = PostgresSaver(conn)
checkpointer.setup()

app = g.compile(checkpointer=checkpointer)

config = {
        "configurable": {
            "thread_id": f"run_{uuid.uuid4().hex[:12]}"
        }
}


def run(topic: str):
    out = app.invoke(
        {
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
        },
        config=config,
    )

    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    result = run("neural networks in deep learning")
    out_file = OUT_DIR / safe_filename(result["plan"].blog_title)
    print(f"Complete. Blog written to: {out_file}")
    print(f"Length: {len(result['final'])} chars, images: {len(result.get('image_specs', []))}")