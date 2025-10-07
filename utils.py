from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from prompt import user_goal_prompt
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, Any, Callable, List, Dict, Tuple
import asyncio
import os
import json
import re
from datetime import datetime
from urllib.parse import quote_plus
import base64

try:
    from fpdf import FPDF  
except Exception:
    FPDF = None

cfg = RunnableConfig(recursion_limit=100)

def initialize_model(google_api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key
    )

async def setup_agent_with_tools(
    google_api_key: str,
    youtube_pipedream_url: str,
    drive_pipedream_url: Optional[str] = None,
    notion_pipedream_url: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Any:
    """
    Set up the agent with YouTube (mandatory) and optional Drive or Notion tools.
    """
    try:
        if progress_callback:
            progress_callback("Setting up agent with tools... ✅")
        
       
        tools_config = {
            "youtube": {
                "url": youtube_pipedream_url,
                "transport": "streamable_http"
            }
        }

        
        if drive_pipedream_url:
            tools_config["drive"] = {
                "url": drive_pipedream_url,
                "transport": "streamable_http"
            }
            if progress_callback:
                progress_callback("Added Google Drive integration... ✅")

        
        if notion_pipedream_url:
            tools_config["notion"] = {
                "url": notion_pipedream_url,
                "transport": "streamable_http"
            }
            if progress_callback:
                progress_callback("Added Notion integration... ✅")

        if progress_callback:
            progress_callback("Initializing MCP client... ✅")
        
        mcp_client = MultiServerMCPClient(tools_config)
        
        if progress_callback:
            progress_callback("Getting available tools... ✅")
        
        tools = await mcp_client.get_tools()
        
        if progress_callback:
            progress_callback("Creating AI agent... ✅")
        
        mcp_orch_model = initialize_model(google_api_key)
        agent = create_react_agent(mcp_orch_model, tools)
        
        if progress_callback:
            progress_callback("Setup complete! Starting to generate learning path... ✅")
        
        return agent
    except Exception as e:
        print(f"Error in setup_agent_with_tools: {str(e)}")
        raise

def run_agent_sync(
    google_api_key: str,
    youtube_pipedream_url: str,
    drive_pipedream_url: Optional[str] = None,
    notion_pipedream_url: Optional[str] = None,
    user_goal: str = "",
    progress_callback: Optional[Callable[[str], None]] = None
) -> dict:
    """
    Synchronous wrapper for running the agent.
    """
    async def _run():
        try:
            agent = await setup_agent_with_tools(
                google_api_key=google_api_key,
                youtube_pipedream_url=youtube_pipedream_url,
                drive_pipedream_url=drive_pipedream_url,
                notion_pipedream_url=notion_pipedream_url,
                progress_callback=progress_callback
            )
            
            
            learning_path_prompt = "User Goal: " + user_goal + "\n" + user_goal_prompt
            
            if progress_callback:
                progress_callback("Generating your learning path...")
            
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=learning_path_prompt)]},
                config=cfg
            )
            
            if progress_callback:
                progress_callback("Learning path generation complete!")
            
            return result
        except Exception as e:
            print(f"Error in _run: {str(e)}")
            raise

    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


# -----------------------------
# Learning Path Post-processing
# -----------------------------

def concatenate_messages(agent_result: dict) -> str:
    """
    Concatenate text content from agent result into a single markdown string.
    """
    if not agent_result:
        return ""
    messages = agent_result.get("messages") or []
    parts: List[str] = []
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            parts.append(content)
    return "\n\n".join(parts).strip()


def sanitize_learning_path_text(text: str) -> str:
    """
    Remove agent/tool traces and keep only user-facing learning path content.
    Strategy: if Day-sections exist, retain title/header lines and Day blocks; drop
    common agent patterns like Thought/Action/Observation and code fences.
    """
    if not text:
        return ""
    # Remove triple backtick blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove ToolException/error JSON blocks
    text = re.sub(r"Error:\s*ToolException\([\s\S]*?\)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\{\s*\"data\"\s*:\s*\{[\s\S]*?\}\s*\}\s*", "", text)
    # Remove common agent trace lines
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            lines.append("")
            continue
        if re.match(r"^(Thought|Action|Observation)\s*:?", s, flags=re.IGNORECASE):
            continue
        if s.startswith("{") and s.endswith("}"):
            # likely tool json
            continue
        lines.append(ln)
    cleaned = "\n".join(lines).strip()

    # If Day sections found, keep from first Day onward
    m = re.search(r"(^|\n)\s*Day\s+\d+\s*[:\-]", cleaned, flags=re.IGNORECASE)
    if m:
        start = m.start() if m.start() > 0 else 0
        # Try to keep any preceding title line if present
        head = cleaned[:start].strip().splitlines()
        title = "\n".join(head[-2:]) if head else ""
        body = cleaned[start:].strip()
        return (title + "\n\n" + body).strip() if title else body
    return cleaned


def extract_days_and_questions(text: str) -> List[Dict[str, Any]]:
    """
    Parse the generated learning path text and extract per-day sections with optional
    practice questions. This uses forgiving regex to handle typical formats.
    Returns a list of dicts: {"day_title", "topic", "video_link", "questions": [..]}.
    """
    if not text:
        return []

    # Ensure each Day starts on its own line to simplify splitting
    canon = re.sub(r"\s+(Day\s+\d+\s*[:\-])", r"\n\1", text, flags=re.IGNORECASE)
    # Split by Day headers like 'Day 1:' or 'Day 1 -'
    day_blocks = re.split(r"\n(?=\s*Day\s+\d+\s*[:\-])", canon, flags=re.IGNORECASE)
    results: List[Dict[str, Any]] = []
    for block in day_blocks:
        block = block.strip()
        if not block or not re.match(r"^Day\s+\d+\s*[:\-]", block, flags=re.IGNORECASE):
            continue
        title_match = re.match(r"^(Day\s+\d+\s*[:\-]\s*)(.*)$", block, flags=re.IGNORECASE)
        day_title = title_match.group(1).strip() if title_match else "Day"
        rest = title_match.group(2).strip() if title_match else block

        # Topic can be inline or on its own line
        topic_match = re.search(r"Topic\s*:\s*([^\n]+)", rest, flags=re.IGNORECASE)
        topic = topic_match.group(1).strip() if topic_match else ""

        # YouTube link: capture first URL in the block if explicit label missing or multiple links exist
        link_match = re.search(r"YouTube\s*Link(?:\s*\d+)?\s*:?\s*(https?://\S+)", rest, flags=re.IGNORECASE)
        if link_match:
            video_link = link_match.group(1).strip()
        else:
            any_url = re.search(r"https?://\S+", rest)
            video_link = any_url.group(0) if any_url else ""

        # Fix placeholder YouTube links
        if "VIDEO_ID_HERE" in video_link:
            # Replace with a Google search link for the topic
            if topic:
                video_link = f"https://www.youtube.com/results?search_query={quote_plus(topic + ' tutorial')}"
            else:
                video_link = "https://www.youtube.com/"

        # Practice Questions section
        q_section_match = re.search(
            r"Practice\s*Questions\s*\(\s*10\s*\)\s*:?\s*(.*)$",
            rest,
            flags=re.IGNORECASE | re.DOTALL,
        )
        questions: List[str] = []
        if q_section_match:
            q_text = q_section_match.group(1).strip()
            # Split by numbered/bulleted lines
            for line in q_text.splitlines():
                line = line.strip(" -\t")
                line = re.sub(r"^\d+\.?\s*", "", line)
                if line:
                    questions.append(line)
        # Fallback to Google tutorial search if no video link
        if not video_link and topic:
            video_link = f"https://www.youtube.com/results?search_query={quote_plus(topic + ' tutorial')}"
        elif not video_link:
            video_link = "https://www.youtube.com/"
            
        results.append({
            "day_title": day_title,
            "topic": topic,
            "video_link": video_link,
            "questions": questions[:10] if questions else []
        })
    # If primary parsing succeeded
    if results:
        return results

    # ------------- Fallback parser -------------
    # Locate day headers by indices and slice blocks more permissively
    blocks: List[Tuple[int, int]] = []
    header_re = re.compile(r"(?im)^\s*Day\s+\d+\s*[:\-]?.*$")
    matches = list(header_re.finditer(text))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append((start, end))
    fallback: List[Dict[str, Any]] = []
    for start, end in blocks:
        block = text[start:end].strip()
        header_line = block.splitlines()[0] if block else "Day"
        # Derive a day title from the header line
        day_title = header_line.split("Topic:")[0].strip()
        # Topic
        topic_match = re.search(r"Topic\s*:\s*([^\n]+)", block, flags=re.IGNORECASE)
        topic = topic_match.group(1).strip() if topic_match else ""
        # First URL in block as video link
        link_match = re.search(r"https?://\S+", block)
        video_link = link_match.group(0) if link_match else ""
        
        # Fix placeholder YouTube links
        if "VIDEO_ID_HERE" in video_link:
            # Replace with a Google search link for the topic
            if topic:
                video_link = f"https://www.youtube.com/results?search_query={quote_plus(topic + ' tutorial')}"
            else:
                video_link = "https://www.youtube.com/"

        # Questions section
        q_match = re.search(r"Practice\s*Questions\s*\(\s*10\s*\)\s*:?\s*(.*)$", block, flags=re.IGNORECASE | re.DOTALL)
        questions: List[str] = []
        if q_match:
            q_text = q_match.group(1)
            for line in q_text.splitlines():
                line = line.strip(" -\t")
                line = re.sub(r"^\d+\.?\s*", "", line)
                if line:
                    questions.append(line)
        # Fallback to Google tutorial search if no video link
        if not video_link and topic:
            video_link = f"https://www.youtube.com/results?search_query={quote_plus(topic + ' tutorial')}"
        elif not video_link:
            video_link = "https://www.youtube.com/"
            
        fallback.append({
            "day_title": day_title or "Day",
            "topic": topic,
            "video_link": video_link,
            "questions": questions[:10]
        })
    return fallback


# -----------------------------
# Progress Tracking
# -----------------------------

def load_progress(user_goal: str) -> Dict[str, Any]:
    """Load progress for a specific user goal"""
    _ensure_data_dir()
    progress_file = os.path.join(DATA_DIR, f"progress_{hash(user_goal)}.json")
    if not os.path.isfile(progress_file):
        return {}
    try:
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_progress(user_goal: str, progress: Dict[str, Any]) -> None:
    """Save progress for a specific user goal"""
    _ensure_data_dir()
    progress_file = os.path.join(DATA_DIR, f"progress_{hash(user_goal)}.json")
    try:
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # Silently fail to avoid interrupting the main flow


# -----------------------------
# Local Persistence (History / Progress)
# -----------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")


def _ensure_data_dir() -> None:
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def load_history() -> List[Dict[str, Any]]:
    _ensure_data_dir()
    if not os.path.isfile(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history_record(record: Dict[str, Any]) -> None:
    _ensure_data_dir()
    history = load_history()
    history.append(record)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# -----------------------------
# PDF Export
# -----------------------------

def export_to_pdf(title: str, content: str, output_dir: str) -> Tuple[bool, str]:
    """
    Export text content to a PDF file using FPDF. Returns (ok, path).
    """
    if FPDF is None:
        return False, "FPDF not available. Install 'fpdf' package."
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    safe_title = re.sub(r"[^a-zA-Z0-9_\-]+", "_", title or "learning_path")
    filename = f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    out_path = os.path.join(output_dir, filename)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, txt=title or "Learning Path", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    # Simple text flow
    for line in content.splitlines():
        if not line.strip():
            pdf.ln(4)
            continue
        pdf.multi_cell(0, 6, txt=line)
    try:
        pdf.output(out_path)
        return True, out_path
    except Exception as e:
        return False, str(e)


# -----------------------------
# Drive Upload via MCP Agent
# -----------------------------

async def _create_drive_agent(google_api_key: str, drive_pipedream_url: str, progress_callback: Optional[Callable[[str], None]] = None):
    if progress_callback:
        progress_callback("Initializing Drive MCP client... ✅")
    tools_config = {
        "drive": {
            "url": drive_pipedream_url,
            "transport": "streamable_http",
        }
    }
    mcp_client = MultiServerMCPClient(tools_config)
    tools = await mcp_client.get_tools()
    if progress_callback:
        progress_callback("Creating Drive-only agent... ✅")
    model = initialize_model(google_api_key)
    agent = create_react_agent(model, tools)
    return agent


def upload_pdf_to_drive_via_agent(
    google_api_key: str,
    drive_pipedream_url: str,
    pdf_path: str,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[bool, str]:
    """
    Best-effort upload of a local PDF to Google Drive using the Drive MCP tool via an agent.
    Returns (ok, info) where info is file id/url or error message.
    """
    if not os.path.isfile(pdf_path):
        return False, f"File not found: {pdf_path}"

    file_name = os.path.basename(pdf_path)
    try:
        with open(pdf_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return False, f"Failed to read PDF: {e}"

    async def _run():
        try:
            agent = await _create_drive_agent(google_api_key, drive_pipedream_url, progress_callback)
            if progress_callback:
                progress_callback("Uploading PDF to Drive via MCP... ✅")
            prompt = (
                "Task: Upload the provided PDF to Google Drive using your Drive tool.\n"
                f"Desired filename: {file_name}\n"
                "MIME type: application/pdf\n"
                "Content (base64, no data URL):\n"
                f"{b64}\n\n"
                "Return ONLY a short confirmation line with the created file id and a view link if available."
            )
            result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]}, config=cfg)
            # Extract concise message content
            msg_list = result.get("messages") if isinstance(result, dict) else None
            if msg_list:
                for m in msg_list:
                    content = getattr(m, "content", None)
                    if isinstance(content, str) and content.strip():
                        return True, content.strip()
            return True, "Uploaded to Drive (details not available)."
        except Exception as e:
            return False, f"Drive upload failed: {e}"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()
