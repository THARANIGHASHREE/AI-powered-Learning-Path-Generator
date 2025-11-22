import streamlit as st
from utils import run_agent_sync, concatenate_messages, sanitize_learning_path_text, extract_days_and_questions, save_history_record, load_history, export_to_pdf, upload_pdf_to_drive_via_agent, load_progress, save_progress
import os

st.set_page_config(page_title="MCP POC", page_icon="🤖", layout="wide")

st.title("Model Context Protocol(MCP) - Learning Path Generator")

# Initialize session state for progress
if 'current_step' not in st.session_state:
    st.session_state.current_step = ""
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'last_section' not in st.session_state:
    st.session_state.last_section = ""
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False

# Sidebar for API and URL configuration
st.sidebar.header("Configuration")
debug_mode = st.sidebar.checkbox("Debug mode", value=False, help="Show detailed error tracebacks")

# API Key input (prefill if provided by user)
default_google_key = ""
google_api_key = st.sidebar.text_input("Google API Key", value=default_google_key, type="password")

st.sidebar.subheader("Pipedream URLs")
# Allow user to paste provided Composio/MCP URLs; do not hardcode in code for security
_youtube_input = st.sidebar.text_input("YouTube URL (Required)", 
    placeholder="Enter your Pipedream YouTube URL")
youtube_pipedream_url = (_youtube_input or "").strip()

# Secondary tool selection
secondary_tool = st.sidebar.radio(
    "Select Secondary Tool:",
    ["Drive", "Notion"]
)

# Secondary tool URL input
if secondary_tool == "Drive":
    _drive_input = st.sidebar.text_input("Drive URL", 
        placeholder="Enter your Pipedream Drive URL")
    drive_pipedream_url = (_drive_input or "").strip()
    notion_pipedream_url = None
else:
    _notion_input = st.sidebar.text_input("Notion URL", 
        placeholder="Enter your Pipedream Notion URL")
    notion_pipedream_url = (_notion_input or "").strip()
    drive_pipedream_url = None

# Quick guide before goal input
st.info("""
**Quick Guide:**
1. Enter your Google API key and YouTube URL (required)
2. Select and configure your secondary tool (Drive or Notion)
3. Enter a clear learning goal, for example:
    - "I want to learn python basics in 3 days"
    - "I want to learn data science basics in 10 days"
""")

# Main content area
st.header("Enter Your Goal")
user_goal = st.text_input("Enter your learning goal:",
                        help="Describe what you want to learn, and we'll generate a structured path using YouTube content and your selected tool.")

# Progress area
progress_container = st.container()
progress_bar = st.empty()

def update_progress(message: str):
    """Update progress in the Streamlit UI"""
    st.session_state.current_step = message
    
    # Determine section and update progress
    if "Setting up agent with tools" in message:
        section = "Setup"
        st.session_state.progress = 0.1
    elif "Added Google Drive integration" in message or "Added Notion integration" in message:
        section = "Integration"
        st.session_state.progress = 0.2
    elif "Creating AI agent" in message:
        section = "Setup"
        st.session_state.progress = 0.3
    elif "Generating your learning path" in message:
        section = "Generation"
        st.session_state.progress = 0.5
    elif "Learning path generation complete" in message:
        section = "Complete"
        st.session_state.progress = 1.0
        st.session_state.is_generating = False
    else:
        section = st.session_state.last_section or "Progress"
    
    st.session_state.last_section = section
    
    # Show progress bar
    progress_bar.progress(st.session_state.progress)
    
    # Update progress container with current status
    with progress_container:
        # Show section header if it changed
        if section != st.session_state.last_section and section != "Complete":
            st.write(f"**{section}**")
        
        # Show message with tick for completed steps
        if message == "Learning path generation complete!":
            st.success("All steps completed! 🎉")
        else:
            prefix = "✓" if st.session_state.progress >= 0.5 else "→"
            st.write(f"{prefix} {message}")

col_gen, col_pdf = st.columns([2,1])

# Generate Learning Path button
if col_gen.button("Generate Learning Path", type="primary", disabled=st.session_state.is_generating):
    if not google_api_key:
        st.error("Please enter your Google API key in the sidebar.")
    elif not youtube_pipedream_url:
        st.error("YouTube URL is required. Please enter your Pipedream YouTube URL in the sidebar.")
    elif (secondary_tool == "Drive" and not drive_pipedream_url) or (secondary_tool == "Notion" and not notion_pipedream_url):
        st.error(f"Please enter your Pipedream {secondary_tool} URL in the sidebar.")
    elif not user_goal:
        st.warning("Please enter your learning goal.")
    elif len(google_api_key.strip()) < 20:
        st.error("The Google API key looks invalid. Please paste a valid key.")
    elif not (youtube_pipedream_url.lower().startswith("http")):
        st.error("The YouTube URL must start with http/https.")
    elif secondary_tool == "Drive" and drive_pipedream_url and not drive_pipedream_url.lower().startswith("http"):
        st.error("The Drive URL must start with http/https.")
    else:
        try:
            # Set generating flag
            st.session_state.is_generating = True
            
            # Reset progress
            st.session_state.current_step = ""
            st.session_state.progress = 0
            st.session_state.last_section = ""
            
            result = run_agent_sync(
                google_api_key=google_api_key,
                youtube_pipedream_url=youtube_pipedream_url,
                drive_pipedream_url=drive_pipedream_url,
                notion_pipedream_url=notion_pipedream_url,
                user_goal=user_goal,
                progress_callback=update_progress
            )
            
            # Display results
            st.header("Your Learning Path")
            full_text = concatenate_messages(result)
            clean_text = sanitize_learning_path_text(full_text)
            if clean_text:
                st.markdown(clean_text)
                # keep in session for PDF export
                st.session_state["last_generated_text"] = clean_text

                # Show day-wise readable plan with tracking
                day_items = extract_days_and_questions(clean_text)
                if day_items:
                    st.subheader("Day-wise Plan and Practice Questions")
                    # Load existing progress
                    progress_data = load_progress(user_goal)
                    
                    for idx, item in enumerate(day_items, start=1):
                        day_key = f"day_{idx}"
                        is_completed = progress_data.get(day_key, {}).get("completed", False)
                        
                        with st.expander(f"{item.get('day_title', f'Day {idx}')} - {item.get('topic','')}", expanded=is_completed):
                            link = item.get('video_link','')
                            if link:
                                st.write(f"YouTube Link: {link}")
                            # Per-day completion
                            completed = st.checkbox("Mark as completed", value=is_completed, key=f"gen_done_day_{idx}")
                            
                            # Save progress when checkbox changes
                            if completed != is_completed:
                                progress_data[day_key] = {
                                    "completed": completed,
                                    "title": item.get('day_title', f'Day {idx}'),
                                    "topic": item.get('topic', '')
                                }
                                save_progress(user_goal, progress_data)
                            
                            # Questions
                            qs = item.get('questions', [])
                            if qs:
                                st.markdown("**Practice Questions (10)**")
                                for q_i, q in enumerate(qs, start=1):
                                    st.write(f"{q_i}. {q}")

                # Extract day sections and questions
                # Save history entry
                save_history_record({
                    "timestamp": st.session_state.get("generated_at") or None,
                    "goal": user_goal,
                    "tool": secondary_tool,
                    "has_drive": bool(drive_pipedream_url),
                    "has_notion": bool(notion_pipedream_url),
                    "content": clean_text
                })

            else:
                st.error("No results were generated. Please try again.")
                st.session_state.is_generating = False
        except Exception as e:
            import traceback
            err_msg = f"An error occurred: {str(e)}"
            st.error(err_msg)
            st.error("Please check your API keys and URLs, and try again.")
            if debug_mode:
                st.exception(e)
                st.code(traceback.format_exc())
            st.session_state.is_generating = False

# Export to PDF (and auto-upload to Drive if configured)
if col_pdf.button("Export to PDF"):
    try:
        full_text = st.session_state.get("last_generated_text")
    except Exception:
        full_text = None
    if not full_text:
        st.warning("Generate a learning path first.")
    else:
        ok, out = export_to_pdf(title="Learning Path", content=full_text, output_dir=os.path.join(os.path.dirname(__file__), "exports"))
        if ok:
            st.success(f"PDF exported: {out}")
            # Auto-upload if Drive URL provided
            if secondary_tool == "Drive" and drive_pipedream_url:
                st.info("Uploading to Google Drive...")
                success, info = upload_pdf_to_drive_via_agent(
                    google_api_key=google_api_key,
                    drive_pipedream_url=drive_pipedream_url,
                    pdf_path=out,
                    progress_callback=update_progress,
                )
                if success:
                    st.success(f"Uploaded to Drive: {info}")
                else:
                    st.error(info)
        else:
            st.error(out)

# Manual Upload button (if PDF already exported)
if secondary_tool == "Drive" and st.button("Upload Last PDF to Drive"):
    exports_dir = os.path.join(os.path.dirname(__file__), "exports")
    if not os.path.isdir(exports_dir):
        st.warning("No exported PDFs found.")
    else:
        # pick the latest PDF
        pdfs = [os.path.join(exports_dir, f) for f in os.listdir(exports_dir) if f.lower().endswith(".pdf")]
        if not pdfs:
            st.warning("No exported PDFs found.")
        else:
            latest = sorted(pdfs, key=os.path.getmtime)[-1]
            success, info = upload_pdf_to_drive_via_agent(
                google_api_key=google_api_key,
                drive_pipedream_url=drive_pipedream_url,
                pdf_path=latest,
                progress_callback=update_progress,
            )
            if success:
                st.success(f"Uploaded to Drive: {info}")
            else:
                st.error(info)

# Analytics & History
st.header("Analytics & History")
history = load_history()
if history:
    st.write(f"Total paths generated: {len(history)}")
    # Simple analytics: tool usage counts
    drive_count = sum(1 for h in history if h.get("has_drive"))
    notion_count = sum(1 for h in history if h.get("has_notion"))
    st.write(f"Used Drive: {drive_count} | Used Notion: {notion_count}")

    with st.expander("View History"):
        for i, rec in enumerate(reversed(history), start=1):
            st.markdown(f"**#{i} Goal:** {rec.get('goal','')}")
            st.caption(f"Tool: {rec.get('tool','')} | Drive: {rec.get('has_drive')} | Notion: {rec.get('has_notion')}")
            # Minimal, readable content only
            st.text_area("Learning Path", value=rec.get("content",""), height=200, key=f"hist_content_{i}")
            # Day-wise tracking with questions
            items = extract_days_and_questions(rec.get("content",""))
            if items:
                # Load progress for this goal
                progress_data = load_progress(rec.get("goal", ""))
                
                with st.expander("Track Progress by Day and Review Questions"):
                    for idx, item in enumerate(items, start=1):
                        day_key = f"day_{idx}"
                        is_completed = progress_data.get(day_key, {}).get("completed", False)
                        
                        st.markdown(f"**{item.get('day_title', f'Day {idx}')} - {item.get('topic','')}**")
                        if is_completed:
                            st.markdown(":white_check_mark: Completed")
                        link = item.get('video_link','')
                        if link:
                            st.write(f"YouTube Link: {link}")
                        # Checkbox to mark as completed
                        completed = st.checkbox("Mark as completed", value=is_completed, key=f"hist_done_{i}_{idx}")
                        
                        # Save progress when checkbox changes
                        if completed != is_completed:
                            progress_data[day_key] = {
                                "completed": completed,
                                "title": item.get('day_title', f'Day {idx}'),
                                "topic": item.get('topic', '')
                            }
                            save_progress(rec.get("goal", ""), progress_data)
                            
                        qs = item.get('questions', [])
                        if qs:
                            st.markdown("Practice Questions (10)")
                            for q_i, q in enumerate(qs, start=1):
                                st.write(f"{q_i}. {q}")
else:
    st.info("No history yet. Generate a learning path to see analytics.")
