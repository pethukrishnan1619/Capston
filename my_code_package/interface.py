"""
Gradio interface for the multi-agent RAG system.

This module provides a function to create a Gradio Blocks demo that exposes
an interface for users to ask questions and see answers, citations, and
reasoning steps.
"""
import json
import gradio as gr

from .orchestrator import handle_user_query


def interface_function(user_query: str):
    """Callback to process a user query and return answer, citations and react trace."""
    res = handle_user_query(user_query)
    answer = res.get("answer", "")
    citations = res.get("citations", [])
    if citations:
        citation_text = "\n".join([f"- {c['source']} (page {c['page']})" for c in citations])
    else:
        citation_text = "No citations available."
    react_trace = json.dumps(res.get("react_steps", []), indent=2)
    return answer, citation_text, react_trace


# def launch_demo():
#     """Launch the Gradio app."""
#     with gr.Blocks(title="Multi-Agent RAG + Tools (No API Key)") as demo:
#         gr.Markdown("## 🧠 Multi-Agent RAG System")
#         with gr.Row():
#             with gr.Column(scale=2):
#                 user_query = gr.Textbox(
#                     label="Ask your question",
#                     placeholder="Examples: weather in Chennai | calculate (10+20)/2 | Applications of AI",
#                     lines=3,
#                 )
#                 ask_btn = gr.Button("Ask")
#                 answer_box = gr.Textbox(label="Answer", lines=5, interactive=False)
#                 citation_box = gr.Textbox(label="Citations", lines=5, interactive=False)
#             with gr.Column(scale=1):
#                 react_box = gr.Code(
#                     label="ReAct Trace (Reason → Act → Observe)",
#                     language="json",
#                     lines=24,
#                 )
#         ask_btn.click(
#             fn=interface_function,
#             inputs=user_query,
#             outputs=[answer_box, citation_box, react_box],
#         )
#     demo.launch()

import gradio as gr
import os

def launch_demo():
    """Launch the Gradio app."""
    with gr.Blocks(title="Multi-Agent RAG + Tools (No API Key)") as demo:
        gr.Markdown("## 🧠 Multi-Agent RAG System")
        with gr.Row():
            with gr.Column(scale=2):
                user_query = gr.Textbox(
                    label="Ask your question",
                    placeholder="Examples: weather in Chennai | calculate (10+20)/2 | Applications of AI",
                    lines=3,
                )
                ask_btn = gr.Button("Ask")
                answer_box = gr.Textbox(label="Answer", lines=5, interactive=False)
                citation_box = gr.Textbox(label="Citations", lines=5, interactive=False)
            with gr.Column(scale=1):
                react_box = gr.Code(
                    label="ReAct Trace (Reason → Act → Observe)",
                    language="json",
                    lines=24,
                )
        ask_btn.click(
            fn=interface_function,
            inputs=user_query,
            outputs=[answer_box, citation_box, react_box],
        )
    return demo   # ✅ return the demo object

# --- Launch outside the function ---
port = int(os.environ.get("PORT", "10000"))  # Render default is 10000
demo = launch_demo()
demo.launch(server_name="0.0.0.0", server_port=port)
