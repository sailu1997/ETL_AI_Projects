import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import base64
import plotly.graph_objects as go 

st.set_page_config(layout="wide")

left, right = st.columns([3,7])

BACKEND_URL = "http://localhost:8000"

if "plots" not in st.session_state:
    st.session_state.plots = []

if "plotly_figures" not in st.session_state:
    st.session_state.plotly_figures = []

if "file_results" not in st.session_state:
    st.session_state.file_results = {}

with left:
    st.header("Data Analysis Assistant")
    uploaded_files = st.file_uploader("Upload a CSV file", type=["csv","xlsx"], accept_multiple_files=True)

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.file_results:
                files = {"csv_file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                response = requests.post(f"{BACKEND_URL}/insights/", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.file_results[uploaded_file.name] = result
                    st.success(f"File {uploaded_file.name} uploaded successfully!")
                else:
                    st.error(f"Error uploading file {uploaded_file.name}: {response.status_code}")
            result = st.session_state.file_results.get(uploaded_file.name, {})
            with st.expander(f"📄 {uploaded_file.name}"):
                st.markdown(result.get("insight", "No insight available."))

with right:
    st.header("Chat with your data")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show a single clean button for all files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            result = st.session_state.file_results.get(uploaded_file.name, {})
            health = result.get("health_summary", {})
            if health:
                with st.expander(f"🧼 Data Health Summary: {uploaded_file.name}", expanded=True):
                    st.json(health)
                if health.get("null_values") or health.get("duplicate_rows",0)>0 or health.get("formatting_issues"):
                    st.warning(f"Dataset {uploaded_file.name} has quality issues. Consider cleaning it.")
     
                    if st.button("✨ Clean Dataset(s) Using AI", key=f"clean_{uploaded_file.name}"):
                        with st.spinner("Cleaning dataset(s)…"):
                            for uploaded_file in uploaded_files:
                                #files = {"csv_file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                                file_id = st.session_state.file_results[uploaded_file.name].get("file_id")
                                print(file_id)
                                clean_response = requests.post(f"{BACKEND_URL}/clean_data/", json={"file_id": file_id})
                                if clean_response.status_code == 200:
                                    clean_data = clean_response.json()
                                    st.session_state[f'cleaned_data_{uploaded_file.name}'] = clean_data
                                    st.session_state[f'cleaned_sample_{uploaded_file.name}'] = pd.DataFrame(clean_data["cleaned_sample"])
                                    st.session_state[f'impact_metrics_{uploaded_file.name}'] = clean_data.get("impact_metrics", {})
                                    st.session_state[f'applied_code_{uploaded_file.name}'] = clean_data["applied_code"]
                                    st.success(f"Dataset {uploaded_file.name} cleaned successfully!")
                                else:
                                    st.error(f"Cleaning {uploaded_file.name} failed. Please try again.")

    # Show chat history (single chat window)
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)
                # Add plot handling here if needed
                if msg.get("plot_index") is not None:
                    idx = msg["plot_index"]
                    if 0 <= idx < len(st.session_state.plots):
                        st.pyplot(st.session_state.plots[idx], use_container_width=False)
                elif msg.get("plotly_index") is not None:
                    plotly_idx = msg["plotly_index"]
                    if 0 <= plotly_idx < len(st.session_state.plotly_figures):
                        fig_data = st.session_state.plotly_figures[plotly_idx]
                        plotly_fig = go.Figure(fig_data)
                        st.plotly_chart(plotly_fig, use_container_width=True)

    # Single chat input for all files
    if uploaded_files:  # Only show chat if at least one file is uploaded
        if user_q := st.chat_input("Ask about your data…"):
            st.session_state.messages.append({"role": "user", "content": user_q})
            with st.spinner("Working …"):
                # You can send all file names or just the first, depending on your backend
                file_ids = [result["file_id"] for result in st.session_state.file_results.values()]
                codegen_response = requests.post(
                    f"{BACKEND_URL}/codegeneration/", 
                    json={"query": user_q, "file_ids": file_ids})
                if codegen_response.status_code != 200:
                    st.error("Code generation failed.")
                    st.stop()
                try:
                    codegen_json = codegen_response.json()
                except ValueError:
                    st.error("Response from /codegeneration/ is not valid JSON.")
                    st.stop()

                code = codegen_json.get("code")
                should_plot_flag = codegen_json.get("should_plot")

                result_obj = requests.post(
                    f"{BACKEND_URL}/executionagent/",
                    json={"code": code, "should_plot": should_plot_flag, "file_ids": file_ids}
                )
                result_data = result_obj.json()
                
                response = requests.post(
                    f"{BACKEND_URL}/reasoningaagent/",
                    json={"query": user_q, "result": result_data, "file_ids": file_ids}
                )
                response_data = response.json()
                
                reasoning_txt = response_data.get("cleaned", "")
                reasoning_txt = reasoning_txt.replace("`", "")

                raw_thinking = response_data.get("thinking_content", "")
                reasoning_txt = response_data.get("cleaned", "")
                reasoning_txt = reasoning_txt.replace("`", "")

                # --- Plot handling ---
                is_plot = False
                plot_idx = None
                plotly_idx = None
                header = ""
                # Check for plotly figure
                if isinstance(result_data.get("result"), dict) and result_data["result"].get("type") == "plotly":
                    fig_data = result_data["result"]["figure_json"]
                    st.session_state.plotly_figures.append(fig_data)
                    plotly_idx = len(st.session_state.plotly_figures) - 1
                    header = "Here is the visualization you requested:"
                elif isinstance(result_data.get("result"), dict) and result_data["result"].get("image_base64"):
                    import io
                    import base64
                    from PIL import Image
                    image_data = base64.b64decode(result_data["result"]["image_base64"])
                    image = Image.open(io.BytesIO(image_data))
                    st.session_state.plots.append(image)
                    plot_idx = len(st.session_state.plots) - 1
                    header = "Here is the visualization you requested:"
                elif isinstance(result_data.get("result"), list):
                    result_list = result_data.get("result")
                    if result_list and isinstance(result_list[0], dict):
                        st.dataframe(pd.DataFrame(result_list))
                    else:
                        st.write(result_list)
                    header = f"Result: {len(result_data['result'])} items"
                elif isinstance(result_data.get("result"), dict):
                    st.json(result_data['result'])
                    header = "Result:"
                else:
                    header = f"Result: {result_data.get('result')}"
                
                # --- Build assistant message with code, reasoning, and plot index ---
                thinking_html = ""
                if raw_thinking:
                    thinking_html = (
                        '<details class="thinking">'
                        '<summary>🧠 Reasoning</summary>'
                        f'<pre>{raw_thinking}</pre>'
                        '</details>'
                    )
                code_html = (
                    '<details class="code">'
                    '<summary>View code</summary>'
                    '<pre><code class="language-python">'
                    f'{code}'
                    '</code></pre>'
                    '</details>'
                )
                assistant_msg = f"{header}\n\n{thinking_html}{reasoning_txt}\n\n{code_html}"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "plot_index": plot_idx,
                    "plotly_index": plotly_idx
                })
                st.rerun()
                
    else:
        st.info("Upload a CSV file to get started")

