import streamlit as st
from loguru import logger
import sys
import hashlib
import pandas as pd
from st_pages import hide_pages, show_pages, Page, Section

logger.remove()
logger.add(sys.__stdout__)
logger.add("prompt.log", rotation="500 MB")


def hash_w2k(input_string):
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode("utf-8"))
    hashed_output = md5_hash.hexdigest()
    return hashed_output


# getw2k = GetW2K()

st.set_page_config(page_title="Newton", page_icon="👋")



# st.set_page_config(page_title=" Aircraft Log Spell Checker", page_icon="😇")
html_temp = """
<div style="background-color:brown;padding:10px">
<h2 style="color:white;text-align:center;"> Newton welcomes you! 👋 </h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)

# st.write("# Welcome to GPT Studio! 👋")

st.warning("It is strongly advised to refrain from using ChatGPT (chat.openai.com) \
           or any open Generative AI tool for work purposes. The data input provided \
           to these tools may be used by OpenAI and other parties to train their next \
           version model, which could potentially lead to a data leak. To ensure the \
           confidentiality and security of your data, it is recommended to use only \
           trusted and secure AI tools for work-related tasks.")


with st.form(key="authentication"):
    usr_input = st.text_input(label="Please input your w2k id")
    #pwd_input = st.text_input(label="Please input your w2k password", type="password")
    department = st.text_input(label="Your Department")
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    # w2k_check = getw2k.authenticate(user_name=usr_input, pwd=pwd_input)
    if usr_input == "" or department == "":
        st.write("You have input wrong/incomplete info")
        st.session_state.authenticated = False
    else:
        # st.write(f"Welcome {usr_input}")
        st.session_state.authenticated = True
        st.session_state.w2k = usr_input
        st.write("You can now toggle to any application.")
        # st.session_state.prompt = ""
        st.session_state.w2k_hash = hash_w2k(usr_input)
        st.session_state.dept = department
        logger.info(f"w2k:{usr_input}|")
#        switch_page("Ask_Newton_anything")
