import openai
import streamlit as st
import pandas as pd
import os
import subprocess
import sys
from streamlit_chat import message
from langchain.agents import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler 
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from tenacity import retry, wait_exponential, stop_after_attempt, RetryError
from PIL import Image
import base64
import streamlit_shadcn_ui as ui




# Check if tabulate is installed and install it if it's not
try:
    import tabulate
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])

# Set API key from Streamlit secrets
api_key = st.secrets["openai"]["api_key"]
openai.api_key = api_key

# Cache data loading function
@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

# Function to clear previous submissions
def clear_submit():
    st.session_state["submit"] = False

# Initialize session state
if "submit" not in st.session_state:
    st.session_state["submit"] = False

# Set Streamlit page configuration
st.set_page_config(page_title="DataLens", page_icon=":speech_balloon:", layout="wide")

st.markdown(
    """
    <style>
    .st-emotion-cache-1mi2ry5 {
        display: flex;
        -webkit-box-pack: justify;
        justify-content: space-between;
        -webkit-box-align: start;
        align-items: start;
        padding-bottom: 0rem;
        padding-top: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown("""
        <style>
               .block-container {
                    padding-top: 4rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)



# Define supported file formats
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

def get_image_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_image_base64("logo.png")

st.sidebar.markdown(f"""
    <div style="display: flex; align-items: center; margin: 0; padding: 0; height: 60px;">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="height: 50px; width: auto; margin: 0;">
        <span style="font-size:32px; margin-left:10px; line-height: 50px;">DataLens</span>
    </div>
""", unsafe_allow_html=True)


st.sidebar.markdown("---")

# File uploader widget
uploaded_file = st.sidebar.file_uploader(
    "Upload a Data file",
    help="Various file formats are supported",
    on_change=clear_submit,
)
st.markdown("</br>", unsafe_allow_html=True)

# Load data if a file is uploaded
if uploaded_file:
    df = load_data(uploaded_file)

# Initialize messages in session state
if "messages" not in st.session_state or st.sidebar.button("Clear history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Retry logic for API requests
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
def make_api_request(prompt):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=api_key, streaming=True)
    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )
    return pandas_df_agent.run(prompt, callbacks=[st_cb])

# Handle user input and generate response
if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Handle API response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = make_api_request(prompt)
            st.write(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})
        except RetryError:
            st.error("Rate limit exceeded. Please try again later.")
        except openai.APIError as e:
            st.error(f"An API error occurred: {e}")
        except openai.OpenAIError as e:
            st.error(f"An error occurred: {e}")



st.sidebar.markdown("""
<style>
    .badge-container {
        display: flex;
        justify-content: center;
    }
    .badge-container a {
        margin: 0 3px;  
    }
</style>
<hr>
<div class="badge-container">
    <a href="https://www.linkedin.com/in/amit-yadav-674a9722b">
        <img src="https://img.shields.io/badge/-LinkedIn-306EA8?style=flat&logo=Linkedin&logoColor=white" alt="LinkedIn">
    </a>
    <a href="https://github.com/Amit2465">
        <img src="https://img.shields.io/badge/-GitHub-2F2F2F?style=flat&logo=github&logoColor=white" alt="GitHub">
    </a>
    <a href="mailto:your.amityadav23461@gmail.com.com">
        <img src="https://img.shields.io/badge/-Email-D14836?style=flat&logo=gmail&logoColor=white" alt="Email">
    </a>
</div>
</br>
<div style="text-align: center;">© 2024 Amit Yadav</div>
""", unsafe_allow_html=True)   