import streamlit as st
from streamlit_chat import message
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from streamlit_extras.add_vertical_space import add_vertical_space

"""
    Chatbot
"""

st.title("💬 Streamlit GPT")
choose = st.selectbox("List of models", ("Select", "DialoGPT", "Godel"))

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 HugChat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [HugChat](https://github.com/Soulter/hugging-chat-api)
    
    💡 Note: No API key required!
    ''')
    add_vertical_space(2)
    st.write('Made with ❤️ by [DSAI Group](https://github.com/quangbinh113/NLP.2022.2.Generative-Based-Chatbot)')
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"

def on_btn_click():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    del st.session_state.chat_history_ids
    del st.session_state.user_input
    st.session_state.is_clear = True

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

if "chat_history_ids" not in st.session_state:
    st.session_state["chat_history_ids"] = None

if "user_input" not in st.session_state:
    st.session_state["user_input"] = None

if "dialogpt" not in st.session_state:
    st.session_state["dialogpt"] = False

if "tokenizer" not in st.session_state:
    st.session_state["tokenizer"] = None

if "model" not in st.session_state:
    st.session_state["model"] = None

if "is_clear" not in st.session_state:
    st.session_state["is_clear"] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if choose == "DialoGPT" and not st.session_state.dialogpt:
    st.session_state.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    st.session_state.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to(device)
    st.session_state.dialogpt = True
    print('DialoGPT loaded!')

with st.form("chat_input", clear_on_submit = True):
    a, b = st.columns([4, 1])
    user_input = a.text_input(
            label = "Your message:",
            placeholder = "What would you like to say?",
            label_visibility = "collapsed",
        )
    if not st.session_state.is_clear:
        st.session_state.user_input = user_input
    b.form_submit_button("Send", use_container_width = True)
  
if user_input and choose == "Select":
    st.info("Please select a model to continue.")
    st.session_state.user_input = user_input

if st.session_state.user_input and choose != "Select":
    st.session_state.messages.append({"role": "user", "content": st.session_state.user_input})
    # msg = {"content": 'Hello', "role": "assistant"}
    print(st.session_state.messages)
    new_user_input_ids = st.session_state.tokenizer.encode(st.session_state.user_input + st.session_state.tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if len(st.session_state.messages) > 2 else new_user_input_ids
    print(bot_input_ids)
    st.session_state.chat_history_ids = st.session_state.model.generate(bot_input_ids, max_length=1000, pad_token_id = st.session_state.tokenizer.eos_token_id)
    msg = {"content": st.session_state.tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True), "role": "assistant"}
    st.session_state.messages.append(msg)

st.session_state.is_clear = False
    
for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user = msg["role"] == "user", key = i)

st.button("Clear message", on_click = on_btn_click)