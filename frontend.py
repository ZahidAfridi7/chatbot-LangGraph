import streamlit as st
import requests
import time

st.set_page_config(page_title="DexterzSol Assistant", page_icon="💬", layout="wide")

st.markdown(
    "<h2 style='text-align:center;'>💬 DexterzSol Technologies Assistant</h2>",
    unsafe_allow_html=True
)

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Display chat history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Stream text directly from backend
def get_chat_response(user_input):
    try:
        with requests.post(
            "http://localhost:8000/query/",
            data={"question": user_input, "thread_id": "1"},
            stream=True,
            timeout=60
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=32):
                if chunk:
                    yield chunk.decode("utf-8")
    except Exception as e:
        yield f"Error: {e}"

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    with st.chat_message('user'):
        st.markdown(f"**You:** {user_input}")

    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        displayed_text = ""
        first_chunk = True

        for text_piece in get_chat_response(user_input):
            if first_chunk:
                displayed_text = ""
                first_chunk = False

            # Smooth streaming + word-wise rendering
            words = text_piece.split(" ")
            for word in words:
                displayed_text += word + " "
                message_placeholder.markdown(displayed_text + "▌")
                time.sleep(0.03)

        message_placeholder.markdown(displayed_text.strip())
        st.session_state['message_history'].append(
            {'role': 'assistant', 'content': displayed_text.strip()}
        )

    # auto-scroll to latest message
    st.rerun()
