import streamlit as st
import requests
import time

st.header("DexterzSol Technologies Assistant - 🤖")

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Display chat history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# ✅ Stream text in small increments for smoother effect
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
                    text_piece = chunk.decode("utf-8")
                    yield text_piece  # yield small piece directly
    except Exception as e:
        yield f"Error: {e}"

user_input = st.chat_input("Type your message here:")

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        displayed_text = ""
        first_chunk = True

        for text_piece in get_chat_response(user_input):
            if first_chunk:
                displayed_text = ""
                first_chunk = False

            displayed_text += text_piece
            message_placeholder.markdown(displayed_text + "▌")  # typing cursor
            time.sleep(0.03)  # simulate live typing speed

        # remove cursor at end
        message_placeholder.markdown(displayed_text)
        st.session_state['message_history'].append({'role': 'assistant', 'content': displayed_text})
