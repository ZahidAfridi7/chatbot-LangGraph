import streamlit as st
from chatbot_with_out_memory import get_chat_response  # Import the backend function
from langchain_core.messages import HumanMessage

# Streamlit session state setup
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Display chat history (previous conversation)
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# User input for chat
user_input = st.chat_input("Type your message here:")

# Process the user's message if provided
if user_input:
    # Append the user's message to session state
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    
    # Display the user's message
    with st.chat_message('user'):
        st.text(user_input)

    # Get the chatbot's response from the backend
    try:
        ai_message = get_chat_response(user_input)  # Call the backend function to get response
        
        # Append the assistant's response to session state
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
        
        # Display the assistant's response
        with st.chat_message('assistant'):
            st.text(ai_message)

    except Exception as e:
        st.error(f"Error: {e}")
