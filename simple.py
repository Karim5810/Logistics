import streamlit as st

# Set page config
st.set_page_config(
    page_title="Simple Streamlit App",
    page_icon="🚀",
    layout="wide"
)

# Force initialize session state
st.session_state["user_name"] = "Guest"
st.session_state["counter"] = 0

# App header
st.title("Simple Streamlit App")
st.subheader(f"Welcome, {st.session_state['user_name']}!")

# Counter example
if st.button("Increment Counter"):
    st.session_state["counter"] += 1

st.write(f"Current counter value: {st.session_state['counter']}")

# Show session state
st.subheader("Current Session State:")
st.write(st.session_state) 