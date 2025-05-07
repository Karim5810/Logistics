import streamlit as st

def show_counter():
    """
    Display a simple counter example with increment functionality
    """
    st.title('Simple Counter Example')
    
    # Initialize counter in session state if it doesn't exist
    if 'counter' not in st.session_state:
        st.session_state.counter = 0
    
    # Display the current count
    st.write(f"Current count: {st.session_state.counter}")
    
    # Create three columns for buttons
    col1, col2, col3 = st.columns(3)
    
    # Add buttons to control the counter
    with col1:
        if st.button("Decrement", use_container_width=True):
            st.session_state.counter -= 1
            
    with col2:
        if st.button("Reset", use_container_width=True):
            st.session_state.counter = 0
            
    with col3:
        if st.button("Increment", use_container_width=True):
            st.session_state.counter += 1
    
    # Display the value in different formats
    st.metric("Counter Value", st.session_state.counter)


if __name__ == "__main__":
    show_counter() 