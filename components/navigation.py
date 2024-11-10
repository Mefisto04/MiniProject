import streamlit as st

def navigate_to(page_name):
    st.session_state.page = page_name
