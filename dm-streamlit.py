import streamlit as st

st.title("My Streamlit App")

st.write("This is my first Streamlit app")

name = st.text_input("What is your name?", "")

if name:
    st.write(f"Hello, {name}!")
