import streamlit as st

st.title("My Streamlit App")

st.write("This is my first Streamlit app")

name = st.text_input("What is your bank account number?", "")

if name:
    st.write(f"Thank you your money HCNY, {name}!")
