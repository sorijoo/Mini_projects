import streamlit.components.v1 as components

file = open("example.html","r",encoding="UTF-8")
components.html(file.read(),height=800)
