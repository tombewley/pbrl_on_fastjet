"""

https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py
python3 -m streamlit run app.py

"""
import streamlit as st
import numpy as np
import pandas as pd

st.sidebar.write("Here's our first attempt at using data to create a table:")

left, mid, right = st.columns(3)

with left:
    st.text_input("Your name", key="name")

with right:
    st.write(pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]
    }))

    x = st.slider('x')  # 👈 this is a widget
    st.write(x, 'squared is', x * x)

chart_data = pd.DataFrame(
    np.array([[0,x,x],[x,0,x]]),
    columns=['a', 'b', 'c'])

st.line_chart(chart_data)

# You can access the value at any point with:
print(st.session_state.name)