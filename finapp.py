import pandas as pd
import streamlit as st
import numpy as np
import altair as alt

st.title("Team5 Final Project ITC")

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(dataframe.style.highlight_max(axis=0))

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

chart_data

c = alt.Chart(dataframe, title='measure of different elements over time').mark_line().encode(
     x='col 1', y="col 2", color='parameter')

st.altair_chart(c, use_container_width=True)