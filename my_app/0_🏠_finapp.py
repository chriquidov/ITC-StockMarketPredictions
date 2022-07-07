import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu

st.set_page_config(layout="centered", page_icon="📈", page_title="Stock prediction app")
st.title("📈Stock prediction Application")

st.markdown('# Some infromation ')
st.markdown('In order for the user to have a great experience using our app, here are a few instructions and infromations.'
            'Using this app, the user will be able to:')
st.markdown('* Test the different strategies we used for predictions on a stock of yout choice')
with st.expander("Tell me more about it!"):
    st.write("""
        We trained different models to predict stock prices for the next period, Visiting the demo page you 
        will be able to input the ticker of your choice and choose a period, we will run for you a simulation 
        of the value of portfolio when following the different prediction models. 
    """)

