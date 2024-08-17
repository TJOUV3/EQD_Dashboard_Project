import streamlit as st

stock_screeners = st.Page("Screeners/Stocks_Screeners.py", title="Stocks_Screeners", icon=":material/search:")
options = st.Page("Visualization/Options_strat.py", title="Options_Strat", icon=":material/history:")

pg = st.navigation(
        {
            "Screeners": [stock_screeners],
            "Visualization": [options],
        })
pg.run()