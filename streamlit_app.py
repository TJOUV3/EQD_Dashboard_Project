import streamlit as st

options = st.Page("Visualization\Options_strat.py", title="Options_Strat", icon=":material/history:", default=True)
stock_screeners = st.Page("Screeners\Stocks_Screener.py", title="Stocks_Screeners", icon=":material/search:")


pg = st.navigation(
        {
            "Screeners": [stock_screeners],
            "Visualization": [options],
        },
        position="sidebar")

pg.run()