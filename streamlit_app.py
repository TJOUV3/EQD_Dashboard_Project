import streamlit as st

options = st.Page("Visualization\Options_strat.py", title="Options_Strat", icon=":material/history:", default=False)
stock_screeners = st.Page("Screeners\Stocks_Screener.py", title="Stocks_Screeners", icon=":material/search:", default=True)


pg = st.navigation(
        {
            "Screeners": [stock_screeners],
            "Visualization": [options],
        },
        position="sidebar")

pg.run()