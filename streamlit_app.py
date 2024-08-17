import streamlit as st

stock_screeners = st.Page("pages/Screeners/Stocks_Screener.py", title="Stocks_Screeners", icon=":material/search:")
options = st.Page("pages/Visualization/Options_strat.py", title="Options_Strat", icon=":material/history:", default=True)

pg = st.navigation(
        {
            "Screeners": [stock_screeners],
            "Visualization": [options],
        })
pg.run()