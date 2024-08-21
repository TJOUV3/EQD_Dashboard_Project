import streamlit as st

options = st.Page("Visualization\Options_strat.py", title="Options_Strat", icon=":material/history:",default=True)
options_vantage = st.Page("Visualization\Options.py", title="Options", icon=":material/history:")

stock_screeners = st.Page("Screeners\Stocks_Screener.py", title="Stocks_Screeners", icon=":material/search:")
heatmap = st.Page("Screeners\Heatmap.py", title="Heatmap", icon=":material/square:")
tasks = st.Page("Tasks\TodoList.py", title="TodoList", icon=":material/square:")


pg = st.navigation(
        {
            "Screeners": [stock_screeners,heatmap],
            "Visualization": [options,options_vantage],
            "Next Tasks" : [tasks]
        },
        position="sidebar")

pg.run()