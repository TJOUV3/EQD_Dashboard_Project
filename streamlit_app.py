import streamlit as st

options = st.Page("Visualization/Options_strat.py", title="Options_Strat", icon=":material/history:",default=True)
options_vantage = st.Page("Visualization/Options.py", title="Options", icon=":material/history:")
volatility_surface = st.Page("Visualization/Volatility_surface.py", title="Volatility Surface", icon=":material/history:")
heston = st.Page("Visualization/Heston.py", title="Heston", icon=":material/history:")
test_heston = st.Page("Visualization/Prepa_Heston.py", title="Prepa Heston", icon=":material/history:")

stock_screeners = st.Page("Screeners/Stocks_Screener.py", title="Stocks_Screeners", icon=":material/search:")
heatmap = st.Page("Screeners/Heatmap.py", title="Heatmap", icon=":material/square:")
tasks = st.Page("Tasks/TodoList.py", title="TodoList", icon=":material/square:")


pg = st.navigation(
        {
            "Screeners": [stock_screeners,heatmap],
            "Visualization": [options,options_vantage,volatility_surface,heston,test_heston],
            "Next Tasks" : [tasks]
        },
        position="sidebar")

pg.run()