import streamlit as st

st.write('Filtre actif pour options US')
st.write('Compléter S&P 500 Stocks Screener tab avec des nouvelles colonnes')
st.write("ajouter données dans Options strat à la place de X et Y (graph prix ....), restyliser l'ajout d'options avec les bonnes couleurs, en X mettre la valeur de Delta, Y Gamma etc etc ")
st.write('repenser le graph option price')
st.write('penser au hedging, faire pour chaque grecque, possibilité de rentrer les niveaux de grecques souhaités et proposition de montage')

st.subheader('Hedging')

st.write('On choisit 1 produit casi pur par grecque, ex delta == stock')
st.write('Vega == Straddle')
st.write('Gamma == opt atm + short term')
st.write('Theta == impossible put > call')
st.write("Rho == maturité élevée ")