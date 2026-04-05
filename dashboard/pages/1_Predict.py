import streamlit as st

st.title("Predict")
st.markdown("Input text and view side-by-side predictions from both pipelines.")

text_input = st.text_area("Enter tweet text:")
if st.button("Predict") and text_input:
    st.info("Prediction logic coming soon — wire up src/predict.py here.")
