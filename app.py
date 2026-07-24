import streamlit as st
from predict import predict_toxicity

st.set_page_config(
    page_title="Toxicity Detection",
    page_icon="🛡️"
)

st.title("🛡️ Real-Time Toxicity Detection")

st.write("Enter a comment below to check whether it is toxic.")

comment = st.text_area("Enter Comment")

if st.button("Predict"):

    if comment.strip() == "":
        st.warning("Please enter a comment.")
    else:

        prediction = predict_toxicity(comment)

        score = float(prediction[0][0])

        st.write(f"### Confidence Score: {score:.2%}")

        if score >= 0.5:
            st.error("🚨 Toxic Comment Detected")
        else:
            st.success("✅ Non-Toxic Comment")