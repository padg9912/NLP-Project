import streamlit as st
from full_flow import full_pipeline

st.title("Time-based Fact-Checking and Fake News Detection")

st.write("""
Enter a claim and (optionally) a date. The system will retrieve evidence, classify the claim, and provide an LLM-generated summary and explanation.
""")

claim = st.text_input("Enter your claim:")
date = st.text_input("Enter the date of the claim (MM/DD/YYYY):", value="12/8/2014")

if st.button("Check Claim"):
    if not claim or not date:
        st.warning("Please enter both a claim and a date.")
    else:
        with st.spinner("Running fact-checking pipeline..."):
            result = full_pipeline(claim, date)
        st.subheader("BERT Classifier Verdict")
        st.write(f"**Verdict:** {result['classification']['prediction']}")
        st.write(f"**Confidence:** {result['classification']['confidence']}")
        st.subheader("LLM Evidence Synthesis & Explanation (All Models)")
        for model_name, output in result['llm_outputs'].items():
            st.markdown(f"### {model_name.capitalize()} Output")
            st.write(output)
        st.subheader("Supporting Evidence")
        for i, doc in enumerate(result['supporting_docs']):
            st.markdown(f"**[{i+1}]** {doc.page_content}")
            st.markdown(f"- Date: {doc.metadata.get('time_stamp')}")
            st.markdown(f"- Truth: {doc.metadata.get('truthfulness')}") 