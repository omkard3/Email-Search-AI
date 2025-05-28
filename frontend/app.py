import streamlit as st
import pandas as pd
import sys
import os

# Backend path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.functions import get_top_emails_text, get_full_thread_text
from backend.functions import generate_summary


def main():
    st.set_page_config(page_title="Email Thread Summarizer", layout="wide")
    st.title("Email Thread Viewer & Summarizer")

    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("data/processed_email_summarization_dataset.csv")
        except FileNotFoundError:
            st.error("Dataset file not found at data/processed_email_summarization_dataset.csv")
            return pd.DataFrame()

    df = load_data()
    if df.empty:
        return

    thread_ids = df['thread_id'].unique()
    selected_thread = st.selectbox("Select Thread ID", thread_ids)

    col_spacer, col1, col2 = st.columns([6, 1, 1])

    with col1:
        show_btn = st.button("Show Emails")

    with col2:
        summarize_btn = st.button("Summarize Thread")

    if show_btn:
        emails = get_top_emails_text(df, selected_thread)
        if not emails:
            st.warning("No emails found or date extraction failed.")
        else:
            st.subheader("Emails in Thread")
            for i, email_text in enumerate(emails, start=1):
                st.markdown(f"---\n**Emails:**\n```\n{email_text}\n```")

    if summarize_btn:
        full_thread_text = get_full_thread_text(df, selected_thread)
        if not full_thread_text.strip():
            st.warning("No content available to summarize.")
        else:
            st.subheader("Preview: Email Thread Text")
            st.text_area("Email Thread Input (first 1000 characters)", full_thread_text[:1000], height=200)

            with st.spinner("Summarizing..."):
                summary = generate_summary(full_thread_text)

            st.subheader("Thread Summary")
            st.markdown(f"> {summary}")


if __name__ == "__main__":
    main()
