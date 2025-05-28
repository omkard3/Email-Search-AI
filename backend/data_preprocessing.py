import pandas as pd


def preprocess_email_data(details_csv_path, summaries_csv_path, output_csv_path):
    # Load datasets
    emails_df = pd.read_csv(details_csv_path)
    summaries_df = pd.read_csv(summaries_csv_path)

    # Sort emails by timestamp
    emails_df = emails_df.sort_values(by=["thread_id", "timestamp"])

    # Group by thread and concatenate email bodies
    grouped_emails = emails_df.groupby("thread_id")["body"].apply(lambda texts: " ".join(str(t) for t in texts)).reset_index()
    grouped_emails.columns = ["thread_id", "emails_text"]

    # Merge with summaries
    merged_df = pd.merge(grouped_emails, summaries_df, on="thread_id")

    # Save preprocessed data
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Processed dataset saved to: {output_csv_path}")

if __name__ == "__main__":
    preprocess_email_data(
        details_csv_path="data/email_thread_details.csv",
        summaries_csv_path="data/email_thread_summaries.csv",
        output_csv_path="data/processed_email_summarization_dataset.csv"
    )
