import pandas as pd
import re
from dateutil import parser
import logging
import quopri  # Decode quoted-printable email content
from transformers import BartTokenizer, BartForConditionalGeneration

# --- Email Utilities ---



logging.basicConfig(level=logging.WARNING)

def decode_quoted_printable(text):
    """Decode quoted-printable text (like =09, =20, =\n)."""
    if not isinstance(text, str):
        return ""
    try:
        decoded = quopri.decodestring(text).decode('utf-8', errors='ignore')
        return decoded
    except Exception as e:
        logging.warning(f"Failed to decode text: {e}")
        return text


def extract_sent_date(email_text):
    """
    Extract 'Sent' date from decoded email content.
    """
    if not isinstance(email_text, str):
        return pd.NaT

    decoded_text = decode_quoted_printable(email_text)

    patterns = [
        r"Sent:\s*(.+)",                       # Sent: Tuesday, Jan 29, 2002...
        r"Date:\s*(.+)",                       # Date: style
        r"On (.+), .* wrote:",                 # On Jan 29, John wrote:
    ]

    for pattern in patterns:
        match = re.search(pattern, decoded_text, re.IGNORECASE)
        if match:
            date_str = match.group(1).strip()
            try:
                return parser.parse(date_str, fuzzy=True)
            except Exception as e:
                logging.warning(f"Date parsing failed: {date_str} — {e}")
                return pd.NaT

    return pd.NaT


def get_top_emails_text(df, thread_id):
    """
    Get top N emails' texts in a thread, sorted by date.
    """
    df = df.copy()
    df['date'] = df['emails_text'].apply(extract_sent_date)
    filtered_sorted = df[df['thread_id'] == thread_id].sort_values('date', ascending=True)
    top_emails = filtered_sorted.head()
    # Return decoded and cleaned emails
    return [decode_quoted_printable(t) for t in top_emails['emails_text'].tolist()]


def get_full_thread_text(df, thread_id):
    """
    Concatenate top N emails into a formatted thread text.
    """
    emails = get_top_emails_text(df, thread_id)
    return "\n\n---\n\n".join(emails)


# --- Summarizer Class ---

class EmailSummarizer:
    def __init__(self, model_path: str = "facebook/bart-large-cnn"):
        print(f" Loading summarizer model from: {model_path}")
        try:
            self.tokenizer = BartTokenizer.from_pretrained(model_path)
            self.model = BartForConditionalGeneration.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}. Error: {e}")
        self.max_input_length = 1024

    def summarize_thread(self, email_thread: str, max_length=100, min_length=30) -> str:
        if len(email_thread) > 4000:
            email_thread = email_thread[:4000]

        inputs = self.tokenizer(
            email_thread,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True
        )

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


# Singleton for reuse
_summarizer_instance = None

def get_summarizer():
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = EmailSummarizer()
    return _summarizer_instance

def generate_summary(email_thread_text: str) -> str:
    if not email_thread_text.strip():
        return " No email content provided."

    try:
        summarizer = get_summarizer()
        summary = summarizer.summarize_thread(email_thread_text)
        if not summary.strip():
            return " Summarizer returned empty output."
        return summary
    except Exception as e:
        return f" Error during summarization: {e}"
