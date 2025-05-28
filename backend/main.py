from backend.functions import EmailSummarizer
from backend.mock_responses import MOCK_EMAIL_THREAD

_summarizer_instance = EmailSummarizer()

def get_summarizer():
    return _summarizer_instance

def generate_summary(email_thread_text: str = MOCK_EMAIL_THREAD) -> str:
    if not email_thread_text.strip():
        return "No email content provided."

    summarizer = get_summarizer()
    return summarizer.summarize_thread(email_thread_text)
