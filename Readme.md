#  EmailSearchAI

**EmailSearchAI** is an intelligent email thread summarizer that uses advanced NLP techniques to generate concise summaries of long email conversations. It provides a Streamlit-based frontend interface, a HuggingFace transformer-powered backend, and tools for preprocessing and training custom summarization models.

---

##  Features

- Summarizes email threads using BART transformer models.
- Preprocesses raw email and summary datasets.
- Fine-tunes transformer models on custom datasets.
- Visualizes and explores threads through a Streamlit interface.
- Offers mock responses and standalone summary functions.

---

##  Project Structure
```text
EmailSearchAI/
├── backend/
│ ├── summarizer_model/ # Trained model output directory
│ ├── data_preprocessing.py # Preprocess raw email datasets
│ ├── functions.py # Utilities for decoding, summarizing
│ ├── main.py # Summarization runner using summarizer class
│ ├── mock_responses.py # Mock input email thread for testing
│ └── train_model.py # Script to fine-tune the summarizer model
├── data/
│ ├── email_thread_details.csv # Raw email thread data
│ ├── email_thread_summaries.csv # Corresponding manual summaries
│ └── processed_email_summarization_dataset.csv # Preprocessed dataset
├── frontend/
│ └── app.py # Streamlit frontend for viewing & summarizing
├── .env # Environment configuration (optional)
├── config.toml # Config file for managing model paths/settings
├── docker-compose.yml # Docker Compose setup (if applicable)
├── Dockerfile # Dockerfile to containerize the app
└── requirements.txt # Python dependencies
```

##  How It Works
- Preprocessing: data_preprocessing.py combines raw email threads and their summaries into a clean dataset.
- Summarizer: The EmailSummarizer class loads a pretrained (or custom) BART model to generate summaries.
- Frontend: app.py allows users to explore emails by thread ID and generate summaries with one click.

## Technologies Used
- Python 3.9+
- Hugging Face Transformers (BART)
- Streamlit
- Pandas
- Datasets (by HuggingFace)
- PyTorch
- Docker (optional)

