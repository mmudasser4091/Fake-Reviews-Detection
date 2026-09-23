# Fake Review Detection System

An AI-powered desktop application built with Python and Tkinter to identify and classify genuine vs. fake online reviews using Natural Language Processing (NLP) and Machine Learning.

## Features

- **Tkinter GUI**: Modern, user-friendly graphical interface with real-time progress tracking.
- **NLP Preprocessing**: Tokenization, Stopword removal, and WordNet Lemmatization using NLTK.
- **Pre-trained ML Model**: TF-IDF vectorization paired with a trained classification model for fast inference.
- **Multiple File Formats**: Supports uploading batch reviews in `.txt`, `.csv`, and `.docx` (Microsoft Word) formats.
- **Automated Export**: Splits predictions automatically and exports them into separate `fake_reviews.csv` and `real_reviews.csv` files.
- **Multithreaded Execution**: File processing and classification run on a background thread to prevent UI freezing.

---

## Project Structure

```text
├── fake_review_detector.py   # Main Tkinter application and classification logic
├── fake_review_model.pkl      # Pre-trained Machine Learning model
├── tfidf_vectorizer.pkl       # Pre-trained TF-IDF vectorizer
├── fake_review_icon.ico       # Application icon
├── reviews.txt                # Sample reviews for testing
├── requirements.txt           # Project dependencies
├── .gitignore                 # Files excluded from git tracking
└── README.md                  # Project documentation
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/fake-reviews-detection.git
   cd fake-reviews-detection
   ```

2. **Create and activate a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

Launch the application by running:

```bash
python fake_review_detector.py
```

1. Click on **"Upload Review File"**.
2. Select a `.txt`, `.csv`, or `.docx` file containing reviews (you can test with the provided `reviews.txt`).
3. Monitor the progress bar as the model classifies the text.
4. Once completed, check the project folder for:
   - `fake_reviews.csv`
   - `real_reviews.csv`

---

## Technologies Used

- **Python 3**
- **Tkinter** (UI Framework)
- **Scikit-Learn** & **Joblib** (Machine Learning & Model Serialization)
- **NLTK** (Natural Language Processing)
- **Pandas** (Data Manipulation)
- **python-docx** (Word Document Parsing)
