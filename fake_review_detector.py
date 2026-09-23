import nltk
nltk.data.path.append("./nltk_data")


import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import threading
import time
import os
import docx

# Load the model and vectorizer
import os
import sys
import joblib

# Get correct path for .exe and during Python execution
base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(base_path, "fake_review_model.pkl")
vectorizer_path = os.path.join(base_path, "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)



# Preprocessing function
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def process_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.readlines()
        df = pd.DataFrame(data, columns=["review"])
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        df = df.rename(columns={df.columns[0]: "review"})
    elif ext == ".docx":
        doc = docx.Document(file_path)
        data = [p.text for p in doc.paragraphs if p.text.strip() != ""]
        df = pd.DataFrame(data, columns=["review"])
    else:
        raise ValueError("Unsupported file format")
    return df

def classify_reviews(df, progress_callback):
    df["clean_text"] = ""
    results = []
    total = len(df)
    for idx, row in df.iterrows():
        clean = preprocess_text(str(row["review"]))
        vector = vectorizer.transform([clean])
        pred = model.predict(vector)[0]
        results.append((row["review"], "Fake" if pred == 1 else "Real"))
        progress_callback((idx + 1) / total * 100)
    return results

def save_output(results):
    df_out = pd.DataFrame(results, columns=["Review", "Prediction"])
    fake_df = df_out[df_out["Prediction"] == "Fake"]
    real_df = df_out[df_out["Prediction"] == "Real"]
    fake_df.to_csv("fake_reviews.csv", index=False)
    real_df.to_csv("real_reviews.csv", index=False)

def run_classification(file_path, progress_var, status_label):
    try:
        df = process_file(file_path)
        status_label.config(text="Processing reviews...")

        def update_progress(value):
            progress_var.set(value)
            progress_bar.update()

        results = classify_reviews(df, update_progress)
        save_output(results)
        status_label.config(text="Classification complete! Output saved.")
        messagebox.showinfo("Done", "Reviews classified. Files saved as fake_reviews.csv and real_reviews.csv")
    except Exception as e:
        status_label.config(text="Error occurred.")
        messagebox.showerror("Error", str(e))

def start_thread():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("Word Documents", "*.docx")])
    if file_path:
        progress_var.set(0)
        status_label.config(text="Starting classification...")
        thread = threading.Thread(target=run_classification, args=(file_path, progress_var, status_label))
        thread.start()

# UI Setup
root = tk.Tk()
root.title("Fake Review Detector")
root.geometry("500x300")
root.resizable(False, False)

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Segoe UI", 12), padding=6)
style.configure("TLabel", font=("Segoe UI", 11))

frame = ttk.Frame(root, padding=20)
frame.pack(fill=tk.BOTH, expand=True)

title = ttk.Label(frame, text="Fake Review Detection System", font=("Segoe UI", 16, "bold"))
title.pack(pady=(0, 10))

upload_btn = ttk.Button(frame, text="Upload Review File", command=start_thread)
upload_btn.pack(pady=10)

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(frame, length=400, variable=progress_var)
progress_bar.pack(pady=10)

status_label = ttk.Label(frame, text="Awaiting file upload...")
status_label.pack(pady=(10, 0))

root.mainloop()
