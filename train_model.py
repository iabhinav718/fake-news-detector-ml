import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk

# Download required NLTK data
nltk.download('stopwords')

# Load dataset
print("Loading datasets...")
df_fake = pd.read_csv('Fake.csv')
df_real = pd.read_csv('True.csv')

# Add labels
df_fake['label'] = 0  # Fake
df_real['label'] = 1  # Real

# Combine datasets
df = pd.concat([df_fake, df_real], axis=0).reset_index(drop=True)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total articles: {len(df)}")
print(f"Fake articles: {len(df[df['label']==0])}")
print(f"Real articles: {len(df[df['label']==1])}")

# Data Preprocessing Function
def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Combine title and text
print("\nCombining title and text...")
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Apply text cleaning
print("Cleaning text data...")
df['content'] = df['content'].apply(clean_text)

# Remove empty content
df = df[df['content'].str.len() > 0].reset_index(drop=True)

print(f"Articles after cleaning: {len(df)}")

# Prepare features and labels
X = df['content']
y = df['label']

# Train-test split (80-20)
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Feature Extraction using TF-IDF
print("\nExtracting features using TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF feature shape: {X_train_tfidf.shape}")

# Train Logistic Regression Model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

print("Training complete!")

# Make predictions
print("\nMaking predictions on test set...")
y_pred = model.predict(X_test_tfidf)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"MODEL PERFORMANCE")
print(f"{'='*50}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nCorrectly classified Fake: {cm[0][0]}")
print(f"Misclassified Fake as Real: {cm[0][1]}")
print(f"Misclassified Real as Fake: {cm[1][0]}")
print(f"Correctly classified Real: {cm[1][1]}")

# Save model and vectorizer
print(f"\n{'='*50}")
print("Saving model and vectorizer...")
print(f"{'='*50}")

with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Saved: fake_news_model.pkl")

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print("✓ Saved: tfidf_vectorizer.pkl")

print(f"\n{'='*50}")
print("MODEL TRAINING COMPLETE!")
print(f"{'='*50}")
print("\nYou can now run the Streamlit app:")
print("python -m streamlit run app.py")
print(f"{'='*50}")