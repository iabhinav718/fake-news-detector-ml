# 📰 Fake News Detector ML

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

An automated fake news detection system using NLP and Machine Learning. Achieves 99% accuracy with a simple web interface.

---

## 🎯 Features

- ✅ 99% accuracy using Logistic Regression + TF-IDF
- ⚡ Real-time predictions via Streamlit web interface
- 🧠 NLP-powered text analysis
- 📊 Confidence scores for predictions

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/fake-news-detector-ml.git
cd fake-news-detector-ml

# Install dependencies
pip install pandas numpy scikit-learn==1.3.2 nltk streamlit

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"
```

### Download Dataset

1. Get the dataset from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Place `Fake.csv` and `True.csv` in project root

### Train Model

```bash
python train_model.py
```

Training takes 5-8 minutes and creates:
- `fake_news_model.pkl`
- `tfidf_vectorizer.pkl`

### Run Application

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📁 Project Structure

```
fake-news-detector-ml/
├── app.py                   # Streamlit web app
├── train_model.py           # Model training script
├── requirements.txt         # Dependencies
├── Fake.csv                 # Dataset (download)
├── True.csv                 # Dataset (download)
├── fake_news_model.pkl      # Trained model (generated)
└── tfidf_vectorizer.pkl     # Vectorizer (generated)
```

---

## 🔬 How It Works

### Pipeline

```
Input Text → Clean Text → TF-IDF Vectorization → Logistic Regression → Prediction
```

### Text Preprocessing
- Lowercase conversion
- URL removal
- Punctuation removal
- Whitespace normalization

### Feature Extraction
- **TF-IDF**: Converts text to 5000 numerical features
- **N-grams**: Uses unigrams and bigrams
- **Stop words**: Filters common English words

---

## 📊 Performance

| Metric | Score |
|--------|-------|
| Accuracy | 99.12% |
| Precision | 99% |
| Recall | 99% |
| F1-Score | 99% |

**Confusion Matrix:**
```
              Predicted
            Fake    Real
Actual Fake 4639      56
       Real   30    4253
```

---

## 🧪 Test Examples

### Fake News (99%+ confidence)
```
"BREAKING: Scientists confirm earth will stop rotating tomorrow! 
Government hiding the truth from citizens!"
```

### Real News (99%+ confidence)
```
"WASHINGTON (Reuters) - The U.S. Federal Reserve signaled that 
interest rates could remain high as inflation moderates."
```

---

## ⚠️ Limitations

- **Dataset bias**: Trained on specific writing styles
- **Short text**: Generic statements may be misclassified
- **English only**: No multilingual support
- **Linguistic patterns**: Detects style, not factual accuracy
- **Context**: Cannot verify actual truth of claims

---

## 🔮 Future Enhancements

- [ ] Deep learning models (BERT, LSTM)
- [ ] Multilingual support
- [ ] Fact-checking API integration
- [ ] Browser extension
- [ ] Explainability (LIME/SHAP)

---

## 🤝 Contributing

Contributions welcome! Fork the repo and submit a pull request.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---



## 🙏 Acknowledgments

- **Dataset**: [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Tools**: scikit-learn, NLTK, Streamlit, Pandas
- **Support**: Open Source Community

---

## 📝 Disclaimer

This tool detects linguistic patterns, **NOT factual accuracy**. Always verify information through multiple reliable sources. Use responsibly with critical thinking.

---

<div align="center">

Made with ❤️ 

⭐ Star this repo if you found it helpful!

</div>
