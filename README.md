# Sentimo: WhatsApp Sentiment Analysis Tool

**Sentimo** is an AI-powered tool designed for **sentiment analysis of WhatsApp group chats**. It uses Natural Language Processing (NLP) techniques and transformer-based models to analyze chat messages and classify their sentiment as positive, negative, or neutral.

Built with Python, Hugging Face Transformers, and Streamlit, this project simplifies the process of analyzing informal text data from WhatsApp exports.

---

## 🚀 Features

- ✅ Automatically parses WhatsApp chat `.txt` exports  
- ✅ Preprocesses and cleans messages for model input  
- ✅ Fine-tuned DistilBERT sentiment classifier  
- ✅ Classifies messages as **Positive**, **Negative**, or **Neutral**  
- ✅ Visualizes overall sentiment trends  
- ✅ Easy-to-use interface via Streamlit  

---

## 📂 Project Structure

```
Sentimo Project/
├── WA Analyzer/
│   ├── app.py
│   ├── chat_parser.py
│   ├── model_predictor.py
│   ├── distilbert-sentiment-model/
│   │   ├── config.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   └── requirements.txt
├── Model Training/
│   ├── Dataset/
│   │   ├── twitter_training.csv
│   │   └── twitter_validation.csv
│   ├── Sem 8th Update Model Training Code.ipynb
│   ├── distilbert_sentiment_model/
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   └── predicted_results.csv
├── Sentiment Analysis.ipynb
├── app.py
├── chat_parser.py
├── README.md
```

---

## 🧠 Model Information

- Model Architecture: **DistilBERT (Transformer)**
- Frameworks: PyTorch, Hugging Face Transformers
- Dataset: Twitter Sentiment Dataset (English)
- Accuracy: **97.71%**
- Output Labels: Positive, Negative, Neutral

---

## 📦 Model File Downloads (External)

Due to GitHub’s 100MB file size limit, the following model files are **not included in the repository**. Please download them manually and place them in the appropriate folders:

🔗 **Google Drive:**
- [`distilbert_sentiment_model.pth`](https://drive.google.com/your-google-drive-link-here) → place in `Model Training/`
- [`pytorch_model.bin`](https://drive.google.com/your-google-drive-link-here) → place in `WA Analyzer/distilbert-sentiment-model/`

🔗 **Hugging Face (Optional Mirror):**
- [`Sentimo Model on Hugging Face`](https://huggingface.co/your-huggingface-link-here)

> ⚠️ These files are required for local execution of the sentiment predictor.

---

## 📄 Research Paper Publication

This project has been officially published in a peer-reviewed journal.

> 📘 **Paper Title:** *Sentimo: A Tool for Sentiment Analysis on WhatsApp Chat Data Using Transformer Models*  
> 🏛 **Published In:** [IJCRT – International Journal of Creative Research Thoughts](https://www.ijcrt.org/papers/your-paper-link-here)  
> 📅 **Year:** 2025  
> 🔗 [Read Full Paper (PDF)](https://www.ijcrt.org/papers/your-paper-link-here)

---

## ⚙️ How to Run Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Aayushman028/Sentimo-WhatsApp-Sentiment-Analysis.git
   cd Sentimo-WhatsApp-Sentiment-Analysis
   ```

2. **Install Dependencies**
   ```bash
   pip install -r WA\ Analyzer/requirements.txt
   ```

3. **Download Model Files**
   - See the "Model File Downloads" section above.
   - Place them in the correct folders.

4. **Run the Streamlit App**
   ```bash
   streamlit run WA\ Analyzer/app.py
   ```

---

## 👤 Contributor

| Name            | GitHub Profile                                      | LinkedIn Profile                                 |
|------------------|-----------------------------------------------------|--------------------------------------------------|
| Harsh Tripathi   | [@harshtripathi](##) | [Harsh on LinkedIn](##) |

---

> © 2025 Aayushman Sharma, Harsh Tripathi – All rights reserved.
