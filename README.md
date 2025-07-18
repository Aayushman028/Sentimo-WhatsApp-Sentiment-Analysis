# Sentimo: WhatsApp Sentiment Analysis Tool

## 🧾 Abstract

This project examines WhatsApp sentiment analysis conversations for identifying emotional patterns, behavior tendencies, and communication dynamics. Using Natural Language Processing (NLP) applications and the DistilBERT transformer model, messages were categorized as negative, neutral, or positive based on contextual understanding.

Message usage, user activity, emoji frequencies, and important words were examined based on TF-IDF vectorization. The output was presented using an interactive Streamlit dashboard with visualisations such as word clouds, heatmaps, and sentiment trend charts. These results provide a clearer picture of group interactions and variation in emotional response in the chat environment.

---

## 📊 Jupyter Notebook Visualizations

The file `Sentiment Analysis.ipynb` contains the complete exploratory data analysis and visualization pipeline. It was used to generate insights from the chat data, including:

- Word clouds
- Emoji frequency analysis
- Sentiment trend charts
- TF-IDF–based keyword extraction

This notebook played a key role in developing and refining the visual outputs later integrated into the Streamlit dashboard.

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

- Model Architecture: [**DistilBERT (Transformer)**](https://huggingface.co/docs/transformers/en/model_doc/distilbert)
- Frameworks: PyTorch, Hugging Face Transformers
- Dataset:
  - Twitter Sentiment Dataset (English)
  - Custom WhatsApp chat dataset with **100,000+ messages**
- Accuracy: **97.71%**
- Output Labels: Positive, Negative, Neutral

---

## 📦 Model File Downloads (External)

Due to GitHub’s 100MB file size limit, the following model files are **not included in the repository**. Please download them manually and place them in the appropriate folders:

🔗 **Google Drive:**
- [`distilbert_sentiment_model.pth`](https://drive.google.com/file/d/1D_Wa5HOKNbQj-UgbNaVMhFgpcFg8Pvbd/view?usp=sharing) → place in `Model Training/`
- [`pytorch_model.bin`](https://drive.google.com/file/d/1K7d-A5xqs4G_t5DmJnfo0DHO7HmfAhhV/view?usp=sharing) → place in `WA Analyzer/distilbert-sentiment-model/`

> ⚠️ These files are required for local execution of the sentiment predictor.

---

## 📄 Research Paper Publication

This project has been officially published in a peer-reviewed journal.

> 📘 **Paper Title:** *Sentimo: WhatsApp Sentiment Analysis Tools*
> 📅 **Year:** 2025  
> 🔗 [Read Full Paper (PDF)]((https://drive.google.com/file/d/1s_hjQ2B6tSVAoQb0KWff67Yz8rO7lFW7/view?usp=sharing))

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

## 👤 Contributors

| Name                   | GitHub Profile                                         | LinkedIn Profile                                      |
|------------------------|--------------------------------------------------------|-------------------------------------------------------|
| Harsh Tripathi         | [@Harsh-x-Tripathi](https://github.com/Harsh-x-Tripathi) | [Harsh on LinkedIn](https://www.linkedin.com/in/harsh-x-tripathi) |
| Mr. Mithlesh Kumar Yadav | *N/A*                                                  | [Mithlesh on LinkedIn](https://www.linkedin.com/me?trk=p_mwlite_feed-secondary_nav) |

---

> © 2025 Aayushman Sharma, Harsh Tripathi, Mr. Mithlesh Kumar Yadav – All rights reserved.
