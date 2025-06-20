import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import io

from chat_parser import load_chat
from model_predictor import SentimentService

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")

# Sidebar: load model & upload chat
model_name = st.sidebar.text_input("Model folder", "distilbert-sentiment-model")
if st.sidebar.button("Load Model"):
    st.session_state.service = SentimentService(model_name)
    st.sidebar.success("Model loaded ✓")

uploaded = st.sidebar.file_uploader("Upload WhatsApp .txt", type="txt")
if uploaded:
    text_stream = io.StringIO(uploaded.getvalue().decode("utf-8"))
    st.session_state.df = load_chat(text_stream)
    st.sidebar.success(f"Parsed {len(st.session_state.df)} messages")

# Main UI
if st.session_state.get("service") and st.session_state.get("df") is not None:
    df = st.session_state.df.copy()
    service = st.session_state.service

    if 'sentiment' not in df:
        df['sentiment'] = [r['label'] for r in service.predict(df['message'].tolist())]
        st.session_state.df = df

    tabs = st.tabs(["Preview", "Table", "Metrics", "WordCloud", "Trends"])
    with tabs[0]:
        st.dataframe(df[['date','time','contact','message']].head(10))
    with tabs[1]:
        st.dataframe(df[['date','time','contact','message','sentiment']], height=500)
    with tabs[2]:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(df))
        col2.metric("Contacts", df['contact'].nunique())
        col3.metric("Emojis", df['message'].str.count(r'[\U0001F600-\U0001FAFF]').sum())
        fig = px.histogram(df, x='sentiment')
        st.plotly_chart(fig, use_container_width=True)
    with tabs[3]:
        wc = WordCloud(stopwords=STOPWORDS, background_color="white") \
             .generate(" ".join(df['message']))
        st.image(wc.to_array(), use_column_width=True)
    with tabs[4]:
        trend = df.groupby([df['date'], df['sentiment']]) \
                  .size().reset_index(name='count')
        fig2 = px.line(trend, x='date', y='count', color='sentiment')
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("▶️ Load the model and upload a chat to begin.")
