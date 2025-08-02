import streamlit as st
import pandas as pd
from scrape_reddit import scrape_reddit_posts
from scrape_twitter import scrape_tweets
from sentiment import analyze_sentiment
from data_processor import extract_and_aggregate
from data_sources.stock_price import get_live_price
from data_sources.news_scraper import get_news_articles
from data_sources.sec_scraper import get_sec_filings
from features.trend_similarity import compare_with_historical
from features.model import predict_signal

st.set_page_config(page_title="Meme Stock Predictor AI", page_icon="🤖")
st.title("🤖 Meme Stock Swing Predictor")

st.markdown("Now including: live stock pricing, news parsing, SEC filings, trend matching, and AI predictions.")

# Step 1: Scrape posts
reddit_posts = scrape_reddit_posts()
twitter_posts = scrape_tweets()
all_posts = reddit_posts + twitter_posts

# Step 2: Sentiment scoring
scored_posts = analyze_sentiment(all_posts)

# Step 3: Ticker extraction and aggregation
summary_df = extract_and_aggregate(scored_posts)

# Step 4: Enrich data
summary_df["Live Price"] = summary_df["Ticker"].apply(get_live_price)
summary_df["News"] = summary_df["Ticker"].apply(get_news_articles)
summary_df["SEC Filings"] = summary_df["Ticker"].apply(get_sec_filings)
summary_df["Trend Match"] = summary_df["Ticker"].apply(compare_with_historical)
summary_df["AI Prediction"] = summary_df.apply(predict_signal, axis=1)

# Step 5: Filter and display
if not summary_df.empty and "Live Price" in summary_df.columns:
    price_filter = st.slider("Max Price", 1, 200, 50)
    filtered = summary_df[summary_df["Live Price"] <= price_filter]
    st.subheader("📊 Predicted Meme Stock Opportunities")
    st.dataframe(filtered.reset_index(drop=True))
else:
    st.warning("No tickers found.")