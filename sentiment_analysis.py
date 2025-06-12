import pandas as pd
from textblob import TextBlob
import re
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)  # remove URLs
    tweet = re.sub(r'@\w+', '', tweet)     # remove mentions
    tweet = re.sub(r'#', '', tweet)        # remove hashtag symbol
    tweet = re.sub(r'\n', ' ', tweet)      # remove new lines
    tweet = re.sub(r'[^\w\s]', '', tweet)  # remove punctuation
    tweet = tweet.lower()                   # convert to lowercase
    return tweet.strip()
def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity == 0:
        return 'neutral'
    else:
        return 'negative'
def main():
    df = pd.read_csv('Tweets (1)-checkpoint.csv')
    # Clean tweets
    print("Cleaning tweets...")
    df['cleaned_tweet'] = df['text'].astype(str).apply(clean_tweet)

    # Perform sentiment analysis
    print("Analyzing sentiment...")
    df['sentiment'] = df['cleaned_tweet'].apply(get_sentiment)

    # Show sample results
    print(df[['text', 'cleaned_tweet', 'sentiment']].head())

    # Sentiment distribution
    print("\nSentiment distribution:")
    print(df['sentiment'].value_counts())

    # Save results to CSV
    df.to_csv('sentiment_analysis_results.csv', index=False)
    print("\nResults saved to 'sentiment_analysis_results.csv'")

if __name__ == "__main__":
    main()