import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the data
twitter_training = pd.read_csv(r'D:\Data Analyst\Prodigy InfoTech\Task-4\Raw_data\twitter_training.csv')
twitter_validation = pd.read_csv(r'D:\Data Analyst\Prodigy InfoTech\Task-4\Raw_data\twitter_validation.csv')

# Standardize column names
twitter_training.columns = ['ID', 'Topic', 'Sentiment', 'Tweet']
twitter_validation.columns = ['ID', 'Topic', 'Sentiment', 'Tweet']

# Drop rows with missing tweet text
twitter_training_cleaned = twitter_training.dropna(subset=['Tweet'])
twitter_validation_cleaned = twitter_validation.dropna(subset=['Tweet'])

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply text preprocessing
twitter_training_cleaned['Tweet'] = twitter_training_cleaned['Tweet'].apply(preprocess_text)
twitter_validation_cleaned['Tweet'] = twitter_validation_cleaned['Tweet'].apply(preprocess_text)

# Function to classify sentiment using TextBlob
def classify_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Apply sentiment classification
twitter_training_cleaned['Sentiment'] = twitter_training_cleaned['Tweet'].apply(classify_sentiment)
twitter_validation_cleaned['Sentiment'] = twitter_validation_cleaned['Tweet'].apply(classify_sentiment)

# Plot sentiment distribution
def plot_sentiment_distribution(df, dataset_name):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Sentiment', palette='viridis')
    plt.title(f'Sentiment Distribution in {dataset_name}')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

plot_sentiment_distribution(twitter_training_cleaned, 'Training Data')
plot_sentiment_distribution(twitter_validation_cleaned, 'Validation Data')

# Analyze sentiment distribution across different topics
def plot_sentiment_across_topics(df, dataset_name):
    plt.figure(figsize=(14, 8))
    sns.countplot(data=df, x='Topic', hue='Sentiment', palette='viridis')
    plt.title(f'Sentiment Distribution Across Topics in {dataset_name}')
    plt.xlabel('Topic')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()

plot_sentiment_across_topics(twitter_training_cleaned, 'Training Data')
plot_sentiment_across_topics(twitter_validation_cleaned, 'Validation Data')

# Generate word clouds
def generate_word_cloud(df, sentiment, title):
    tweets = ' '.join(df[df['Sentiment'] == sentiment]['Tweet'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(tweets)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

generate_word_cloud(twitter_training_cleaned, 'Positive', 'Word Cloud for Positive Tweets')
generate_word_cloud(twitter_training_cleaned, 'Negative', 'Word Cloud for Negative Tweets')

# Insights and Recommendations
def print_insights(df):
    print("Top Topics with Positive Sentiment:")
    positive_topics = df[df['Sentiment'] == 'Positive']['Topic'].value_counts().head(10)
    print(positive_topics)
    
    print("\nTop Topics with Negative Sentiment:")
    negative_topics = df[df['Sentiment'] == 'Negative']['Topic'].value_counts().head(10)
    print(negative_topics)
    
    print("\nOverall Sentiment Distribution:")
    print(df['Sentiment'].value_counts())

print_insights(twitter_training_cleaned)
print_insights(twitter_validation_cleaned)
