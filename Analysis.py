import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import emoji
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import seaborn as sns
from collections import Counter
import html

# Set page configuration
st.set_page_config(page_title="WhatsApp Chat Analysis", layout="wide")

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

def displayLocalGIF(placeholder, imagePath, caption):
    placeholder.image(
        imagePath,
        # use_column_width=False,  # Disable container width
        width=500,               # Set the width
        caption=caption       # Optional caption
    )
# Function to load and process the chat data from uploaded file
@st.cache_data
def load_chat_data(file):
    # Read chat data as text, not bytes
    data = file.read().decode("utf-8").splitlines()
    dates, times, senders, messages = [], [], [], []
    
    # Regular expression pattern for message lines
    pattern = r"(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}\s[APM]{2}) - ([^:]+): (.+)"
    
    # Extract structured data
    for line in data:
        # Skip encryption notice lines
        if "Messages and calls are end-to-end encrypted" in line:
            continue
        match = re.match(pattern, line)
        if match:
            date, time, sender, message = match.groups()
            dates.append(date)
            times.append(time)
            senders.append(sender)
            messages.append(message)
    
    # Create DataFrame
    chat_df = pd.DataFrame({
        'Date': dates,
        'Time': times,
        'Sender': senders,
        'Message': messages
    })
    
    # Convert 'Date' and 'Time' to datetime objects
    chat_df['Date'] = pd.to_datetime(chat_df['Date'], format='%m/%d/%y')
    chat_df['Time'] = pd.to_datetime(chat_df['Time'], format='%I:%M %p').dt.time
    return chat_df


# File uploader to upload WhatsApp chat .txt file
uploaded_file = st.file_uploader("Upload WhatsApp Chat .txt", type="txt")
image_placeholder1 = st.empty()
imagePath1 ="https://github.com/tarun261003/WhatsappChat/blob/main/Intro.gif?raw=true"
displayLocalGIF(image_placeholder1, imagePath1,"Remote Image")
if uploaded_file is not None:
    chat_df = load_chat_data(uploaded_file)
    image_placeholder1.empty()
    # st.image('https://github.com/tarun261003/WhatsappChat/blob/main/Intro.gif?raw=true', use_column_width=True)
    # Initialize sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Clean and preprocess data
    chat_df['Message'] = chat_df['Message'].fillna('')
    chat_df = chat_df[~chat_df['Message'].isin(["Media omitted", "GIF omitted", "Sticker omitted"])]  

    # Perform sentiment analysis on each message
    chat_df['Sentiment'] = chat_df['Message'].apply(lambda x: sid.polarity_scores(x)['compound'])

    # Streamlit UI setup
    st.title("WhatsApp Chat Analysis Dashboard")

    # Sidebar Filters
    st.sidebar.header("Filter Messages")
    start_date = st.sidebar.date_input("Start Date", chat_df['Date'].min())
    end_date = st.sidebar.date_input("End Date", chat_df['Date'].max())
    selected_senders = st.sidebar.multiselect("Select Participants", options=chat_df['Sender'].unique())

    # Filter data based on user input
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = chat_df[(chat_df['Date'] >= start_date) & (chat_df['Date'] <= end_date)]
    if selected_senders:
        filtered_df = filtered_df[filtered_df['Sender'].isin(selected_senders)]

    # Display key metrics
    st.header("Key Metrics")
    total_messages = len(filtered_df)
    unique_participants = filtered_df['Sender'].nunique()
    most_active_day = filtered_df['Date'].value_counts().idxmax().strftime('%Y-%m-%d')

    st.metric("Total Messages", total_messages)
    st.metric("Unique Participants", unique_participants)
    st.metric("Most Active Day", most_active_day)

    # Activity Over Time
    st.subheader("Daily and Hourly Activity")
    daily_messages = filtered_df.groupby(filtered_df['Date']).size()
    hourly_messages = filtered_df.groupby(filtered_df['Time'].apply(lambda x: x.hour)).size()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    daily_messages.plot(ax=ax1, title='Messages per Day')
    hourly_messages.plot(kind='bar', ax=ax2, title='Messages per Hour')
    st.pyplot(fig)

    # Word Cloud with Zoom-in Effect using CSS
    st.subheader("Most Common Words")
    all_words = ' '.join(filtered_df['Message'])
    wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(all_words)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    # Emoji Analysis
    st.subheader("Top Emojis")
    emoji_list = [char for char in all_words if char in emoji.EMOJI_DATA]
    emoji_freq = pd.Series(emoji_list).value_counts().head(10)
    st.bar_chart(emoji_freq)

    # Sentiment Over Time
    st.subheader("Sentiment Trend")
    filtered_df['Rolling Sentiment'] = filtered_df['Sentiment'].rolling(window=30, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    filtered_df.set_index('Date')['Rolling Sentiment'].plot(ax=ax)
    ax.set_title("Sentiment Over Time")
    st.pyplot(fig)

    # Sentiment Breakdown by Sender
    st.subheader("Sentiment Breakdown by Participant")
    sentiment_by_sender = filtered_df.groupby('Sender')['Sentiment'].mean()
    st.bar_chart(sentiment_by_sender)

    # Response Time Analysis
    response_times = filtered_df.groupby('Sender')['Date'].diff().fillna(pd.Timedelta(seconds=0))
    filtered_df['Response Time (mins)'] = response_times.dt.total_seconds() / 60
    response_time_by_sender = filtered_df.groupby('Sender')['Response Time (mins)'].mean()
    st.subheader("Average Response Time by Participant")
    st.bar_chart(response_time_by_sender)

    # Message Frequency by Day of the Week
    st.subheader("Message Frequency by Day of the Week")
    chat_df['Day of Week'] = chat_df['Date'].dt.day_name()
    day_of_week_counts = chat_df['Day of Week'].value_counts()
    st.bar_chart(day_of_week_counts)

    # Average Message Length by Participant
    st.subheader("Average Message Length by Participant")
    chat_df['Message Length'] = chat_df['Message'].apply(len)
    avg_msg_length = chat_df.groupby('Sender')['Message Length'].mean()
    st.bar_chart(avg_msg_length)

    # Message Activity Heatmap (Hourly by Day of Week)
    st.subheader("Message Activity Heatmap (Hourly by Day of Week)")
    chat_df['Hour'] = chat_df['Time'].apply(lambda x: x.hour)
    activity_heatmap = pd.crosstab(chat_df['Hour'], chat_df['Day of Week'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(activity_heatmap, cmap="YlGnBu", ax=ax)
    ax.set_title("Hourly Activity by Day of the Week")
    st.pyplot(fig)


    # Top Words by Participant with Zoom-in Effect
    st.subheader("Top Words by Participant")
    for sender in chat_df['Sender'].unique():
        words = ' '.join(chat_df[chat_df['Sender'] == sender]['Message']).split()
        most_common_words = Counter(words).most_common(10)
        
        word_list_html = f"<h3>Top words for {sender}:</h3>"
        for idx, (word, _) in enumerate(most_common_words):
            sanitized_word = html.escape(word)
            word_list_html += f"<span class='zoom-word' style='animation-delay:{idx * 0.1}s'>{sanitized_word}</span> "

        st.markdown("""
        <style>
            .zoom-word {
                display: inline-block;
                opacity: 0;
                transform: scale(0);
                animation: zoom-in 0.5s forwards;
                margin-right: 10px;
            }

            @keyframes zoom-in {
                0% {
                    opacity: 0;
                    transform: scale(0);
                }
                100% {
                    opacity: 1;
                    transform: scale(1);
                }
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(word_list_html, unsafe_allow_html=True)

    # Monthly Message Trends
    st.subheader("Monthly Message Trends")
    chat_df['Month'] = chat_df['Date'].dt.to_period('M')
    monthly_trends = chat_df['Month'].value_counts().sort_index()
    st.line_chart(monthly_trends)
