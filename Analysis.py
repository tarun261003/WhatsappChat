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
from dateutil import parser

# Set page configuration
st.set_page_config(page_title="WhatsApp Chat Analysis", layout="wide")

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

def displayLocalGIF(placeholder, imagePath, caption):
    # Center-align using Markdown and CSS
    placeholder.markdown(
        f"""
        <div style="text-align: center;">
            <img src="{imagePath}" width="500" />
            <p>{caption}</p>
        </div>
        """,
        unsafe_allow_html=True  # Required to use HTML in Markdown
    )


# Function to load and process the chat data from uploaded file
@st.cache_data
def load_chat_data(file):
    message_pattern = r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4},?\s\d{1,2}:\d{2}(?:\s?[APap][Mm])?\s-\s"  
    lines = file.read().decode("utf-8").splitlines()

    messages = []
    current_message = None

    for line in lines:
        if re.match(message_pattern, line):
            if current_message:
                messages.append(current_message)

            date_time, rest = line.split(" - ", 1)
            sender_message_split = rest.split(": ", 1)
            if len(sender_message_split) > 1:
                sender, message = sender_message_split
            else:
                sender, message = "System", rest.strip()

            try:
                parsed_date = parser.parse(date_time, fuzzy=True)
            except Exception:
                parsed_date = None

            current_message = {
                "DateTime": parsed_date,
                "Sender": sender.strip(),
                "Message": message.strip()
            }
        else:
            if current_message:
                current_message["Message"] += f" {line.strip()}"

    if current_message:
        messages.append(current_message)

    df = pd.DataFrame(messages)
    df['Date'] = df['DateTime'].dt.date
    df['Time'] = df['DateTime'].dt.time
    return df

# File uploader to upload WhatsApp chat .txt file
uploaded_file = st.file_uploader("Upload WhatsApp Chat .txt", type="txt")
image_placeholder1 = st.empty()
imagePath1 = "https://github.com/tarun261003/WhatsappChat/blob/main/Intro.gif?raw=true"
displayLocalGIF(image_placeholder1, imagePath1, "Remote Image")

if uploaded_file is not None:
    chat_df = load_chat_data(uploaded_file)
    image_placeholder1.empty()

    sid = SentimentIntensityAnalyzer()
    chat_df = chat_df[chat_df['Sender'] != "System"]
    chat_df['Message'] = chat_df['Message'].fillna('')
    chat_df = chat_df[~chat_df['Message'].isin(["Media omitted", "GIF omitted", "Sticker omitted"])]
    chat_df['Sentiment'] = chat_df['Message'].apply(lambda x: sid.polarity_scores(x)['compound'])

    st.title("WhatsApp Chat Analysis Dashboard")

    st.sidebar.header("Filter Messages")
    start_date = st.sidebar.date_input("Start Date", min_value=chat_df['Date'].min(), value=chat_df['Date'].min())
    end_date = st.sidebar.date_input("End Date", max_value=chat_df['Date'].max(), value=chat_df['Date'].max())
    selected_senders = st.sidebar.multiselect("Select Participants", options=chat_df['Sender'].unique())

    filtered_df = chat_df[
        (chat_df['Date'] >= start_date) & (chat_df['Date'] <= end_date)
    ]

    if selected_senders:
        filtered_df = filtered_df[filtered_df['Sender'].isin(selected_senders)]

    st.header("Key Metrics")
    total_messages = len(filtered_df)
    unique_participants = filtered_df['Sender'].nunique()
    most_active_day = filtered_df['Date'].value_counts().idxmax()

    st.metric("Total Messages", total_messages)
    st.metric("Unique Participants", unique_participants)
    st.metric("Most Active Day", most_active_day.strftime('%Y-%m-%d'))


    st.subheader("Daily and Hourly Activity")
    daily_messages = filtered_df.groupby('Date').size()
    hourly_messages = filtered_df.groupby(filtered_df['DateTime'].dt.hour).size()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    daily_messages.plot(ax=ax1, title='Messages per Day')
    hourly_messages.plot(kind='bar', ax=ax2, title='Messages per Hour')
    st.pyplot(fig)

    st.subheader("Most Common Words")
    all_words = ' '.join(filtered_df['Message'])
    wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(all_words)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    st.subheader("Top Emojis")
    emoji_list = [char for char in all_words if char in emoji.EMOJI_DATA]
    emoji_freq = pd.Series(emoji_list).value_counts().head(10)
    st.bar_chart(emoji_freq)

    st.subheader("Sentiment Trend")
    filtered_df['Rolling Sentiment'] = filtered_df['Sentiment'].rolling(window=30, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    filtered_df.set_index('DateTime')['Rolling Sentiment'].plot(ax=ax)
    ax.set_title("Sentiment Over Time")
    st.pyplot(fig)

    st.subheader("Sentiment Breakdown by Participant")
    sentiment_by_sender = filtered_df.groupby('Sender')['Sentiment'].mean()
    st.bar_chart(sentiment_by_sender)

    st.subheader("Message Frequency by Day of the Week")
    chat_df['Day of Week'] = chat_df['DateTime'].dt.day_name()
    day_of_week_counts = chat_df['Day of Week'].value_counts()
    st.bar_chart(day_of_week_counts)

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

    # Calculate streaks and largest gaps
    chat_df['Sender Shift'] = chat_df['Sender'].shift()
    chat_df['Streak ID'] = (chat_df['Sender'] != chat_df['Sender Shift']).cumsum()

    # Calculate the streak lengths
    streak_lengths = chat_df.groupby(['Sender', 'Streak ID']).size().reset_index(name='Streak Length')

    # Find the longest streak for each sender
    longest_streaks = streak_lengths.groupby('Sender')['Streak Length'].max().reset_index()
    longest_streaks.columns = ['Sender', 'Longest Streak']

    # Calculate time gaps for each sender
    chat_df['Time Gap'] = chat_df['DateTime'] - chat_df['DateTime'].shift()
    chat_df['Time Gap'] = chat_df.apply(lambda row: row['Time Gap'] if row['Sender'] == row['Sender Shift'] else pd.Timedelta(0), axis=1)

    # Find the largest gap for each sender
    largest_gaps = chat_df.groupby('Sender')['Time Gap'].max().reset_index()
    largest_gaps.columns = ['Sender', 'Largest Time Gap']

    # Merge results into a single DataFrame
    streak_gap_df = pd.merge(longest_streaks, largest_gaps, on='Sender')

    # Display results
    st.subheader("Streaks and Largest Gaps")
    st.dataframe(streak_gap_df)

    # Optional: Visualize the longest streaks
    st.bar_chart(streak_gap_df.set_index('Sender')['Longest Streak'])

    # Optional: Visualize the largest time gaps
    largest_gaps['Largest Time Gap (Hours)'] = largest_gaps['Largest Time Gap'].dt.total_seconds() / 3600
    st.bar_chart(largest_gaps.set_index('Sender')['Largest Time Gap (Hours)'])
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
    chat_df['Date'] = pd.to_datetime(chat_df['Date'])  # Ensure 'Date' is a datetime object
    chat_df['Month'] = chat_df['Date'].dt.to_period('M')

    monthly_trends = chat_df['Month'].value_counts().sort_index()
    st.line_chart(monthly_trends)
