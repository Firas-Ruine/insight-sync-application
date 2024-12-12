from flask import Flask, render_template, request, flash
from googleapiclient.discovery import build
import joblib
import csv
import re
import psycopg2
import os

app = Flask(__name__)

MODEL_PATH = 'src/models/emotion_classifier_pipe_lr.pkl'
model = joblib.load(MODEL_PATH)

API_KEY = 'AIzaSyDTWzUmomxive8x9Q_GYmF9CTxmzDJ2qVg'
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("YOUTUBE_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host="postgres",
        port="5432"
    )
    
def extract_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    Handles various YouTube URL formats.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

@app.route('/')
def home():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT video_id, video_title, positive_count, negative_count, neutral_count, created_at
        FROM youtube_video_sentiments
        ORDER BY created_at DESC;
    """)
    data = cur.fetchall()
    cur.close()
    conn.close()
    
    return render_template('home.html', data=data)

@app.route('/facebook')
def facebook_route():
    return render_template('facebook.html')

@app.route('/youtube', methods=['GET', 'POST'])
def youtube_route():
    comments = []  # Initialize comments list
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    if request.method == 'POST':
        youtube_url = request.form.get('youtube_url')
        if youtube_url:
            video_id = extract_video_id(youtube_url)
            if video_id:
                try:
                    # Log the URL into the CSV file
                    try:
                        csv_file_path = 'src/data_ingestion/youtube_comments/inputs/channels.csv'
                        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
                            existing_urls = {row[0] for row in csv.reader(csvfile)}
                    except FileNotFoundError:
                        # If the file doesn't exist, create an empty set
                        existing_urls = set()

                    # Add the URL only if it doesn't already exist
                    if youtube_url not in existing_urls:
                        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([youtube_url])

                    # Fetch video details and comments using YouTube API
                    video_response = youtube.videos().list(
                        part='snippet',
                        id=video_id
                    ).execute()

                    video_title = video_response['items'][0]['snippet']['title']

                    comments_response = youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        maxResults=100
                    ).execute()

                    for item in comments_response.get('items', []):
                        comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                        # Predict sentiment
                        sentiment = model.predict([comment_text])[0]
                        comments.append({
                            'text': comment_text,
                            'sentiment': sentiment
                        })

                        # Count sentiment
                        if sentiment == "positive":
                            positive_count += 1
                        elif sentiment == "negative":
                            negative_count += 1
                        else:
                            neutral_count += 1

                    # Save video ID, title, and sentiment counts to database
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute(
                        """
                        INSERT INTO youtube_video_sentiments (video_id, video_title, positive_count, negative_count, neutral_count)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (video_id, video_title, positive_count, negative_count, neutral_count)
                    )
                    conn.commit()
                    cur.close()
                    conn.close()

                    # Flash message with sentiment counts
                    flash(f"Comments fetched successfully for '{video_title}' (ID: {video_id})! Positive: {positive_count}, Negative: {negative_count}, Neutral: {neutral_count}", "success")

                except Exception as e:
                    flash(f"Error fetching comments: {str(e)}", "danger")
            else:
                flash("Invalid YouTube URL. Please check and try again.", "danger")
        else:
            flash("Please enter a valid YouTube URL.", "danger")

    sentiment_data = {
        'positive': positive_count,
        'negative': negative_count,
        'neutral': neutral_count
    }
    return render_template('youtube.html', comments=comments, sentiment_data=sentiment_data)

def initialize_database():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS youtube_video_sentiments (
            id SERIAL PRIMARY KEY,
            video_id VARCHAR(255) NOT NULL,
            video_title TEXT NOT NULL,
            positive_count INT NOT NULL,
            negative_count INT NOT NULL,
            neutral_count INT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    
if __name__ == "__main__":
    initialize_database()
    app.run(debug=True)
