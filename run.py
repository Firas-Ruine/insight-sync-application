from flask import Flask, render_template, request, flash
from googleapiclient.discovery import build
import joblib
import csv
import re
import psycopg2
import os
import logging
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Sentry
sentry_sdk.init(
    dsn="https://6e4bf6e3cacce1c0bba82f1266074e05@o4508458067034112.ingest.de.sentry.io/4508458068672592",
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0
)

app = Flask(__name__)
app.secret_key = "secret_key"

MODEL_PATH = 'src/models/emotion_classifier_pipe_lr.pkl'
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

API_KEY = "AIzaSyDTWzUmomxive8x9Q_GYmF9CTxmzDJ2qVg"
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("YOUTUBE_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432")
        )
        logger.debug("Database connection established.")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise

def extract_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    Handles various YouTube URL formats.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

@app.route('/')
def home():
    try:
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
    except Exception as e:
        logger.error(f"Error fetching data for home page: {e}")
        flash("An error occurred while loading the homepage data.", "danger")
        data = []
    
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
                    csv_file_path = 'src/data_ingestion/youtube_comments/inputs/channels.csv'
                    existing_urls = set()
                    try:
                        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                            existing_urls = {row[0] for row in csv.reader(csvfile)}
                    except FileNotFoundError:
                        logger.warning("CSV file not found. Creating a new one.")

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

                    flash(f"Comments fetched successfully for '{video_title}' (ID: {video_id})! Positive: {positive_count}, Negative: {negative_count}, Neutral: {neutral_count}", "success")

                except Exception as e:
                    logger.error(f"Error fetching comments: {e}")
                    flash(f"Error fetching comments: {e}", "danger")
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

@app.route("/test-error")
def hello_world():
    1/0  # raises an error
    return "<p>Hello, World!</p>"

def initialize_database():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        logger.info("Checking and creating youtube_video_sentiments table if not exists.")
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
        logger.info("Database initialized successfully.")
    except psycopg2.Error as db_err:
        logger.error(f"Database error: {db_err.pgerror}")
        raise
    except Exception as e:
        logger.error(f"Unhandled error during database initialization: {e}")
        raise

if __name__ == "__main__":
    initialize_database()
    app.run(debug=True, host="0.0.0.0")
