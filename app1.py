# -----------------------
# APP.PY - LOGIN + FULL SENTIMENT ANALYSIS
# -----------------------

import streamlit as st
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
import re
import requests
from bs4 import BeautifulSoup

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Sentiment Studio — Turning words into feelings", layout="wide")

# LOGIN SYSTEM
# -----------------------
USER_CREDENTIALS = {
    "admin": "password123",
    "user": "userpass",
    "moderator" :"123#"
}

if "login" not in st.session_state:
    st.session_state.login = False


def login_page():
    # Full-page CSS for login with background image
    st.markdown("""
    <style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1507525428034-b723cf961d3e'); /* Replace with your image URL */
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            font-family: "Inter", "Segoe UI", Roboto, Arial, sans-serif;
        }
        .login-card {
            max-width: 400px;
            margin: 100px auto;
            padding: 30px;
            background: rgba(255, 255, 255, 0.85);  /* semi-transparent */
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
        }
        .login-card h2 {
            color: #0f172a;
            margin-bottom: 20px;
            font-weight: 700;
        }
        .stTextInput>div>div>input {
            height: 40px;
            font-size: 16px;
        }
        .stButton>button {
            width: 100%;
            height: 45px;
            background: linear-gradient(90deg,#06b6d4,#10b981);
            color: white;
            font-weight: bold;
            font-size: 16px;
            border-radius: 12px;
            border: none;
            cursor: pointer;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<h2>🔒 Login to Sentiment Studio</h2>', unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.login = True
            st.rerun()
        else:
            st.error("❌ Invalid username or password")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
if "login" not in st.session_state:
    st.session_state.login = False
if not st.session_state.login:
    login_page()
    st.stop()

st.markdown(
    """
    <style>
   
    /* Header */
    .app-title {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
    }
    .app-sub {
        color: #475569;
        margin-bottom: 12px;
    }

    /* Card container */
    .card {
        background: white;
        padding: 14px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 14px;
    }

    /* Tweet card */
    .tweet-card {
        padding: 12px;
        border-radius: 10px;
        border-left: 4px solid rgba(59,130,246,0.08);
    }
    .tweet-header {
        display:flex; align-items:center; gap:8px; margin-bottom:6px;
    }
    .avatar {
        width:36px; height:36px; border-radius:50%; background:#e6f0ff; display:inline-block; text-align:center; line-height:36px; font-weight:700; color:#0369a1;
    }
    .username { font-weight:700; color:#0f172a; }
    .handle { color:#64748b; font-size:13px; margin-left:6px; }

    .tweet-text { color:#0b1220; margin-top:6px; margin-bottom:8px; line-height:1.45; }

    /* Sentiment badge */
    .badge {
        padding:6px 10px; border-radius:999px; font-weight:700; font-size:13px; display:inline-block;
    }
    .pos { background: linear-gradient(90deg,#06b6d4,#10b981); color:white; }
    .neu { background: linear-gradient(90deg,#94a3b8,#e2e8f0); color:white; }
    .neg { background: linear-gradient(90deg,#fb7185,#f97316); color:white; }

    /* Highlights */
    .highlight-box {
        background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(255,255,255,0.7));
        padding:10px; border-radius:10px; border-left:6px solid rgba(59,130,246,0.12);
    }

    /* Small responsive tweaks */
    @media (max-width: 600px) {
        .tweet-card { padding:10px; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Helpers & Model loading
# -----------------------
@st.cache_resource
def load_data_and_model():
    # Attempt to load CSV (graceful if missing)
    try:
        df = pd.read_csv("Tweets123456.csv")
    except Exception:
        df = pd.DataFrame(columns=["usernames", "text"])
    # ensure no "tips" column shown in app
    if "tips" in df.columns:
        df = df.drop(columns=["tips"])
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return df, model, vectorizer

def safe_map_label(label):
    """Map numeric or string labels to emoji+text and short class"""
    try:
        lab = int(label)
        mapping_text = {0: ("😡 Negative", "neg"), 1: ("😐 Neutral", "neu"), 2: ("😊 Positive", "pos")}
        return mapping_text.get(lab, (str(label), "neu"))
    except Exception:
        s = str(label).lower()
        if "pos" in s: return ("😊 Positive", "pos")
        if "neg" in s: return ("😡 Negative", "neg")
        if "neu" in s: return ("😐 Neutral", "neu")
        return (str(label), "neu")

def predict_with_proba(model, X):
    """Return (labels, probs_or_None)"""
    try:
        probs = model.predict_proba(X)
        labels = probs.argmax(axis=1)
        return labels, probs
    except Exception:
        labels = model.predict(X)
        return labels, None

def clean_text(t):
    if pd.isna(t): return ""
    return re.sub(r'\s+', ' ', str(t)).strip()

def fetch_text_from_url(url, limit=3000):
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text[:limit]
    except Exception:
        return None

# Load
df, model, vectorizer = load_data_and_model()

# -----------------------
# UI Header
# -----------------------
st.markdown('<div class="app-title"> Sentiment Decoder - Emotion Insight Analyzer  </div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Modern card-based interface with ,manual, CSV, Username modes, and YouTube Analyzer.</div>', unsafe_allow_html=True)

# Layout: left main, right summary
left, right = st.columns([3, 1])

# -----------------------
# LEFT: Main interactive area
# -----------------------
with left:
    # MANUAL INPUT CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✍️ Quick Sentiment (Manual)")
    manual = st.text_area("Type a sentence to analyze", height=80, placeholder="I love this product — it works great!")
    if st.button("Predict (Manual)"):
        if manual.strip() == "":
            st.warning("Enter text to analyze.")
        else:
            X = vectorizer.transform([manual])
            labels, probs = predict_with_proba(model, X)
            label_text, short_class = safe_map_label(labels[0] if hasattr(labels, "__len__") else labels)
            badge_class = {"pos":"pos","neg":"neg","neu":"neu"}.get(short_class, "neu")
            # show tweet-like card
            st.markdown(f'''
                <div class="tweet-card card">
                    <div class="tweet-header">
                        <div class="avatar">U</div>
                        <div>
                            <div class="username">You</div>
                            <div class="handle">@manual</div>
                        </div>
                        <div style="margin-left:auto;">
                            <span class="badge {badge_class}">{label_text}</span>
                        </div>
                    </div>
                    <div class="tweet-text">{clean_text(manual)}</div>
                </div>
            ''', unsafe_allow_html=True)

            # probabilities if available
            if probs is not None:
                prob_row = probs[0]
                prob_df = pd.DataFrame({
                    "Sentiment": ["Negative", "Neutral", "Positive"],
                    "Probability": [f"{p:.2f}" for p in prob_row]
                })
                st.table(prob_df)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")  # spacer

    # CSV UPLOAD CARD
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📥 Upload CSV ")
    uploaded = st.file_uploader("Upload a CSV with a 'text' column", type=["csv"], key="upload1")
    if uploaded is not None:
        try:
            up_df = pd.read_csv(uploaded)
            if "text" not in up_df.columns:
                st.error("CSV needs a 'text' column.")
            else:
                if st.button("Predict uploaded CSV"):
                    Xnew = vectorizer.transform(up_df["text"].astype(str))
                    labels, probs = predict_with_proba(model, Xnew)
                    up_df = up_df.copy()
                    up_df["Predicted"] = [safe_map_label(l)[0] for l in labels]
                    st.success(f"Predicted {len(up_df)} rows.")
                    st.dataframe(up_df.head(200))
                    st.download_button("Download results CSV", data=up_df.to_csv(index=False), file_name="predicted_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    # USERNAME-BASED ANALYSIS (from default CSV)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧭 Username Sentiment (from dataset)")
    if "usernames" not in df.columns or "text" not in df.columns:
        st.warning("Dataset missing 'usernames' or 'text' columns.")
    else:
        usernames = sorted(df["usernames"].dropna().unique().tolist())
        c1, c2 = st.columns([3,1])
        with c1:
            username = st.selectbox("Choose username", ["-- select --"] + usernames)
        with c2:
            if st.button("🎲 Random User", key="rand1"):
                username = random.choice(usernames)
                st.info(f"Selected: {username}")

        if username and username != "-- select --":
            user_tweets = df[df["usernames"] == username].copy()
            st.markdown(f"**Found {len(user_tweets)} tweets for @{username}**")
            if not user_tweets.empty:
                Xu = vectorizer.transform(user_tweets["text"].astype(str))
                labels, probs = predict_with_proba(model, Xu)
                user_tweets["pred_label"] = labels
                user_tweets["pred_text"] = [safe_map_label(l)[0] for l in labels]

                # show top 50 tweets as cards
                st.markdown("#### Recent tweets (sample)")
                for idx, row in user_tweets.head(50).iterrows():
                    label_text, short_class = safe_map_label(row["pred_label"])
                    badge_class = {"pos":"pos","neg":"neg","neu":"neu"}.get(short_class, "neu")
                    st.markdown(f'''
                        <div class="tweet-card card">
                            <div class="tweet-header">
                                <div class="avatar">{str(row['usernames'])[0:1].upper()}</div>
                                <div>
                                    <div class="username">{row['usernames']}</div>
                                    <div class="handle">@{row['usernames']}</div>
                                </div>
                                <div style="margin-left:auto;">
                                    <span class="badge {badge_class}">{label_text}</span>
                                </div>
                            </div>
                            <div class="tweet-text">{clean_text(row['text'])}</div>
                        </div>
                    ''', unsafe_allow_html=True)

                # Distribution chart
                cleaned = (
                    user_tweets["pred_text"].astype(str)
                    .str.replace(r'[^\w\s]', ' ', regex=True)
                    .str.replace(r'[_\d]+', ' ', regex=True)
                    .str.strip()
                    .str.lower()
                )
                counts = cleaned.value_counts()
                # colors
                colors = []
                for lab in counts.index:
                    if "positive" in lab: colors.append("#06b6d4")
                    elif "negative" in lab: colors.append("#fb7185")
                    elif "neutral" in lab: colors.append("#94a3b8")
                    else: colors.append("#60a5fa")
                   
                
                fig, ax = plt.subplots(figsize=(4,3))
                counts.plot(kind="bar", ax=ax, color=colors, edgecolor="black", width=0.55)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=5)
                ax.tick_params(axis='y', labelsize=5)
                ax.set_title(f"Sentiment distribution for @{username}",fontsize=4)
                ax.set_xlabel("Sentiment", fontsize=4)
                ax.set_ylabel("Count", fontsize=4)
                for i, v in enumerate(counts.values):
                    ax.text(i, v + 0.2, str(v), ha="center", fontweight="regular")
                    plt.tight_layout()
                    col1,col2,col3= st.columns([1,2,1])
                    with col2:
                        st.pyplot(fig)
    

                # Find most positive & negative using probs if available
                most_pos = None
                most_neg = None
                if probs is not None:
                    try:
                        classes = list(model.classes_)
                        # heuristics
                        pos_idx = None; neg_idx = None
                        for i_c, c in enumerate(classes):
                            s = str(c).lower()
                            if "pos" in s: pos_idx = i_c
                            if "neg" in s: neg_idx = i_c
                        if pos_idx is None and 2 in classes: pos_idx = classes.index(2)
                        if neg_idx is None and 0 in classes: neg_idx = classes.index(0)
                        # pick rows with highest pos prob and highest neg prob
                        if pos_idx is not None:
                            pos_scores = probs[:, pos_idx]
                            most_pos = user_tweets.iloc[int(pos_scores.argmax())]["text"]
                        if neg_idx is not None:
                            neg_scores = probs[:, neg_idx]
                            most_neg = user_tweets.iloc[int(neg_scores.argmax())]["text"]
                    except Exception:
                        most_pos = None
                        most_neg = None

                # fallback: use predicted labels
                if most_pos is None:
                    posrows = user_tweets[user_tweets["pred_label"] == 2]
                    if not posrows.empty:
                        most_pos = posrows.iloc[0]["text"]
                if most_neg is None:
                    negrows = user_tweets[user_tweets["pred_label"] == 0]
                    if not negrows.empty:
                        most_neg = negrows.iloc[0]["text"]


# 🎥 YouTube Comment Sentiment Analyzer (API)
# -------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🎥 YouTube Comment Sentiment Analyzer (API)")

yt_api = st.text_input("Enter YouTube API Key", type="password")
yt_link = st.text_input("Paste YouTube Video Link")

def extract_video_id(url):
    import re
    patterns = [
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"youtube\.com/watch\?v=([A-Za-z0-9_-]{11})",
        r"youtube\.com/shorts/([A-Za-z0-9_-]{11})",
        r"youtube\.com/live/([A-Za-z0-9_-]{11})"
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None

def fetch_youtube_comments(api_key, video_id, max_comments=50):
    import requests
    comments = []
    url = (
        f"https://www.googleapis.com/youtube/v3/commentThreads"
        f"?part=snippet&videoId={video_id}&key={api_key}&maxResults=100"
    )

    try:
        r = requests.get(url).json()
        if "items" not in r:
            return []

        for item in r["items"]:
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(text)
            if len(comments) >= max_comments:
                break
        return comments
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return []

if st.button("Analyze YouTube Comments"):
    if yt_api.strip() == "" or yt_link.strip() == "":
        st.warning("Please enter YouTube API key and video link.")
    else:
        video_id = extract_video_id(yt_link)
        if video_id is None:
            st.error("Invalid YouTube link.")
        else:
            with st.spinner("Fetching comments via API..."):
                comments = fetch_youtube_comments(yt_api, video_id, max_comments=100)

            if not comments:
                st.error("No comments found or API error.")
            else:
                st.success(f"Fetched {len(comments)} comments!")

                # Predict sentiment
                Xc = vectorizer.transform(comments)
                labels, probs = predict_with_proba(model, Xc)

                results = []
                for comment, lab in zip(comments, labels):
                    label_text, short_class = safe_map_label(lab)
                    results.append((comment, label_text, short_class))

                # Show results as cards
                st.markdown("#### 📝 Comment Results")
                for text, label_text, short in results[:50]:
                    cls = {"pos":"pos","neg":"neg","neu":"neu"}.get(short, "neu")
                    st.markdown(f'''
                    <div class="tweet-card card">
                        <div style="display:flex;gap:10px;align-items:center;">
                            <span class="badge {cls}">{label_text}</span>
                            <div style="font-weight:200;">Comment</div>
                        </div>
                        <div class="tweet-text">{clean_text(text)}</div>
                    </div>
                    ''', unsafe_allow_html=True)

                # Chart
                st.markdown("#### 📊 Sentiment Distribution")
                lab_clean = pd.Series([x[2] for x in results])
                counts = lab_clean.value_counts()
                # --- Percentage Calculation ---
                total = int(counts.sum())

                pos_count = int(counts.get("pos", 0))
                neg_count = int(counts.get("neg", 0))
                neu_count = int(counts.get("neu", 0))

                if total==0:
                    pos_per = neg_per = neu_per = 0.0
                else:
                    pos_per = round((pos_count / total) * 100, 2)
                    neg_per = round((neg_count / total) * 100, 2)
                    neu_per = round((neu_count / total) * 100, 2)
                    colors = {"pos": "#06b6d4", "neu": "#94a3b8", "neg": "#fb7185"}
                    fig, ax = plt.subplots(figsize=(2.5, 1.5))
                    fig.set_dpi(60)
                    ax.bar(counts.index, counts.values,
                            color=[colors.get(i, "#60a5fa") for i in counts.index],
                            edgecolor="black", width=0.55)
                ax.set_xticklabels(counts.index,rotation=0, fontsize=4)
                ax.tick_params(axis='y', labelsize=4)
                ax.set_title("Sentiment Distribution", fontsize=4)
                col1, col2, col3= st.columns([1,2,1])
                with col2:
                    st.pyplot(fig)

#----show percentages---
st.markdown(f"""
<div style='text-align:center; margin-top:10px;'>
    <span style='font-size:14px; font-weight:600; color:#06b6d4;'>Positive: {pos_count} ({pos_per}%)</span><br>
    <span style='font-size:14px; font-weight:600; color:#fb7185;'>Negative: {neg_count} ({neg_per}%)</span><br>
    <span style='font-size:14px; font-weight:600; color:#94a3b8;'>Neutral: {neu_count} ({neu_per}%)</span>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
# -----------------------
# RIGHT: Legend & small info
# -----------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📌 Legend")
    st.markdown('<div style="margin-top:8px;">'
                '<div style="margin-bottom:8px;"><span class="badge pos" style="margin-right:8px;">😊 Positive</span></div>'
                '<div style="margin-bottom:8px;"><span class="badge neu" style="margin-right:8px;">😐 Neutral</span></div>'
                '<div style="margin-bottom:8px;"><span class="badge neg" style="margin-right:8px;">😡 Negative</span></div>'
                '</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    '<div style="text-align:center; margin-top:12px; color:#475569; font-size:16px; font-weight:bold;font-style:italic;">''Your Words, Our Emotional Lens''</div>',unsafe_allow_html=True
    )       
