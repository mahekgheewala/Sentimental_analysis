import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import plotly.graph_objects as go

# Define EnhancedSentimentAnalyzer
class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression()
    
    def train_model(self, df, text_col, label_col):
        X = self.vectorizer.fit_transform(df[text_col])
        y = df[label_col]
        self.model.fit(X, y)

        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')

        return {
            'test_accuracy': acc,
            'f1_test': f1,
            'train_samples': len(df),
            'feature_count': X.shape[1]
        }

    def predict_comprehensive(self, text, detailed=True):
        X = self.vectorizer.transform([text])
        prob = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]
        label = int(pred)

        linguistic_features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': text.count('.') + 1,
            'positive_words': sum(word in text.lower() for word in ['good', 'great', 'excellent', 'love', 'happy']),
            'negative_words': sum(word in text.lower() for word in ['bad', 'terrible', 'hate', 'worst', 'sad']),
            'sentiment_balance': (text.lower().count('good') + 1) / (text.lower().count('bad') + 1)
        }

        return {
            'sentiment_label': label,
            'confidence': max(prob),
            'probabilities': {str(i): float(p) for i, p in enumerate(prob) if i-1 in [-1, 0, 1]},
            'linguistic_features': linguistic_features
        }

# Set page configuration
st.set_page_config(page_title="🎭 Enhanced Sentiment Analyzer", page_icon="🎭", layout="wide")

# Custom CSS styling
st.markdown("""
<style>
body, .main {
    background-color: #1e1e2f;
    color: #e0e0e0;
}
.main-header {
    font-size: 3rem !important;
    text-align: center;
    color: #00bcd4;
    margin-bottom: 2rem;
}
.sentiment-positive {
    background-color: #223d26;
    color: #a5d6a7;
    border-left: 5px solid #66bb6a;
    padding: 1rem;
    border-radius: 5px;
}
.sentiment-negative {
    background-color: #3b2225;
    color: #ef9a9a;
    border-left: 5px solid #ef5350;
    padding: 1rem;
    border-radius: 5px;
}
.sentiment-neutral {
    background-color: #333327;
    color: #fff59d;
    border-left: 5px solid #fdd835;
    padding: 1rem;
    border-radius: 5px;
}
.feature-box {
    background-color: #2e2e3a;
    color: #ffffff;
    border: 1px solid #44475a;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.sidebar .sidebar-content {
    background-color: #2e2e3a;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_train_model():
    try:
        analyzer = EnhancedSentimentAnalyzer()
        data = pd.read_csv("/content/Reddit_Data.csv")
        data.rename(columns={'clean_comment': 'comment'}, inplace=True)

        data = data.dropna(subset=['comment'])
        data['comment'] = data['comment'].astype(str)

        pos_data = data[data['category'] == 1]
        neu_data = data[data['category'] == 0]
        neg_data = data[data['category'] == -1]

        min_samples = min(len(pos_data), len(neu_data), len(neg_data))
        pos_balanced = resample(pos_data, n_samples=min_samples, random_state=42)
        neu_balanced = resample(neu_data, n_samples=min_samples, random_state=42)
        neg_balanced = resample(neg_data, n_samples=min_samples, random_state=42)

        balanced_data = pd.concat([pos_balanced, neu_balanced, neg_balanced]).sample(frac=1).reset_index(drop=True)

        training_stats = analyzer.train_model(balanced_data, 'comment', 'category')
        return analyzer, training_stats, balanced_data
    except Exception as e:
        st.error(f"Error loading/training model: {str(e)}")
        return None, None, None

def predict_sentiment_with_details(analyzer, user_text):
    try:
        result = analyzer.predict_comprehensive(user_text, detailed=True)
        sentiment_map = {
            '-1': {'name': 'Negative', 'emoji': '❌', 'color': 'negative'},
            '0': {'name': 'Neutral', 'emoji': '⚖️', 'color': 'neutral'},
            '1': {'name': 'Positive', 'emoji': '✅', 'color': 'positive'}
        }
        label = str(result['sentiment_label'])
        sentiment = sentiment_map.get(label, {'name': 'Unknown', 'emoji': '❓', 'color': 'neutral'})
        result['sentiment'] = sentiment
        result['original_text'] = user_text
        return result
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def create_probability_chart(probabilities):
    labels = []
    values = []
    colors = []

    color_map = {
        '-1': '#ef5350',
        '0': '#fdd835',
        '1': '#66bb6a'
    }
    name_map = {
        '-1': 'Negative',
        '0': 'Neutral',
        '1': 'Positive'
    }

    for label, prob in probabilities.items():
        labels.append(name_map.get(str(label), label))
        values.append(prob)
        colors.append(color_map.get(str(label), '#90a4ae'))

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=colors),
            text=[f"{p*100:.1f}%" for p in values],
            textposition='auto',
            hoverinfo="x+y"
        )
    ])

    fig.update_layout(
        plot_bgcolor='#1e1e2f',
        paper_bgcolor='#1e1e2f',
        font=dict(color='#e0e0e0'),
        title='📊 Sentiment Probability Distribution',
        xaxis_title='Sentiment',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1]),
        margin=dict(t=50, l=40, r=40, b=40)
    )
    return fig

def main():
    st.markdown('<h1 class="main-header">🎭 Enhanced Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### AI-powered sentiment analysis with deep linguistic insights")

    with st.sidebar:
        st.header("📊 About This App")
        st.markdown("This tool analyzes sentiment using Logistic Regression + TF-IDF.")
        st.markdown("It provides linguistic features and confidence probabilities.")

    analyzer, training_stats, _ = load_and_train_model()
    if not analyzer:
        return

    st.header("💬 Enter Text to Analyze")
    user_input = st.text_area("Write your message or comment below:", height=100)

    if st.button("🔍 Analyze Sentiment") and user_input.strip():
        with st.spinner("Analyzing..."):
            result = predict_sentiment_with_details(analyzer, user_input)
            if result:
                sentiment = result['sentiment']
                confidence = result['confidence']
                features = result['linguistic_features']
                sentiment_class = f"sentiment-{sentiment['color']}"

                st.markdown(f"""
                <div class="{sentiment_class}">
                    <h3>{sentiment['emoji']} Predicted Sentiment: {sentiment['name']}</h3>
                    <p><strong>Confidence Score:</strong> {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📈 Sentiment Graph")
                    st.plotly_chart(create_probability_chart(result['probabilities']), use_container_width=True)
                with col2:
                    st.subheader("🧠 Linguistic Features")
                    st.markdown(f"""
                    <div class="feature-box">
                        <strong>Length:</strong> {features['text_length']} chars<br>
                        <strong>Words:</strong> {features['word_count']}<br>
                        <strong>Sentences:</strong> {features['sentence_count']}<br>
                        <strong>Positive Words:</strong> {features['positive_words']}<br>
                        <strong>Negative Words:</strong> {features['negative_words']}<br>
                        <strong>Sentiment Balance:</strong> {features['sentiment_balance']:.2f}
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
