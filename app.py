import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Topic Modeling App",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .topic-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load semua model dan preprocessor"""
    try:
        lda_model = joblib.load('lda_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        
        with open('topics.pkl', 'rb') as f:
            topics = pickle.load(f)
        
        return lda_model, vectorizer, preprocessor, topics
    except FileNotFoundError:
        st.error("⚠️ Model files tidak ditemukan! Pastikan file model sudah ada di folder yang sama.")
        st.info("📝 File yang diperlukan: lda_model.pkl, vectorizer.pkl, preprocessor.pkl, topics.pkl")
        return None, None, None, None

@st.cache_data
def load_data():
    """Load processed data"""
    try:
        df = pd.read_csv('processed_data.csv')
        return df
    except FileNotFoundError:
        return None

def predict_topic(text, preprocessor, vectorizer, model, n_words=10):
    """Prediksi topik untuk teks baru"""
    # Preprocess
    cleaned = preprocessor.preprocess(text)
    
    # Transform
    vectorized = vectorizer.transform([cleaned])
    
    # Predict
    topic_dist = model.transform(vectorized)[0]
    dominant_topic = topic_dist.argmax()
    
    # Get top words
    feature_names = vectorizer.get_feature_names_out()
    topic = model.components_[dominant_topic]
    top_indices = topic.argsort()[-n_words:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    
    return {
        'dominant_topic': int(dominant_topic),
        'topic_distribution': topic_dist,
        'top_words': top_words,
        'confidence': float(topic_dist[dominant_topic]),
        'cleaned_text': cleaned
    }

def create_wordcloud(words_dict):
    """Membuat word cloud dari dictionary kata"""
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis'
    ).generate_from_frequencies(words_dict)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def plot_topic_distribution(topic_dist, n_topics):
    """Plot distribusi topik"""
    fig, ax = plt.subplots(figsize=(10, 4))
    topics = [f'Topic {i+1}' for i in range(n_topics)]
    colors = plt.cm.viridis(np.linspace(0, 1, n_topics))
    
    bars = ax.barh(topics, topic_dist, color=colors)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title('Topic Distribution', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.1%}', 
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">📚 Topic Modeling Application</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    lda_model, vectorizer, preprocessor, topics = load_models()
    
    if lda_model is None:
        st.stop()
    
    n_topics = lda_model.n_components
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/topic.png", width=100)
        st.title("📊 Navigation")
        
        page = st.radio(
            "Pilih Halaman:",
            ["🏠 Home", "🔍 Predict Topic", "📈 Dataset Analysis", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### 📌 Model Info")
        st.info(f"**Topics:** {n_topics}\n\n**Algorithm:** LDA")
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Developer")
        st.text("NLP Topic Modeling\nStreamlit App")
    
    # Pages
    if page == "🏠 Home":
        home_page(topics, n_topics)
    
    elif page == "🔍 Predict Topic":
        predict_page(preprocessor, vectorizer, lda_model, topics, n_topics)
    
    elif page == "📈 Dataset Analysis":
        analysis_page(n_topics)
    
    elif page == "ℹ️ About":
        about_page()

def home_page(topics, n_topics):
    """Halaman utama - tampilkan topics"""
    st.header("🎯 Discovered Topics")
    st.write("Berikut adalah topik-topik yang ditemukan dari analisis:")
    
    # Display topics in columns
    cols = st.columns(min(n_topics, 3))
    
    for idx, (topic_name, words) in enumerate(topics.items()):
        with cols[idx % len(cols)]:
            st.markdown(f"""
                <div class="topic-card">
                    <h3 style="color: #1f77b4;">📍 {topic_name}</h3>
                    <p style="font-size: 1.1rem;"><strong>Top Keywords:</strong></p>
                    <p>{', '.join(words[:8])}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📊 Quick Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h2 style="color: #1f77b4;">{}</h2>
                <p>Total Topics</p>
            </div>
        """.format(n_topics), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h2 style="color: #2ca02c;">10</h2>
                <p>Words per Topic</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h2 style="color: #ff7f0e;">LDA</h2>
                <p>Algorithm</p>
            </div>
        """, unsafe_allow_html=True)

def predict_page(preprocessor, vectorizer, lda_model, topics, n_topics):
    """Halaman prediksi topik"""
    st.header("🔍 Topic Prediction")
    st.write("Masukkan teks untuk memprediksi topiknya:")
    
    # Input methods
    input_method = st.radio("Pilih metode input:", ["✍️ Text Input", "📄 File Upload"])
    
    if input_method == "✍️ Text Input":
        user_text = st.text_area(
            "Masukkan teks di sini:",
            height=200,
            placeholder="Contoh: Machine learning is transforming the way we analyze data..."
        )
        
        if st.button("🚀 Predict Topic", type="primary"):
            if user_text.strip():
                with st.spinner("Analyzing text..."):
                    result = predict_topic(user_text, preprocessor, vectorizer, lda_model)
                    display_prediction_result(result, topics, n_topics)
            else:
                st.warning("⚠️ Please enter some text first!")
    
    else:  # File Upload
        uploaded_file = st.file_uploader("Upload text file (.txt)", type=['txt'])
        
        if uploaded_file is not None:
            user_text = uploaded_file.read().decode('utf-8')
            st.text_area("Preview:", user_text, height=150)
            
            if st.button("🚀 Predict Topic", type="primary"):
                with st.spinner("Analyzing text..."):
                    result = predict_topic(user_text, preprocessor, vectorizer, lda_model)
                    display_prediction_result(result, topics, n_topics)

def display_prediction_result(result, topics, n_topics):
    """Tampilkan hasil prediksi"""
    st.success("✅ Analysis Complete!")
    
    # Dominant topic
    dominant_topic = result['dominant_topic']
    confidence = result['confidence']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
            <div class="topic-card">
                <h2 style="color: #1f77b4;">Predicted Topic</h2>
                <h1 style="color: #2ca02c; font-size: 3rem;">Topic {dominant_topic + 1}</h1>
                <p style="font-size: 1.2rem;">Confidence: <strong>{confidence:.1%}</strong></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Top Keywords:**")
        st.write(", ".join(result['top_words']))
    
    with col2:
        # Topic distribution plot
        fig = plot_topic_distribution(result['topic_distribution'], n_topics)
        st.pyplot(fig)
    
    # Processed text
    with st.expander("📝 Show Preprocessed Text"):
        st.code(result['cleaned_text'])
    
    # All topics probabilities
    with st.expander("📊 Show All Topic Probabilities"):
        prob_df = pd.DataFrame({
            'Topic': [f'Topic {i+1}' for i in range(n_topics)],
            'Probability': [f"{p:.2%}" for p in result['topic_distribution']]
        })
        st.table(prob_df)

def analysis_page(n_topics):
    """Halaman analisis dataset"""
    st.header("📈 Dataset Analysis")
    
    df = load_data()
    
    if df is None:
        st.warning("⚠️ Processed data tidak ditemukan. Upload file 'processed_data.csv'")
        return
    
    # Dataset overview
    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", len(df))
    with col2:
        st.metric("Total Topics", n_topics)
    with col3:
        avg_words = df['cleaned_text'].str.split().str.len().mean()
        st.metric("Avg Words/Doc", f"{avg_words:.0f}")
    
    # Topic distribution
    if 'dominant_topic' in df.columns:
        st.subheader("📍 Document Distribution Across Topics")
        
        topic_counts = df['dominant_topic'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, n_topics))
        bars = ax.bar(range(n_topics), 
                      [topic_counts.get(i, 0) for i in range(n_topics)], 
                      color=colors)
        ax.set_xlabel('Topic', fontsize=12)
        ax.set_ylabel('Number of Documents', fontsize=12)
        ax.set_title('Distribution of Documents Across Topics', fontsize=14, fontweight='bold')
        ax.set_xticks(range(n_topics))
        ax.set_xticklabels([f'Topic {i+1}' for i in range(n_topics)])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Sample documents
    st.subheader("📄 Sample Documents")
    n_samples = st.slider("Number of samples to display:", 5, 20, 10)
    
    display_df = df[['text', 'cleaned_text']].head(n_samples)
    st.dataframe(display_df, use_container_width=True)
    
    # Download processed data
    st.subheader("💾 Download Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Processed Data (CSV)",
        data=csv,
        file_name="topic_modeling_results.csv",
        mime="text/csv"
    )

def about_page():
    """Halaman about"""
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 Topic Modeling App
    
    Aplikasi ini menggunakan **Latent Dirichlet Allocation (LDA)** untuk melakukan topic modeling 
    pada dokumen teks.
    
    #### 🔧 Features:
    - 📝 **Text Preprocessing**: Cleaning, tokenization, lemmatization
    - 🤖 **LDA Topic Modeling**: Automatic topic discovery
    - 🔍 **Topic Prediction**: Predict topics for new documents
    - 📊 **Visualization**: Interactive charts and word clouds
    - 📈 **Dataset Analysis**: Analyze document distribution
    
    #### 🛠️ Technology Stack:
    - **Streamlit**: Web framework
    - **Scikit-learn**: Machine learning
    - **NLTK**: Natural language processing
    - **Matplotlib/Seaborn**: Visualization
    - **Pandas**: Data manipulation
    
    #### 📚 How It Works:
    1. **Preprocessing**: Text dibersihkan dan diproses (lowercase, remove stopwords, lemmatization)
    2. **Vectorization**: Text dikonversi menjadi numerical representation
    3. **LDA Training**: Model mencari pola kata yang sering muncul bersama
    4. **Topic Discovery**: Kata-kata dikelompokkan menjadi topik-topik
    5. **Prediction**: Model memprediksi topik untuk teks baru
    
    #### 📖 Use Cases:
    - Analisis dokumen research papers
    - Kategorisasi berita
    - Analisis customer reviews
    - Content recommendation
    - Document clustering
    
    ---
    
    ### 🚀 How to Use:
    
    1. **Home**: Lihat topik-topik yang telah ditemukan
    2. **Predict Topic**: Input teks untuk prediksi topik
    3. **Dataset Analysis**: Analisis distribusi dokumen
    
    ---
    
    ### 📞 Contact & Support:
    
    Untuk pertanyaan atau dukungan, silakan hubungi developer.
    
    ---
    
    **Version:** 1.0.0  
    **Last Updated:** December 2025
    """)
    
    # Additional info in expanders
    with st.expander("🔬 Understanding LDA"):
        st.markdown("""
        **Latent Dirichlet Allocation (LDA)** adalah algoritma probabilistic untuk topic modeling.
        
        **Key Concepts:**
        - Setiap dokumen adalah campuran dari beberapa topik
        - Setiap topik adalah distribusi probabilitas atas kata-kata
        - LDA mencari pola kata yang sering muncul bersama
        
        **Assumptions:**
        - Dokumen dengan topik yang sama menggunakan kata-kata yang mirip
        - Setiap dokumen dapat memiliki multiple topics
        """)
    
    with st.expander("📊 Model Evaluation Metrics"):
        st.markdown("""
        **Perplexity**: Mengukur seberapa baik model memprediksi sampel baru. Lower is better.
        
        **Coherence Score**: Mengukur seberapa semantically coherent topik-topik yang dihasilkan.
        
        **Topic Diversity**: Mengukur seberapa berbeda topik satu dengan lainnya.
        """)

if __name__ == "__main__":
    main()
