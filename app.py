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

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data (only once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Page configuration
st.set_page_config(
    page_title="Aplikasi Pemodelan Topik",
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

# TextPreprocessor class (must match the one used in training)
class TextPreprocessor:
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Membersihkan teks dari karakter khusus"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenisasi dan lemmatisasi"""
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                 if word not in self.stop_words and len(word) > 2]
        return tokens
    
    def preprocess(self, text):
        """Pipeline preprocessing lengkap"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        return ' '.join(tokens)

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
        st.error("Model tidak ditemukan. Pastikan file model ada di folder yang sama.")
        st.info("File diperlukan: lda_model.pkl, vectorizer.pkl, preprocessor.pkl, topics.pkl")
        return None, None, None, None

@st.cache_data
def load_data():
    """Load processed data"""
    try:
        df = pd.read_csv('processed_data.csv')
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None

def predict_topic(text, preprocessor, vectorizer, model, n_words=10):
    """Prediksi topik untuk teks baru"""
    # Preprocess
    cleaned = preprocessor.preprocess(text)
    tokens = cleaned.split()
    total_tokens = len(tokens)
    
    # Transform
    vectorized = vectorizer.transform([cleaned])
    vocab_hits = int(vectorized.nnz)
    vocab_coverage = float(vocab_hits / total_tokens) if total_tokens > 0 else 0.0
    
    # Predict
    topic_dist = model.transform(vectorized)[0]
    dominant_topic = topic_dist.argmax()
    
    # Get top words (dari model untuk topik dominan)
    feature_names = vectorizer.get_feature_names_out()
    topic = model.components_[dominant_topic]
    top_indices = topic.argsort()[-n_words:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    
    return {
        'dominant_topic': int(dominant_topic),
        'topic_distribution': topic_dist,
        'top_words': top_words,
        'confidence': float(topic_dist[dominant_topic]),
        'cleaned_text': cleaned,
        'vocab_hits': vocab_hits,
        'total_tokens': total_tokens,
        'vocab_coverage': vocab_coverage
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
    topics = [f'Topik {i+1}' for i in range(n_topics)]
    colors = plt.cm.viridis(np.linspace(0, 1, n_topics))
    
    bars = ax.barh(topics, topic_dist, color=colors)
    ax.set_xlabel('Probabilitas', fontsize=12)
    ax.set_title('Distribusi Topik', fontsize=14, fontweight='bold')
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
    st.markdown('<h1 class="main-header">Aplikasi Pemodelan Topik</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    lda_model, vectorizer, preprocessor, topics = load_models()
    
    if lda_model is None:
        st.stop()
    
    n_topics = lda_model.n_components
    
    # Sidebar
    with st.sidebar:
        st.title("Navigasi")
        
        page = st.radio(
            "Pilih Halaman:",
            ["Beranda", "Prediksi Topik", "Analisis Dataset", "Tentang"]
        )
        
        st.markdown("---")
        st.markdown("### Informasi Model")
        st.info(f"**Topik:** {n_topics}\n\n**Algoritme:** LDA")
        
        st.markdown("---")
        st.markdown("### Pengembang")
        st.text("NLP Topic Modeling\nAplikasi Streamlit")
    
    # Pages
    if page == "Beranda":
        home_page(topics, n_topics)
    
    elif page == "Prediksi Topik":
        predict_page(preprocessor, vectorizer, lda_model, topics, n_topics)
    
    elif page == "Analisis Dataset":
        analysis_page(n_topics)
    
    elif page == "Tentang":
        about_page()

def home_page(topics, n_topics):
    """Halaman utama - tampilkan topik"""
    st.header("Topik yang Ditemukan")
    st.write("Berikut adalah topik-topik yang ditemukan dari analisis:")
    
    # Display topics in columns
    cols = st.columns(min(n_topics, 3))
    
    for idx, (topic_name, words) in enumerate(topics.items()):
        with cols[idx % len(cols)]:
            st.markdown(f"""
                <div class="topic-card">
                    <h3 style="color: #1f77b4;">{topic_name}</h3>
                    <p style="font-size: 1.1rem;"><strong>Kata Kunci Utama:</strong></p>
                    <p>{', '.join(words[:8])}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("Statistik Singkat")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h2 style="color: #1f77b4;">{}</h2>
                <p>Total Topik</p>
            </div>
        """.format(n_topics), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h2 style="color: #2ca02c;">10</h2>
                <p>Kata per Topik</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h2 style="color: #ff7f0e;">LDA</h2>
                <p>Algoritme</p>
            </div>
        """, unsafe_allow_html=True)

def predict_page(preprocessor, vectorizer, lda_model, topics, n_topics):
    """Halaman prediksi topik"""
    st.header("Prediksi Topik")
    st.write("Masukkan teks untuk memprediksi topiknya:")
    
    # Metode input
    input_method = st.radio("Pilih metode input:", ["Input Teks", "Unggah Berkas"])
    
    if input_method == "Input Teks":
        user_text = st.text_area(
            "Masukkan teks di sini:",
            height=200,
            placeholder="Contoh: Pembelajaran mesin mengubah cara kita menganalisis data..."
        )
        
        if st.button("Prediksi Topik", type="primary"):
            if user_text.strip():
                with st.spinner("Menganalisis teks..."):
                    result = predict_topic(user_text, preprocessor, vectorizer, lda_model)
                    display_prediction_result(result, topics, n_topics)
            else:
                st.warning("Silakan masukkan teks terlebih dahulu!")
    
    else:  # Unggah Berkas
        uploaded_file = st.file_uploader("Unggah berkas teks (.txt)", type=['txt'])
        
        if uploaded_file is not None:
            user_text = uploaded_file.read().decode('utf-8')
            st.text_area("Pratinjau:", user_text, height=150)
            
            if st.button("Prediksi Topik", type="primary"):
                if user_text.strip():
                    with st.spinner("Menganalisis teks..."):
                        result = predict_topic(user_text, preprocessor, vectorizer, lda_model)
                        display_prediction_result(result, topics, n_topics)
                else:
                    st.warning("Berkas kosong, silakan unggah berkas dengan konten teks.")

def display_prediction_result(result, topics, n_topics):
    """Tampilkan hasil prediksi"""
    st.success("Analisis selesai!")
    
    # Topik dominan
    dominant_topic = result['dominant_topic']
    confidence = result['confidence']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
            <div class="topic-card">
                <h2 style="color: #1f77b4;">Topik Terprediksi</h2>
                <h1 style="color: #2ca02c; font-size: 3rem;">Topik {dominant_topic + 1}</h1>
                <p style="font-size: 1.2rem;">Kepercayaan: <strong>{confidence:.1%}</strong></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Kata Kunci Utama:**")
        st.write(", ".join(result['top_words']))
    
    with col2:
        # Plot distribusi topik
        fig = plot_topic_distribution(result['topic_distribution'], n_topics)
        st.pyplot(fig)
    
    # Teks yang telah diproses
    with st.expander("Tampilkan Teks yang Diproses"):
        st.code(result['cleaned_text'])
    
    # Kecocokan kosakata terhadap model
    with st.expander("Kecocokan Kosakata terhadap Model"):
        st.write(f"Token cocok: {result['vocab_hits']} dari {result['total_tokens']} token")
        st.write(f"Cakupan kosakata: {result['vocab_coverage']:.0%}")
        if result['vocab_hits'] == 0:
            st.warning("Tidak ada kata dari teks yang dikenali oleh model. Pastikan bahasa/templat preprocessing sesuai dengan data pelatihan.")

    # Probabilitas semua topik
    with st.expander("Tampilkan Probabilitas Semua Topik"):
        prob_df = pd.DataFrame({
            'Topik': [f'Topik {i+1}' for i in range(n_topics)],
            'Probabilitas': [f"{p:.2%}" for p in result['topic_distribution']]
        })
        st.table(prob_df)

def analysis_page(n_topics):
    """Halaman analisis dataset"""
    st.header("Analisis Dataset")
    
    df = load_data()
    
    if df is None:
        st.warning("Processed data tidak tersedia. Silakan latih model terlebih dahulu di Colab dan unggah `processed_data.csv`.")
        st.info("File yang dibutuhkan: `processed_data.csv` dengan kolom minimal: `cleaned_text`, `dominant_topic`")
        return
    
    # Check required columns
    required_cols = ['cleaned_text']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ File `processed_data.csv` harus memiliki kolom: {', '.join(required_cols)}")
        st.info(f"📋 Kolom yang tersedia: {', '.join(df.columns.tolist())}")
        return
    
    # Dataset overview
    st.subheader("Ringkasan Dataset")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Dokumen", len(df))
    with col2:
        st.metric("Total Topik", n_topics)
    with col3:
        avg_words = df['cleaned_text'].str.split().str.len().mean()
        st.metric("Rata-rata Kata/Dok", f"{avg_words:.0f}")
    
    # Topic distribution
    if 'dominant_topic' in df.columns:
        st.subheader("Distribusi Dokumen per Topik")
        
        topic_counts = df['dominant_topic'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, n_topics))
        bars = ax.bar(range(n_topics), 
                      [topic_counts.get(i, 0) for i in range(n_topics)], 
                      color=colors)
        ax.set_xlabel('Topik', fontsize=12)
        ax.set_ylabel('Jumlah Dokumen', fontsize=12)
        ax.set_title('Distribusi Dokumen per Topik', fontsize=14, fontweight='bold')
        ax.set_xticks(range(n_topics))
        ax.set_xticklabels([f'Topik {i+1}' for i in range(n_topics)])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Sample documents
    st.subheader("Contoh Dokumen")
    n_samples = st.slider("Jumlah sampel yang ditampilkan:", 5, 20, 10)
    
    # Show available columns
    display_cols = [col for col in ['text', 'cleaned_text'] if col in df.columns]
    if display_cols:
        display_df = df[display_cols].head(n_samples)
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("Kolom 'text' atau 'cleaned_text' tidak tersedia dalam dataset.")
    
    # Download processed data
    st.subheader("Unduh Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Unduh Data yang Diproses (CSV)",
        data=csv,
        file_name="hasil_pemodelan_topik.csv",
        mime="text/csv"
    )

def about_page():
    """Halaman tentang"""
    st.header("Tentang Aplikasi Ini")
    
    st.markdown("""
    ### Aplikasi Pemodelan Topik
    
    Aplikasi ini menggunakan **Latent Dirichlet Allocation (LDA)** untuk melakukan pemodelan topik 
    pada dokumen teks.
    
    #### Fitur:
    - 📝 **Pemrosesan Teks**: Pembersihan, tokenisasi, lemmatisasi
    - 🤖 **Pemodelan Topik LDA**: Penemuan topik secara otomatis
    - 🔍 **Prediksi Topik**: Prediksi topik untuk dokumen baru
    - 📊 **Visualisasi**: Grafik interaktif dan word cloud
    - 📈 **Analisis Dataset**: Analisis distribusi dokumen
    
    #### Teknologi:
    - **Streamlit**: Kerangka kerja web
    - **Scikit-learn**: Pembelajaran mesin
    - **NLTK**: Pemrosesan bahasa alami
    - **Matplotlib/Seaborn**: Visualisasi
    - **Pandas**: Manipulasi data
    
    #### Cara Kerja:
    1. **Preprocessing**: Teks dibersihkan dan diproses (lowercase, hapus stopwords, lemmatisasi)
    2. **Vectorization**: Teks dikonversi menjadi representasi numerik
    3. **Pelatihan LDA**: Model mencari pola kata yang sering muncul bersama
    4. **Penemuan Topik**: Kata-kata dikelompokkan menjadi topik-topik
    5. **Prediksi**: Model memprediksi topik untuk teks baru
    
    #### Contoh Penggunaan:
    - Analisis paper penelitian
    - Kategorisasi berita
    - Analisis ulasan pelanggan
    - Rekomendasi konten
    - Pengelompokan dokumen
    
    ---
    
    ### Cara Menggunakan:
    
    1. **Beranda**: Lihat topik-topik yang telah ditemukan
    2. **Prediksi Topik**: Masukkan teks untuk prediksi topik
    3. **Analisis Dataset**: Analisis distribusi dokumen
    
    ---
    
    ### Kontak & Dukungan:
    
    Untuk pertanyaan atau dukungan, silakan hubungi pengembang.
    
    ---
    
    **Versi:** 1.0.0  
    **Terakhir Diperbarui:** Desember 2025
    """)
    
    # Informasi tambahan dalam ekspander
    with st.expander("Memahami LDA"):
        st.markdown("""
        **Latent Dirichlet Allocation (LDA)** adalah algoritma probabilistik untuk pemodelan topik.
        
        **Konsep Utama:**
        - Setiap dokumen adalah campuran dari beberapa topik
        - Setiap topik adalah distribusi probabilitas atas kata-kata
        - LDA mencari pola kata yang sering muncul bersama
        
        **Asumsi:**
        - Dokumen dengan topik yang sama menggunakan kata-kata yang mirip
        - Setiap dokumen dapat memiliki banyak topik
        """)
    
    with st.expander("Metrik Evaluasi Model"):
        st.markdown("""
        **Perplexity**: Mengukur seberapa baik model memprediksi sampel baru (semakin rendah semakin baik).
        
        **Coherence Score**: Mengukur seberapa koheren secara semantik topik-topik yang dihasilkan.
        
        **Topic Diversity**: Mengukur seberapa berbeda topik satu dengan lainnya.
        """)

if __name__ == "__main__":
    main()
