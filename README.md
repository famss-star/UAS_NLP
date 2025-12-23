# Topic Modeling with Streamlit

Aplikasi web untuk topic modeling menggunakan Latent Dirichlet Allocation (LDA).

## 📋 Prerequisites

- Python 3.8 atau lebih tinggi
- Google Colab (untuk training model)
- Streamlit Cloud atau local environment (untuk deployment)

## 🚀 Quick Start

### 1. Training Model di Google Colab

1. Upload `topic_modeling_colab.ipynb` ke Google Colab
2. Upload dataset Anda atau gunakan data contoh yang disediakan
3. Jalankan semua cell di notebook
4. Download file-file berikut:
   - `lda_model.pkl`
   - `vectorizer.pkl`
   - `preprocessor.pkl`
   - `topics.pkl`
   - `processed_data.csv`

### 2. Setup Local Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (jalankan sekali)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

### 3. Jalankan Aplikasi Streamlit

```bash
# Pastikan file model sudah ada di folder yang sama
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📁 File Structure

```
uas/
├── topic_modeling_colab.ipynb  # Notebook untuk training
├── app.py                      # Aplikasi Streamlit
├── requirements.txt            # Dependencies
├── README.md                   # Dokumentasi
├── lda_model.pkl              # Model LDA (dari Colab)
├── vectorizer.pkl             # Vectorizer (dari Colab)
├── preprocessor.pkl           # Preprocessor (dari Colab)
├── topics.pkl                 # Topics dictionary (dari Colab)
└── processed_data.csv         # Data processed (dari Colab)
```

## 🎯 Features

### 1. Home Page
- Menampilkan topik-topik yang ditemukan
- Quick statistics
- Top keywords per topik

### 2. Predict Topic
- Input teks manual atau upload file
- Prediksi topik dominan
- Visualisasi distribusi probabilitas topik
- Menampilkan preprocessed text

### 3. Dataset Analysis
- Overview dataset
- Distribusi dokumen per topik
- Sample documents
- Download processed data

### 4. About
- Informasi tentang aplikasi
- Penjelasan LDA
- Technology stack
- Use cases

## 🔧 Customization

### Menggunakan Dataset Sendiri

Di `topic_modeling_colab.ipynb`, ganti bagian load data:

```python
# Ganti ini
df = pd.DataFrame({'text': sample_texts})

# Dengan dataset Anda
df = pd.read_csv('your_dataset.csv')
# Pastikan ada kolom 'text' atau sesuaikan nama kolomnya
```

### Mengubah Jumlah Topik

```python
# Di notebook Colab
n_topics = 5  # Ubah sesuai kebutuhan
```

### Custom Preprocessing

Edit class `TextPreprocessor` di notebook untuk menambah:
- Custom stopwords
- Stemming instead of lemmatization
- Bigrams/trigrams
- dll.

## 🌐 Deploy ke Streamlit Cloud

### Via GitHub

1. Push semua file ke GitHub repository
2. Pastikan file model juga di-push (perhatikan ukuran file)
3. Pergi ke [Streamlit Cloud](https://streamlit.io/cloud)
4. Klik "New app"
5. Pilih repository, branch, dan file `app.py`
6. Deploy!

### Via Streamlit Sharing

```bash
# Login ke Streamlit
streamlit login

# Deploy
streamlit deploy app.py
```

## 📊 Model Information

- **Algorithm**: Latent Dirichlet Allocation (LDA)
- **Library**: Scikit-learn + Gensim
- **Preprocessing**: NLTK
- **Vectorization**: CountVectorizer

## 🐛 Troubleshooting

### Error: Model files not found
- Pastikan file `.pkl` ada di folder yang sama dengan `app.py`
- Check nama file sesuai: `lda_model.pkl`, `vectorizer.pkl`, dll.

### NLTK Data Error
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

### Memory Error di Streamlit Cloud
- Reduce ukuran model
- Gunakan fewer features di vectorizer
- Optimize preprocessing

## 📚 Resources

- [LDA Paper](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn LDA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
- [Gensim LDA](https://radimrehurek.com/gensim/models/ldamodel.html)

## 📝 License

MIT License - Feel free to use and modify!

## 👨‍💻 Author

NLP Topic Modeling Project
December 2025
