import streamlit as st
import joblib
import pandas as pd
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load model & encoder
model_sentimen = joblib.load("model_sentiment.pkl")  # LinearSVC
model_emosi = joblib.load("model_emosi.pkl")          # LogisticRegression
tfidf = joblib.load("tfidf_vectorizer.pkl")
label_sentimen = joblib.load("label_encoder_sentiment.pkl")
label_emosi = joblib.load("label_encoder_emosi.pkl")

# Inisialisasi stemmer
stemmer = StemmerFactory().create_stemmer()

# Fungsi membersihkan teks
def bersihkan(teks):
    teks = teks.lower()
    teks = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", teks)
    teks = re.sub(r"\d+", " ", teks)
    teks = teks.translate(str.maketrans("", "", string.punctuation))
    teks = re.sub(r"(.)\1{2,}", r"\1", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    kata = teks.split()
    kata = [stemmer.stem(w) for w in kata]
    return " ".join(kata)

# Tampilan utama Streamlit
st.title("Klasifikasi Emosi & Sentimen Review Pelanggan")

menu = st.sidebar.selectbox("Menu", ["Form Ulasan", "Upload File CSV"])

# ========================== Form Input Teks ==========================
if menu == "Form Ulasan":
    ulasan = st.text_area("Masukkan Ulasan Pelanggan:")
    if st.button("Prediksi"):
        teks_bersih = bersihkan(ulasan)
        fitur = tfidf.transform([teks_bersih])

        pred_sent = model_sentimen.predict(fitur)[0]
        pred_emo = model_emosi.predict(fitur)[0]

        try:
            hasil_sent = label_sentimen.inverse_transform([pred_sent])[0]
        except ValueError:
            hasil_sent = f"Label tidak dikenali: {pred_sent}"

        try:
            hasil_emo = label_emosi.inverse_transform([pred_emo])[0]
        except ValueError:
            hasil_emo = f"Label tidak dikenali: {pred_emo}"

        st.subheader("Hasil Prediksi:")
        st.write("Teks dibersihkan:", teks_bersih)
        st.success(f"**Sentimen:** {hasil_sent}")
        st.success(f"**Emosi:** {hasil_emo}")

# ========================== Upload File CSV ==========================
elif menu == "Upload File CSV":
    file = st.file_uploader("Unggah File CSV", type=['csv'])
    if file:
        df = pd.read_csv(file)

        if 'Customer Review' not in df.columns:
            st.error("File harus mengandung kolom 'Customer Review'")
        else:
            df['clean'] = df['Customer Review'].astype(str).apply(bersihkan)
            fitur = tfidf.transform(df['clean'])

            pred_sent = model_sentimen.predict(fitur)
            pred_emo = model_emosi.predict(fitur)

            try:
                df['Sentimen'] = label_sentimen.inverse_transform(pred_sent)
            except ValueError:
                df['Sentimen'] = pred_sent

            try:
                df['Emosi'] = label_emosi.inverse_transform(pred_emo)
            except ValueError:
                df['Emosi'] = pred_emo

            st.subheader("Data Hasil Prediksi:")
            st.dataframe(df[['Customer Review', 'Sentimen', 'Emosi']])

            st.subheader("Grafik Sentimen")
            st.bar_chart(df['Sentimen'].value_counts())

            st.subheader("Grafik Emosi")
            st.bar_chart(df['Emosi'].value_counts())
