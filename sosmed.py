import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Judul aplikasi
st.title("Analisis Engagement Media Sosial")
st.write("Aplikasi sederhana untuk menganalisis distribusi *Engagement Rate* dari data media sosial menggunakan Streamlit")

# Upload File CSV
upload_file = st.file_uploader("Unggah file CSV Anda", type=["csv"])
if upload_file is not None:
    # Baca data
    data = pd.read_csv(upload_file)
    st.write("Data yang diunggah:")
    st.dataframe(data.head())

    # Pastikan kolom yang dibutuhkan ada
    # --- PERBAIKAN DI SINI ---
    # Menyesuaikan nama kolom dengan file CSV (likes, comments, followers)
    required_col = {'likes', 'comments', 'followers'}
    
    if required_col.issubset(data.columns):
    # -------------------------

        # Hitung Engagement Rate
        # Tambahkan pemeriksaan untuk menghindari pembagian dengan nol jika followers = 0
        # --- PERBAIKAN DI SINI ---
        if data['followers'].min() == 0:
            st.warning("Peringatan: Terdapat data 'followers' dengan nilai 0. Baris tersebut akan diabaikan dalam perhitungan ER.")
            # Filter data untuk menghindari ZeroDivisionError
            # --- PERBAIKAN DI SINI ---
            data_valid = data[data['followers'] > 0].copy()
        else:
            data_valid = data.copy()

        # --- PERBAIKAN DI SINI ---
        # Menggunakan nama kolom yang benar untuk kalkulasi
        data_valid['engagement_rate'] = (data_valid['likes'] + data_valid['comments']) / data_valid['followers'] * 100

        # Tampilkan statistik dasar
        st.subheader("Statistik Deskriptif Engagement Rate:")
        st.write(data_valid['engagement_rate'].describe())

        # Plot distribusi Engagement Rate
        fig, ax = plt.subplots()
        ax.hist(data_valid['engagement_rate'], bins=30, color='skyblue', edgecolor='black')
        ax.set_title('Distribusi Engagement Rate')
        ax.set_xlabel('Engagement Rate (%)')
        ax.set_ylabel('Frekuensi')
        st.pyplot(fig)

        # Tambahan: tampilkan rata rata engagement
        avg_eng = data_valid['engagement_rate'].mean()
        st.metric(label="Rata-rata Engagement Rate", value=f"{avg_eng:.2f}%")
    else:
        # --- PERBAIKAN DI SINI ---
        # Memperbarui pesan error agar lebih jelas
        st.error(f"File CSV harus mengandung kolom: {', '.join(required_col)}")
        st.info(f"Kolom yang terdeteksi di file Anda: {', '.join(data.columns)}")
else:
    st.info("Silakan unggah file CSV untuk memulai analisis.")

