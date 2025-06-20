import streamlit as st
import pandas as pd
import joblib

# Load model
with open("model_graduation.pkl", "rb") as file:
    nb = joblib.load(file)

st.title("Prediksi Kategori Masa Studi Mahasiswa")

st.markdown("""
Masukkan informasi berikut untuk memprediksi apakah mahasiswa akan **Ontime** atau **Late** lulus.
""")

# Input dari pengguna
new_ACT = st.number_input("Masukkan nilai ACT composite score:", min_value=0.0, step=0.1)
new_SAT = st.number_input("Masukkan nilai SAT total score:", min_value=0.0, step=1.0)
new_GPA = st.number_input("Masukkan nilai rata-rata SMA:", min_value=0.0, max_value=4.0, step=0.01)
new_income = st.number_input("Masukkan pendapatan orang tua:", min_value=0.0, step=100.0)
new_education = st.text_input("Masukkan tingkat pendidikan orang tua (numerik):")

if st.button("Prediksi"):
    try:
        # Validasi input pendidikan numerik
        new_education_numeric = float(new_education)

        # Buat dataframe
        new_data_df = pd.DataFrame([[
            new_ACT, new_SAT, new_GPA, new_income, new_education_numeric
        ]], columns=[
            'ACT composite score',
            'SAT total score',
            'high school gpa',
            'parental income',
            'parent_edu_numerical'
        ])

        # Prediksi
        predicted = nb.predict(new_data_df)

        # Mapping hasil prediksi ke label
        label_mapping = {0: "Ontime", 1: "Late"}
        result_label = label_mapping.get(predicted[0], "Tidak diketahui")

        st.success(f"🎓 Prediksi kategori masa studi adalah: **{result_label}**")

    except ValueError:
        st.error("Input pendidikan orang tua harus berupa angka.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
