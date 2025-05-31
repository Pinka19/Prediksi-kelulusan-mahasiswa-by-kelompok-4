import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load model ===
model = joblib.load("xgb_model.pkl")

# === Fitur input ===
features = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification', 'Nacionality',
    "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation",
    'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date',
    'Gender', 'Scholarship holder', 'Age at enrollment', 'International',
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

dropdown_features = {
    'Marital status': list(range(1, 6)),
    'Application mode': list(range(1, 18)),
    'Application order': list(range(1, 10)),
    'Course': list(range(1, 18)),
    'Daytime/evening attendance': [0, 1],
    'Previous qualification': list(range(1, 22)),
    'Nacionality': list(range(1, 22)),
    "Mother's qualification": list(range(1, 30)),
    "Father's qualification": list(range(1, 30)),
    "Mother's occupation": list(range(1, 30)),
    "Father's occupation": list(range(1, 30)),
    'Displaced': [0, 1],
    'Educational special needs': [0, 1],
    'Debtor': [0, 1],
    'Tuition fees up to date': [0, 1],
    'Gender': [0, 1],
    'Scholarship holder': [0, 1],
    'International': [0, 1],
}

# === Judul Web ===
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="wide")
st.title("🎓 Prediksi Kelulusan Mahasiswa")

# === Tab untuk Input Individu dan File CSV ===
tab1, tab2 = st.tabs(["🧑‍🎓 Prediksi Individu", "📁 Prediksi dari CSV"])

# === Tab 1: Prediksi Individu ===
with tab1:
    st.subheader("Masukkan data mahasiswa:")
    input_data = {}

    for feature in features:
        if feature in dropdown_features:
            input_data[feature] = st.selectbox(feature, dropdown_features[feature], key=feature)
        else:
            input_data[feature] = st.number_input(feature, key=feature)

    if st.button("🔍 Prediksi"):
        try:
            X = np.array([input_data[f] for f in features]).reshape(1, -1)
            pred = model.predict(X)[0]
            hasil = "✅ LULUS" if pred == 1 else "❌ DROPOUT"
            st.success(f"Hasil Prediksi: **{hasil}**")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# === Tab 2: Prediksi CSV ===
with tab2:
    st.subheader("Unggah file CSV")
    file = st.file_uploader("Pilih file CSV", type=["csv"])

    if file:
        try:
            df = pd.read_csv(file)
            missing = [f for f in features if f not in df.columns]
            if missing:
                st.error(f"Kolom hilang dalam file CSV: {missing}")
            else:
                df["Prediksi"] = model.predict(df[features])
                df["Prediksi"] = df["Prediksi"].apply(lambda x: "LULUS" if x == 1 else "DROPOUT")
                st.success("Prediksi selesai!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Hasil", csv, "hasil_prediksi.csv", "text/csv")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
