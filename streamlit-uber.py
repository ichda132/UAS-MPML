import streamlit as st
import joblib
import numpy as np

# Muat model dan scaler yang sudah disimpan
model = joblib.load('random_forest_model.sav')
scaler = joblib.load('scaler.sav')

# Fungsi untuk melakukan prediksi
def predict(features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)  # Terapkan standarisasi
    prediction = model.predict(features_scaled)
    return prediction

# Streamlit UI
st.title('Prediksi Tarif Uber')
st.write('Masukkan fitur untuk prediksi tarif uber')

# Input fitur dari pengguna
distance = st.number_input('Distance (km)')
passenger_count = st.number_input('Passenger Count', min_value=1, max_value=10, step=1)
day = st.number_input('Day', min_value=1, max_value=31, step=1)
month = st.number_input('Month', min_value=1, max_value=12, step=1)
year = st.number_input('Year', min_value=2009, max_value=2015, step=1)
hour = st.number_input('Hour', min_value=0, max_value=23, step=1)
dayofweek = st.number_input('Day of Week', min_value=0, max_value=6, step=1)

# Melakukan prediksi saat tombol ditekan
if st.button('Predict'):
    features = [passenger_count, hour, day, month, year, dayofweek, distance]
    prediction = predict(features)
    st.write(f'Tarif Prediksi: ${prediction[0]:.2f}')
