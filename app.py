import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Prediksi Harga Mobil",
    layout="wide"
)

# ==============================
# LOAD DAN TRAIN MODEL
# ==============================

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("Car_sales.xls")

    df_clean = df.copy()

    # Hapus data jika target harga kosong
    df_clean = df_clean.dropna(subset=["Price_in_thousands"])

    fitur_numerik = [
        "Engine_size",
        "Horsepower",
        "Wheelbase",
        "Width",
        "Length",
        "Curb_weight",
        "Fuel_capacity",
        "Fuel_efficiency"
    ]

    fitur_kategori = ["Vehicle_type"]

    # Isi missing value pada fitur
    for col in fitur_numerik:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in fitur_kategori:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    X = df_clean[fitur_kategori + fitur_numerik]
    y = df_clean["Price_in_thousands"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), fitur_kategori),
            ("num", "passthrough", fitur_numerik)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]
    )

    model.fit(X_train, y_train)

    # Rekomendasi spesifikasi dari 10 mobil terlaris
    top10_sales = df_clean.sort_values(
        by="Sales_in_thousands",
        ascending=False
    ).head(10)

    default_values = {
        "Vehicle_type": top10_sales["Vehicle_type"].mode()[0],
        "Engine_size": float(top10_sales["Engine_size"].median()),
        "Horsepower": float(top10_sales["Horsepower"].median()),
        "Wheelbase": float(top10_sales["Wheelbase"].median()),
        "Width": float(top10_sales["Width"].median()),
        "Length": float(top10_sales["Length"].median()),
        "Curb_weight": float(top10_sales["Curb_weight"].median()),
        "Fuel_capacity": float(top10_sales["Fuel_capacity"].median()),
        "Fuel_efficiency": float(top10_sales["Fuel_efficiency"].median())
    }

    vehicle_type_list = sorted(df_clean["Vehicle_type"].unique().tolist())

    return model, default_values, vehicle_type_list


model, default, vehicle_type_list = load_and_train_model()

# ==============================
# HEADER
# ==============================

st.title("Sistem Prediksi Harga Mobil")

st.write(
    "Aplikasi ini digunakan untuk memperkirakan harga mobil berdasarkan "
    "spesifikasi kendaraan yang dimasukkan oleh pengguna."
)

st.divider()

# ==============================
# SIDEBAR
# ==============================

with st.sidebar:
    st.header("Panduan Penggunaan")
    st.write("1. Pilih tipe kendaraan.")
    st.write("2. Atur spesifikasi mobil.")
    st.write("3. Klik tombol prediksi.")
    st.write("4. Lihat hasil estimasi harga.")

    st.divider()

    st.header("Informasi Model")
    st.write("Algoritma: Linear Regression")
    st.write("Target: Harga mobil")
    st.write("Satuan harga: ribu dolar")

# ==============================
# RINGKASAN ATAS
# ==============================

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Model", "Linear Regression")

with col_b:
    st.metric("Output", "Harga Mobil")

with col_c:
    st.metric("Satuan", "Ribu Dolar")

st.divider()

# ==============================
# TAB UTAMA
# ==============================

tab_prediksi, tab_variabel, tab_sistem = st.tabs(
    ["Prediksi Harga", "Keterangan Variabel", "Tentang Sistem"]
)

# ==============================
# TAB 1 - PREDIKSI
# ==============================

with tab_prediksi:
    st.subheader("Prediksi Harga Mobil")

    st.info(
        "Nilai awal pada form menggunakan rekomendasi dari data mobil dengan penjualan tinggi. "
        "Pengguna dapat mengubah nilai sesuai spesifikasi mobil yang ingin diprediksi."
    )

    col_input, col_output = st.columns([1.25, 1])

    with col_input:
        st.markdown("### Input Spesifikasi")

        with st.form("form_prediksi"):
            vehicle_type = st.selectbox(
                "Tipe Kendaraan",
                vehicle_type_list,
                index=vehicle_type_list.index(default["Vehicle_type"])
                if default["Vehicle_type"] in vehicle_type_list else 0
            )

            st.markdown("#### Spesifikasi Mesin")

            engine_size = st.slider(
                "Engine Size",
                min_value=0.0,
                max_value=10.0,
                value=default["Engine_size"],
                step=0.1
            )

            horsepower = st.slider(
                "Horsepower",
                min_value=0.0,
                max_value=500.0,
                value=default["Horsepower"],
                step=1.0
            )

            st.markdown("#### Dimensi Kendaraan")

            wheelbase = st.slider(
                "Wheelbase",
                min_value=80.0,
                max_value=140.0,
                value=default["Wheelbase"],
                step=0.1
            )

            width = st.slider(
                "Width",
                min_value=50.0,
                max_value=90.0,
                value=default["Width"],
                step=0.1
            )

            length = st.slider(
                "Length",
                min_value=120.0,
                max_value=250.0,
                value=default["Length"],
                step=0.1
            )

            curb_weight = st.slider(
                "Curb Weight",
                min_value=1.0,
                max_value=6.0,
                value=default["Curb_weight"],
                step=0.1
            )

            st.markdown("#### Bahan Bakar")

            fuel_capacity = st.slider(
                "Fuel Capacity",
                min_value=5.0,
                max_value=40.0,
                value=default["Fuel_capacity"],
                step=0.1
            )

            fuel_efficiency = st.slider(
                "Fuel Efficiency",
                min_value=5.0,
                max_value=60.0,
                value=default["Fuel_efficiency"],
                step=1.0
            )

            submit = st.form_submit_button(
                "Prediksi Harga Mobil",
                use_container_width=True
            )

    with col_output:
        st.markdown("### Hasil Prediksi")

        if submit:
            input_data = pd.DataFrame({
                "Vehicle_type": [vehicle_type],
                "Engine_size": [engine_size],
                "Horsepower": [horsepower],
                "Wheelbase": [wheelbase],
                "Width": [width],
                "Length": [length],
                "Curb_weight": [curb_weight],
                "Fuel_capacity": [fuel_capacity],
                "Fuel_efficiency": [fuel_efficiency]
            })

            prediksi = model.predict(input_data)[0]
            harga_usd = prediksi * 1000

            st.success("Prediksi berhasil dilakukan.")

            st.metric(
                label="Perkiraan Harga Mobil",
                value=f"${harga_usd:,.2f}"
            )

            st.write(f"Setara dengan **{prediksi:.2f} ribu dolar**.")

            st.divider()

            st.markdown("#### Ringkasan Input")
            st.dataframe(input_data, use_container_width=True)

            st.caption(
                "Hasil prediksi merupakan estimasi berdasarkan model Linear Regression."
            )

        else:
            st.warning("Silakan atur spesifikasi mobil, lalu klik tombol prediksi.")

            st.metric(
                label="Perkiraan Harga Mobil",
                value="$ -"
            )

            st.write("Hasil prediksi akan muncul pada bagian ini.")

# ==============================
# TAB 2 - KETERANGAN VARIABEL
# ==============================

with tab_variabel:
    st.subheader("Keterangan Variabel Input")

    data_keterangan = pd.DataFrame({
        "Variabel": [
            "Vehicle Type",
            "Engine Size",
            "Horsepower",
            "Wheelbase",
            "Width",
            "Length",
            "Curb Weight",
            "Fuel Capacity",
            "Fuel Efficiency"
        ],
        "Keterangan": [
            "Jenis kendaraan yang dipilih.",
            "Ukuran mesin kendaraan.",
            "Tenaga yang dihasilkan mesin.",
            "Jarak antara roda depan dan roda belakang.",
            "Lebar kendaraan.",
            "Panjang kendaraan.",
            "Berat kendaraan.",
            "Kapasitas bahan bakar kendaraan.",
            "Efisiensi penggunaan bahan bakar."
        ]
    })

    st.dataframe(data_keterangan, use_container_width=True)

    st.info(
        "Variabel tersebut digunakan sebagai input model untuk memperkirakan harga mobil."
    )

# ==============================
# TAB 3 - TENTANG SISTEM
# ==============================

with tab_sistem:
    st.subheader("Tentang Sistem")

    st.write(
        "Sistem ini dibuat untuk membantu memperkirakan harga mobil berdasarkan "
        "spesifikasi kendaraan yang dimasukkan oleh pengguna."
    )

    st.write(
        "Model yang digunakan adalah Linear Regression, dengan target prediksi "
        "berupa harga mobil dalam satuan ribu dolar."
    )

    st.markdown("### Tujuan Sistem")

    st.write("1. Membantu memperkirakan harga mobil berdasarkan spesifikasi.")
    st.write("2. Memberikan gambaran harga untuk kendaraan yang akan dibuat.")
    st.write("3. Menyediakan sistem sederhana yang dapat digunakan oleh end user.")

    st.markdown("### Informasi Project")

    st.write("Mata Kuliah: Sains Data")
    st.write("Project: Prediksi Harga Mobil")
    st.write("Algoritma: Linear Regression")

# ==============================
# FOOTER
# ==============================

st.divider()

st.markdown(
    "**Sistem ini dibuat oleh Fenty Anggraeni | 237006068**"
)
