
import streamlit as st
import pandas as pd
import joblib

# ==============================
# LOAD MODEL DAN CONFIG
# ==============================

model = joblib.load("model_prediksi_harga_mobil.pkl")
config = joblib.load("app_config.pkl")

vehicle_type_list = config["vehicle_type_list"]
default = config["default_values"]

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Prediksi Harga Mobil",
    layout="wide"
)

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
    st.write("2. Masukkan spesifikasi mobil.")
    st.write("3. Klik tombol prediksi.")
    st.write("4. Lihat hasil estimasi harga.")

    st.divider()

    st.header("Informasi Model")
    st.write("Algoritma: Linear Regression")
    st.write("Target: Harga mobil")
    st.write("Satuan harga: ribu dolar")

    st.divider()

    st.caption("Final Project Sains Data")

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
# TAB 1 - PREDIKSI HARGA
# ==============================

with tab_prediksi:
    st.subheader("Prediksi Harga Mobil")

    st.info(
        "Nilai awal pada form menggunakan rekomendasi dari data mobil dengan penjualan tinggi. "
        "Pengguna dapat mengubah nilai sesuai spesifikasi mobil yang ingin diprediksi."
    )

    col_input, col_output = st.columns([1.3, 1])

    with col_input:
        with st.container(border=True):
            st.markdown("### Input Spesifikasi")

            with st.form("form_prediksi"):
                vehicle_type = st.selectbox(
                    "Tipe Kendaraan",
                    vehicle_type_list,
                    index=vehicle_type_list.index(default["Vehicle_type"])
                    if default["Vehicle_type"] in vehicle_type_list else 0
                )

                col1, col2 = st.columns(2)

                with col1:
                    engine_size = st.number_input(
                        "Engine Size",
                        min_value=0.0,
                        value=float(default["Engine_size"]),
                        step=0.1,
                        format="%.2f"
                    )

                    horsepower = st.number_input(
                        "Horsepower",
                        min_value=0.0,
                        value=float(default["Horsepower"]),
                        step=1.0,
                        format="%.2f"
                    )

                    wheelbase = st.number_input(
                        "Wheelbase",
                        min_value=0.0,
                        value=float(default["Wheelbase"]),
                        step=0.1,
                        format="%.2f"
                    )

                    width = st.number_input(
                        "Width",
                        min_value=0.0,
                        value=float(default["Width"]),
                        step=0.1,
                        format="%.2f"
                    )

                with col2:
                    length = st.number_input(
                        "Length",
                        min_value=0.0,
                        value=float(default["Length"]),
                        step=0.1,
                        format="%.2f"
                    )

                    curb_weight = st.number_input(
                        "Curb Weight",
                        min_value=0.0,
                        value=float(default["Curb_weight"]),
                        step=0.1,
                        format="%.2f"
                    )

                    fuel_capacity = st.number_input(
                        "Fuel Capacity",
                        min_value=0.0,
                        value=float(default["Fuel_capacity"]),
                        step=0.1,
                        format="%.2f"
                    )

                    fuel_efficiency = st.number_input(
                        "Fuel Efficiency",
                        min_value=0.0,
                        value=float(default["Fuel_efficiency"]),
                        step=1.0,
                        format="%.2f"
                    )

                submit = st.form_submit_button(
                    "Prediksi Harga Mobil",
                    use_container_width=True
                )

    with col_output:
        with st.container(border=True):
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
                st.warning("Silakan isi spesifikasi mobil, lalu klik tombol prediksi.")

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

    with st.container(border=True):
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
