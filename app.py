# engagement_project/app.py

import gradio as gr
from predict import predict_boredom_from_video

# Deskripsi untuk antarmuka
title = "Deteksi Kebosanan Siswa dari Video"
description = """
Unggah video sesi pembelajaran untuk menganalisis tingkat kebosanan siswa. 
Model ini dilatih menggunakan dataset DAiSEE dan akan memberikan rincian prediksi per frame yang terdeteksi.
"""

iface = gr.Interface(
    fn=predict_boredom_from_video,
    inputs=gr.Video(label="Upload Video Pembelajaran"),
    outputs=gr.JSON(label="Hasil Analisis Kebosanan"),
    title=title,
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    print("Menjalankan antarmuka Gradio...")
    iface.launch()