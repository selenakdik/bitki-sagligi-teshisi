import os
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

model_yolu = 'bitki_hastalik_modeli_v5.h5'
model = tf.keras.models.load_model(model_yolu)

with open('labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]


# --- TAHMİN FONKSİYONU ---
def resim_analiz_et(resim_yolu):
    frame = cv2.imread(resim_yolu)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- MODEL İÇİN ÖN İŞLEME (PREPROCESSING) ---
    img_resized = cv2.resize(frame, (256, 256))
    img_array = img_resized.astype("float32") 
    img_array = np.expand_dims(img_array, axis=0)

    # --- MODEL TAHMİNİ ---
    predictions = model.predict(img_array, verbose=0)
    result_index = np.argmax(predictions[0])

    predicted_class = class_names[result_index]
    confidence = 100 * np.max(predictions[0])

    # --- ARAYÜZÜ GÜNCELLEME ---
    sonuc_yazi = f"Tahmin: {predicted_class}\nGüven Oranı: %{confidence:.2f}"
    lbl_sonuc.config(text=sonuc_yazi)

    # Güven oranına göre yazı rengini ayarla 
    if confidence > 65:
        lbl_sonuc.config(fg="#2ecc71")  
    else:
        lbl_sonuc.config(fg="#e67e22")  

    img_pil = Image.fromarray(frame_rgb)
    img_pil = img_pil.resize((350, 350))  
    img_tk = ImageTk.PhotoImage(img_pil)

    lbl_resim.config(image=img_tk)
    lbl_resim.image = img_tk


# --- DOSYA SEÇME PENCERESİ ---
def dosya_sec():
    dosya_yolu = filedialog.askopenfilename(
        filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png")]
    )

    if dosya_yolu:
        resim_analiz_et(dosya_yolu)


pencere = tk.Tk()
pencere.title("Bitki Hastalığı Teşhis Sistemi")
pencere.geometry("500x600")
pencere.configure(bg="#f5f6fa")  

lbl_baslik = tk.Label(
    pencere,
    text="Yapay Zeka ile Bitki Analizi",
    font=("Arial", 16, "bold"),
    bg="#f5f6fa",
    fg="#2c3e50",
)
lbl_baslik.pack(pady=15)

btn_yukle = tk.Button(
    pencere,
    text="Fotoğraf Seç ve Analiz Et",
    font=("Arial", 12, "bold"),
    bg="#3498db",
    fg="white",
    padx=20,
    pady=10,
    command=dosya_sec,  
    relief="flat",
)
btn_yukle.pack(pady=10)

lbl_resim = tk.Label(pencere, bg="#f5f6fa")
lbl_resim.pack(pady=15)

lbl_sonuc = tk.Label(
    pencere,
    text="Lütfen bir yaprak fotoğrafı seçin.",
    font=("Arial", 13, "bold"),
    bg="#f5f6fa",
    fg="#7f8c8d",
    justify="center",
)
lbl_sonuc.pack(pady=20)

pencere.mainloop()