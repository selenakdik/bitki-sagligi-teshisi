import cv2
import numpy as np
import tensorflow as tf

# Modeli ve etiketleri yükle
model_yolu = 'bitki_hastalik_modeli_v5.h5'
print(f"{model_yolu} yükleniyor, bekleyin...")
model = tf.keras.models.load_model(model_yolu)

with open('labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
print("Model ve etiketler başarıyla yüklendi! ✅")

# IP Webcam URL 
url = "http://192.168.1.102:8080/video" 
cap = cv2.VideoCapture(url)

print("Kamera bağlantısı kuruldu. Canlı analiz başlıyor...")
print("Çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Görüntü alınamadı!")
        break

    # --- ÖN İŞLEME (PREPROCESSING) ---
    # Modelin beklediği boyut (256, 256)
    img = cv2.resize(frame, (256, 256))
    
    # Görüntüyü 0-255 arası tam sayılar halinde tutuyoruz ama tipini float32 yapıyoruz.
    # Model kendi içindeki Rescaling(1./255) katmanıyla bunu halledecek.
    img_array = img.astype("float32")
    
    # Boyut Genişletme (1, 256, 256, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # --- TAHMİN (INFERENCE) ---
    predictions = model.predict(img_array, verbose=0)
    result_index = np.argmax(predictions[0])
    
    predicted_class = class_names[result_index]
    confidence = 100 * np.max(predictions[0])

    # Terminale yazdır (Değişimi takip et)
    print(f"Tahmin: {predicted_class:<40} | Güven: %{confidence:.2f}")

    # --- EKRANA YAZDIRMA ---
    if confidence > 30: # %30 üzeri güvenliyse ismi yaz
        label_text = f"{predicted_class} (%{confidence:.1f})"
        color = (0, 255, 0) # Yeşil
    else: # Emin değilse
        label_text = "Analiz ediliyor ..."
        color = (0, 165, 255) # Turuncu

    cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    cv2.namedWindow('IP Webcam Analiz v4', cv2.WINDOW_NORMAL)
    cv2.imshow('IP Webcam Analiz v4', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("Program kapatildi.")