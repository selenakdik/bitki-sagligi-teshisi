import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

#  Modeli Yükleme
print("Model yukleniyor, lutfen bekleyin...")
model = load_model('bitki_hastalik_modeli_v2.h5')

# Sınıf İsimlerini labels.txt dosyasından al
with open('labels.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Kamera Bağlantısı 
video_url = "http://192.168.1.100:8080/video" 
cap = cv2.VideoCapture(video_url)

print("Analiz sistemi aktif. Cikmak icin 'q' tusuna basin.")

count = 0
label_text = "Bekleniyor..."
confidence = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    count += 1
    # İşlemciyi yormamak için her 5 karede bir tahmin yap
    if count % 5 == 0:
        # Boyutlandırma
        img = cv2.resize(frame, (256, 256))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Tahmin
        preds = model.predict(img, verbose=0)
        label_id = np.argmax(preds)
        
        # Etiketi ve güven oranını listeden çek
        label_text = class_labels[label_id]
        confidence = preds[0][label_id] * 100

    # Yazı rengini güvene göre (%70 altı kırmızı, üstü yeşil)
    color = (0, 255, 0) if confidence > 70 else (0, 0, 255)
    display_text = f"{label_text}: %{confidence:.1f}"
    
    cv2.putText(frame, display_text, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Bitki Hastalik Teshis Sistemi', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program kapatildi.")