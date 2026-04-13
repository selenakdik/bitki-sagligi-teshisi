import cv2

video_url = "http://192.168.1.102:8080/video" 

print("Goruntu aktarimi baslatiliyor...")


cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

window_name = 'Selen - Bitki Analizi'

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Veri akisi kesildi!")
        break

    cv2.putText(frame, "BITKI ANALIZI", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Sistem guvenli bir sekilde kapatildi.")