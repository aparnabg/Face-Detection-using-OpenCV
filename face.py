import cv2

web_cam_video_capture = cv2.VideoCapture(0)
haar = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml') 
while True:
    isTrue, image = web_cam_video_capture.read()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_retangle = haar.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=9)
    for i, (x, y, h, w) in enumerate(face_retangle):
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(image, f'Face {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    cv2.imshow('Detected_face', image)
    k = cv2.waitKey(30) 
    if k==27:
        break
web_cam_video_capture.release()
cv2.destroyAllWindows() 