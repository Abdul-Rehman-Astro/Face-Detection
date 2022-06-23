import cv2
 
face_Cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    # We are getting each frame forom video capture
    # cap.read gives two values , frame was read correctly or not (flag) , image of each frame
    _,img = cap.read()  

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_Cascade.detectMultiScale(img_gray,1.1,4)

    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255.0,0),2)
    cv2.imshow("Face Detection Done!",img)

    # since it is a infinite loop we need to break it 
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break


