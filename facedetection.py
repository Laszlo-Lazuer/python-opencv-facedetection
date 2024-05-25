import cv2

detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # Trained for grayscale
# imp_img = cv2.VideoCapture("elon.jpg") # can be a camera or file # elon face
imp_img = cv2.VideoCapture("mark.jpg") # can be a camera or file


res, img = imp_img.read() # pixel and coordinates stored(bool)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert image to gray

faces = detect.detectMultiScale(gray, 1.3, 5) # x, y, w, h: x,y coords; w,h: dimensions

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 0), 2) # last param is width of border

# cv2.imshow("Elon Image", img) # Elon face
cv2.imshow("Mark Image", img)

# After showing your image you must
cv2.waitKey(0) # ms to open image then close in time, 0 keep open
imp_img.release() # Cleanup
cv2.destroyAllWindows() # Cleanup


