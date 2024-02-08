import cv2
from matplotlib.pyplot import imshow, axis, show

# Read the input image
img = cv2.imread('input.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the pre-trained face classifier imported from cv2 module
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

# Display the image with detected faces
imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axis('off')
show()
