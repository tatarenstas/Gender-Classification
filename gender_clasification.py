from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("man.jpg")
plt.imshow(img[:,:,::-1])
plt.show()

gender = DeepFace.analyze(img, actions = ["gender"])["gender"]
print ("Gender: " + gender)
