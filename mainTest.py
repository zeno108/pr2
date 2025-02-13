import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('Breastcancer10Epochs.h5')

image = cv2.imread('C:\\Users\\Zarqa_2023\\Desktop\\sre\\pred\\P80MB80 FCC.jpg')
img = Image.fromarray(image)

img = img.resize((64, 64))

img = np.array(img)

input_img = np.expand_dims(img, axis=0)

result = model.predict(input_img)
predicted_class = np.argmax(result, axis=-1)

print(predicted_class)






