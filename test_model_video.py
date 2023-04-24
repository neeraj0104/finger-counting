import os
import time
import cv2
import keras
import numpy as np
from keras.preprocessing import image
from PIL import Image

from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from keras.models import Model
from keras.applications import imagenet_utils
from tensorflow.keras.models import load_model

from keras.utils import load_img
from keras.utils import  img_to_array

import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils
import sys
 
model = load_model('finger_model.h5')

def prepare_image2 (img):
    cvt_image =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(cvt_image)

    im_resized = im_pil.resize((224, 224))
    img_array = img_to_array(im_resized)
    image_array_expanded = np.expand_dims(img_array, axis = 0)
    return keras.applications.mobilenet.preprocess_input(image_array_expanded)
    
        
frame_count = 0
cap = cv2.VideoCapture(0)
if not cap.isOpened():
        raise IOError("We cannot open webcam")

while True:
      ret, frame = cap.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     
      cv2.rectangle(gray, (1, 1), (300, 300), (255,155,30), 2)
              
      crop_img = frame[1:300, 1:300]
           
      preprocessed_image = prepare_image2(crop_img)
      predictions = model.predict(preprocessed_image)
      pred = np.argmax(predictions,axis=1)
      print('Prediction returned {0}'.format(pred))

      cv2.putText(gray, str(pred), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
      cv2.imshow('frame',gray)
          
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break                
     


        
        
        
        
        
   
    
