import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import keras
import cv2
import time

model = load_model('model_objects.h5')
class_names = ['bags', 'electric_socket', 'hand', 'pens', 'phone', 'pins', 'plants']

# define a video capture object
vid = cv2.VideoCapture(0)  
while(True):
      
    # Capture the video frame
    # and resize
    ret, frame = vid.read()
    resized = cv2.resize(frame, (224,224))
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    a = image.img_to_array(img, dtype = 'uint8')
    a = np.expand_dims(a, axis=0)

    # Model prediction
    y_hat = model.predict(a)
    print(class_names[y_hat.argmax()] + ' with probability ' + str(y_hat.max()))
        
    # Display the resulting frame
    cv2.imshow('frame', resized)
    time.sleep(0.2)  
    
    #Exit with 'q' 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()