import keras_ocr
from matplotlib import pyplot as plt 
import cv2 
import os 
import glob 
import pandas as pd

#kears-ocr will automatically download pretrained model
#weights for the detector and recognizer
pipline=keras_ocr.pipeline.Pipeline()



data_path = os.path.join("C:/Users/HP/Documents/Newfolder/billboard"  ,'*g') 
files = glob.glob(data_path) 
data = [] 
for f1 in files: 
    img = cv2.imread(f1) 
    plt.imshow(img)
    data.append(img) 
    
prediction_group=pipline.recognize(data)


fix, axs= plt.subplots(nrows=len(data), figsize=(20, 20))
for ax, data, predictions in zip(axs, data, prediction_group):
    keras_ocr.tools.drawAnnotations(image=data, predictions=predictions, ax=ax)
    df = pd.DataFrame(prediction_group[0], columns=['text', 'bbox'])
    print(df)


    

