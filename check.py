import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('FoodResnet.keras')

# Define food classes based on the training data
class_names={
    0:'burger' ,1:'butter_naan' ,2:'chai' ,3:'chapati' ,4:'chole_bhature' ,5:'dal_makhani' ,
    6:'dhokla' ,7:'fried_rice' ,8:'idli' ,9:'jalebi' ,10:'kaathi_rolls' ,
    11:'kadai_paneer' ,12:'kulfi' ,13:'masala_dosa' ,14:'momos' ,15:'paani_puri' ,
    16:'pakode' ,17:'pav_bhaji' ,18:'pizza' ,19:'samosa'
}

def predict_image(image, model):
    test_img=cv2.imread(image)
    plt.imshow(test_img)
    
    test_img=cv2.resize(test_img, (224,224))
    test_img=np.expand_dims(test_img, axis=0)
    
    result=model.predict(test_img)
    
    r=np.argmax(result)
    print(class_names[r])

if __name__ == "__main__":
    while True:
        print("\nIndian Food Classification System")
        print("Enter 'q' to quit")
        image_path = input("\nEnter the path to your food image: ")
        
        if image_path.lower() == 'q':
            break
            
        result = predict_image(image_path, model)
        
        print(result)