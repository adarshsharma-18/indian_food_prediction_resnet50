import tensorflow as tf
import numpy as np
import cv2
import tf2onnx
import onnxruntime as ort

# Load the Keras model
keras_model = tf.keras.models.load_model('FoodResnet.keras')

# Workaround: Convert Sequential to Functional if needed
def sequential_to_functional(model):
    if isinstance(model, tf.keras.Sequential):
        # Build the model if not already built
        if not model.built:
            model.build((None, 224, 224, 3))
        inputs = tf.keras.Input(shape=(224, 224, 3))
        outputs = model(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

keras_model = sequential_to_functional(keras_model)

# Convert the Keras model to ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, opset=13, output_path="FoodResnet.onnx")

print("ONNX model saved as FoodResnet.onnx")

# Define food classes based on the training data
class_names={
    0:'burger' ,1:'butter_naan' ,2:'chai' ,3:'chapati' ,4:'chole_bhature' ,5:'dal_makhani' ,
    6:'dhokla' ,7:'fried_rice' ,8:'idli' ,9:'jalebi' ,10:'kaathi_rolls' ,
    11:'kadai_paneer' ,12:'kulfi' ,13:'masala_dosa' ,14:'momos' ,15:'paani_puri' ,
    16:'pakode' ,17:'pav_bhaji' ,18:'pizza' ,19:'samosa'
}

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)  # shape: (1, 224, 224, 3)
    return img

def predict_onnx(image_path, onnx_path="FoodResnet.onnx"):
    img = preprocess_image(image_path)
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: img})
    pred = np.argmax(outputs[0])
    print(f"Prediction: {class_names[pred]}")
    return class_names[pred]

if __name__ == "__main__":
    print("\nONNX Model Test")
    image_path = input("Enter the path to a food image for ONNX model testing: ")
    predict_onnx(image_path)