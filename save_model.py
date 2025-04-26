
import tensorflow as tf
model = tf.keras.models.load_model('FoodResnet.keras')
tf.saved_model.save(model, 'temp_saved_model')
print("Model saved successfully")
