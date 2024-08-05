import logging
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf

# # Function to predict and evaluate model 
def evaluate_model(model, x, y):
    try:
        return model.evaluate(x, y)
    except Exception as e:
        logging.error(" Error in evaluate_model data: {}". format(e))
        
def predict_model(model, x):
    try:
        return tf.round(model.predict(x))
    except Exception as e:
        logging.error(" Error in predict_model data: {}". format(e))
