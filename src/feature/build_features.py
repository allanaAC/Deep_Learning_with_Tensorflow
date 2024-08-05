import pandas as pd
import logging
import tensorflow as tf

def create_model(layers):
    try:
        model = tf.keras.Sequential(layers)
        return model
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))

def compile_model(model, loss, optimizer, metrics):
    try:
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics) 
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))