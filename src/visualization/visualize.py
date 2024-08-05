import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_learning_rate_vs_loss(history, lrs):
    try:
        plt.figure(figsize=(10, 7))
        plt.semilogx(lrs, history.history["loss"])
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning rate vs. loss")
        plt.show()
    except Exception as e:
        logging.error(" Error in plot_learning_rate_vs_los data: {}". format(e))
        
def plot_training_curves(history):
    try:
        pd.DataFrame(history.history).plot()
        plt.title("Model training curves")
        plt.show()
    except Exception as e:
        logging.error(" Error in procesplot_training_curves data: {}". format(e))
    
   