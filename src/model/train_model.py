import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to train the model
def train_model(model, x_train, y_train, epochs, verbose=0):
    try:
        return model.fit(x_train, y_train, epochs=epochs, verbose=verbose)
    except Exception as e:
        logging.error(" Error in train_model data: {}". format(e))
