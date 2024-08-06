import logging
from src.data.load_dataset import load_and_preprocess_data
from src.visualization.visualize import plot_training_curves
from src.feature.build_features import create_model, compile_model
from src.model.train_model import train_model
from src.model.predict_model import evaluate_model, predict_model
from sklearn.metrics import accuracy_score
import warnings
import tensorflow as tf

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # Load and preprocess the data
    print(tf.__version__)
    x_train, x_test, y_train, y_test = load_and_preprocess_data('src/data/employee_attrition.csv')

    # Example of creating and training a model
    tf.keras.utils.set_random_seed(42)
    model = create_model([tf.keras.layers.Dense(1),tf.keras.layers.Dense(1, activation='sigmoid')])
    
    compile_model(model, 
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.0009),
                  metrics=['accuracy'])
    history = train_model(model, x_train, y_train, epochs=50)
    
    print("Train evaluation:")
    logging.info("Train evaluation:")
    evaluate_model(model, x_train, y_train)
    
    y_preds = predict_model(model, x_test)
    print("Test accuracy:", accuracy_score(y_test, y_preds))
    
    plot_training_curves(history)
    
    