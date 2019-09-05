import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

titanic_train_dataset = pd.read_csv("./train.csv")

selected_features = titanic_train_dataset[
    [
        "Pclass",
        "Sex",
        "Age"
    ]
]
selected_features.Sex = selected_features.Sex.map({'male': 0, 'female': 1})

selected_targets = titanic_train_dataset[["Survived"]]

processed_features = selected_features.copy()
processed_targets = selected_targets.copy()

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key:np.array(value) for key,value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features,targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
      ds = ds.shuffle(10000)
    features, labels = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    return features, labels

periods = 10
steps_per_period = 50 / periods
feature_columns = set([tf.feature_column.numeric_column(my_feature)
                       for my_feature in processed_features])

titanic_optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00003)
titanic_optimizer = tf.contrib.estimator.clip_gradients_by_norm(titanic_optimizer, 5.0)
titanic_linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=titanic_optimizer
)

training_input_fn = lambda: my_input_fn(processed_features, processed_targets, batch_size=5)
predict_training_input_fn = lambda: my_input_fn(processed_features, processed_targets, num_epochs=1, shuffle=False)
titanic_linear_regressor.train(training_input_fn, steps=steps_per_period)
training_predictions = titanic_linear_regressor.predict(predict_training_input_fn)
training_predictions = np.array([item['predictions'][0] for item in training_predictions])

