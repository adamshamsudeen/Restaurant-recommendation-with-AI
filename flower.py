from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
  # If the training and test sets aren't stored locally, download them.
  if not os.path.exists(IRIS_TRAINING):
    raw = urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "wb") as f:
      f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "wb") as f:
      f.write(raw)

  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="/tmp/iris_model")
  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # Train model.
  # classifier.train(input_fn=train_input_fn, steps=2000)

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)

  # Evaluate accuracy.
  # accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

  # print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.
  new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7],
       [4.3, 3.0, 1.1, 0.1],
       [5.6, 2.8, 4.9, 2.0],
       [5.5, 2.3, 4.0, 1.3]], dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

  predictions = list(classifier.predict(input_fn=predict_input_fn))
  # print(predictions)

  predicted_classes = [p["classes"] for p in predictions]

  predicted_prob = [p["probabilities"] for p in predictions]
  # print(predicted_classes)
  for c in predicted_classes:
    print(c)
  print("\n")
  for p in predicted_prob:
    print(p[0]*100, "Probablity for setosa")

    print(p[1]*100, "Probablity for versivolor")
    print(p[2]*100, "Probablity for verginica")  
    print("\n")
    # print(p)
    # for a in p:
    #   print(a*100)
  # print(list(predicted_prob[0]))
  # predicted_prob = dir(predicted_prob)

  # print(predicted_prob)
  # print(
  #     "New Samples, Class Predictions:    {}\n"
  #     .format(predicted_classes))
  # print(
  #     "New Samples, Class Predictions:    {}\n"
  #     .format(predicted_prob))
  print(classifier.latest_checkpoint())
  # predictions = list(classifier.predict_proba(input_fn=predict_input_fn))
  # print(predictions)  
  
if __name__ == "__main__":
    main()