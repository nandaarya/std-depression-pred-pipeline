# pylint: disable=no-member
"""
This module defines the hyperparameter tuning process for a deep learning model
using Keras Tuner and TensorFlow Transform (TFT) in a TFX pipeline.

The tuner applies Bayesian Optimization to search for the best hyperparameters
for a binary classification model predicting student depression.

Main Components:
- `input_fn()`: Reads and prepares the dataset for training and evaluation.
- `model_builder()`: Defines the neural network architecture with tunable parameters.
- `tuner_fn()`: Configures and executes the Bayesian Optimization tuning process.
"""

from typing import NamedTuple, Dict, Text, Any
import keras_tuner as kt
import tensorflow as tf
from keras import layers
import tensorflow_transform as tft
from keras_tuner.engine import base_tuner
from tfx.components.trainer.fn_args_utils import FnArgs
from modules.transform import transformed_name, LABEL_KEY, NUMERIC_FEATURES, CATEGORICAL_FEATURES

NUM_EPOCHS = 10

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])


def gzip_reader_fn(filenames):
    """
    Reads GZIP-compressed TFRecord files and returns a dataset.
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    """
    Prepares the dataset for training and evaluation.
    """
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=LABEL_KEY)
    return dataset


def build_inputs(tf_transform_output):
    """
    Creates input layers for numeric and categorical features.
    Returns a dictionary of inputs and concatenated feature tensors.
    """
    inputs = {}

    # Numeric Features
    numeric_inputs = [
        tf.keras.Input(
            shape=(
                1,
            ),
            name=transformed_name(f),
            dtype=tf.float32) for f in NUMERIC_FEATURES]
    for f in NUMERIC_FEATURES:
        inputs[transformed_name(f)] = numeric_inputs[NUMERIC_FEATURES.index(f)]

    concat_numeric = layers.concatenate(numeric_inputs)

    # Categorical Features
    categorical_inputs = []
    for feature in CATEGORICAL_FEATURES:
        transformed_feature_name = transformed_name(feature)
        vocab_size = tf_transform_output.vocabulary_size_by_name(
            feature.replace(" ", "_").lower() + "_vocab") + 1

        categorical_input = tf.keras.Input(
            shape=(
                vocab_size,
            ),
            name=transformed_feature_name,
            dtype=tf.float32)
        categorical_inputs.append(categorical_input)
        inputs[transformed_feature_name] = categorical_input

    concat_categorical = layers.concatenate(categorical_inputs)

    return inputs, concat_numeric, concat_categorical


def model_builder(hp, tf_transform_output):
    """
    Builds a Keras model with tunable hyperparameters.
    """
    inputs, concat_numeric, concat_categorical = build_inputs(
        tf_transform_output)

    x = layers.Dense(
        hp.Int(
            "dense_units_1",
            min_value=32,
            max_value=128,
            step=32),
        activation='relu')(concat_numeric)
    combined = layers.concatenate([x, concat_categorical])
    x = layers.Dense(
        hp.Int(
            "dense_units_2",
            min_value=32,
            max_value=128,
            step=32),
        activation='relu')(combined)
    x = layers.Dropout(
        hp.Float(
            "dropout_rate",
            min_value=0.1,
            max_value=0.5,
            step=0.1))(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice(
                "learning_rate",
                values=[
                    1e-2,
                    1e-3,
                    1e-4])),
        metrics=[
            tf.keras.metrics.BinaryAccuracy()])
    return model


def tuner_fn(fn_args: FnArgs):
    """
    Defines the hyperparameter tuning function using Bayesian Optimization.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files, tf_transform_output, NUM_EPOCHS)
    eval_set = input_fn(fn_args.eval_files, tf_transform_output, NUM_EPOCHS)

    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: model_builder(hp, tf_transform_output),
        objective=kt.Objective('binary_accuracy', direction='max'),
        max_trials=30,
        directory=fn_args.working_dir,
        project_name="bayesian_tuning",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": eval_set,
            "epochs": NUM_EPOCHS,
        },
    )
