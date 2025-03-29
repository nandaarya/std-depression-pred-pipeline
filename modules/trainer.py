# pylint: disable=no-member
"""
This module defines the training pipeline for a deep learning model used in a TFX pipeline.

The model predicts depression risk based on various numerical and categorical features.
It utilizes TensorFlow Transform (TFT) for preprocessing and TensorFlow Keras for training.

Main Components:
- `input_fn()`: Reads and prepares the dataset for training and validation.
- `model_builder()`: Defines the neural network architecture.
- `_get_serve_tf_example_fn()`: Creates a serving function for model inference.
- `run_fn()`: Trains the model and exports it for serving.
"""

import os
import tensorflow as tf
import tensorflow_transform as tft
from keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from modules.transform import LABEL_KEY
from modules.tuner import input_fn, build_inputs


def model_builder(tf_transform_output):
    """
    Build the deep learning model using transformed features and best hyperparameter from tuning.
    """
    inputs, concat_numeric, concat_categorical = build_inputs(
        tf_transform_output)

    x = layers.Dense(64, activation='relu')(concat_numeric)
    combined = layers.concatenate([x, concat_categorical])
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.01),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    model.summary()
    return model


def _get_serve_tf_example_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs):
    """Train and save the model."""
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch')
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy', mode='max', patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(fn_args.serving_model_dir, "best_model.keras"),
        monitor='val_binary_accuracy',
        mode='max',
        save_best_only=True
    )

    # Load TFT transform graph
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Load training and validation datasets
    train_set = input_fn(
        fn_args.train_files,
        tf_transform_output,
        num_epochs=10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)

    # Build and train the model
    model = model_builder(tf_transform_output)
    model.fit(
        x=train_set,
        validation_data=val_set,
        callbacks=[tensorboard_callback, es, mc],
        epochs=10)

    signatures = {
        "serving_default": _get_serve_tf_example_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name="examples"
            )
        )}

    # Save final model
    tf.saved_model.save(
        model,
        fn_args.serving_model_dir,
        signatures=signatures)
