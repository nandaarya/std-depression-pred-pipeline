"""
This module defines the feature preprocessing function for the TFX pipeline.

It applies transformations to the input dataset using TensorFlow Transform (TFT).
The transformations include feature scaling, categorical encoding, and data cleaning.
Unused features are removed to optimize model performance.

Main Transformations:
- Standardization: Numeric features are normalized using Z-score normalization.
- One-Hot Encoding: Categorical features are encoded with vocabulary-based one-hot encoding.
- Data Cleaning: Entries with missing values in "Financial Stress" are removed.
"""

import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "Depression"
NUMERIC_FEATURES = [
    "Academic Pressure",
    "Age",
    "CGPA",
    "Study Satisfaction",
    "Work/Study Hours"]
CATEGORICAL_FEATURES = [
    "Dietary Habits",
    "Family History of Mental Illness",
    "Financial Stress",
    "Gender",
    "Have you ever had suicidal thoughts ?",
    "Sleep Duration"]

UNUSED_FEATURES = [
    "Job Satisfaction",
    "Work Pressure",
    "id",
    "City",
    "Profession",
    "Degree"]


def transformed_name(key):
    """
    Converts feature names by replacing spaces and special characters
    to make them compatible with TensorFlow Transform.
    """
    return key.replace("/", "_").replace(" ", "_").lower() + "_xf"


def preprocessing_fn(inputs):
    """
    Applies preprocessing to input features.

    Args:
        inputs: A dictionary mapping feature keys to raw tensors.

    Returns:
        outputs: A dictionary mapping feature keys to transformed tensors.
    """
    outputs = {}

    # Remove unused features
    filtered_inputs = {
        key: tensor for key,
        tensor in inputs.items() if key not in UNUSED_FEATURES}

    # Remove data where 'Financial Stress' is missing ("?")
    mask = tf.not_equal(filtered_inputs["Financial Stress"], "?")
    clean_inputs = {
        key: tf.boolean_mask(
            tensor,
            mask) for key,
        tensor in filtered_inputs.items()}

    # Standardize numerical features using Z-score normalization
    for feature in NUMERIC_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(
            clean_inputs[feature])

    # One-Hot Encoding for categorical features
    for feature in CATEGORICAL_FEATURES:
        vocab_filename = feature.replace(" ", "_").lower() + "_vocab"

        # Apply encoding using the created vocabulary
        indexed = tft.compute_and_apply_vocabulary(
            clean_inputs[feature],
            vocab_filename=vocab_filename,
            num_oov_buckets=1
        )

        vocab_size = tf.cast(
            tft.experimental.get_vocabulary_size_by_name(vocab_filename) + 1,
            tf.int32)

        # One-Hot Encoding
        one_hot_encoded = tf.one_hot(
            indexed,
            depth=vocab_size,
            on_value=1.0,
            off_value=0.0)

        outputs[transformed_name(feature)] = one_hot_encoded

    # Retain the target label without modification
    outputs[LABEL_KEY] = clean_inputs[LABEL_KEY]

    return outputs
