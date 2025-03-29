"""
This module initializes the TFX pipeline components.

The components include data ingestion, data validation, feature transformation,
hyperparameter tuning, model training, model evaluation, and model deployment.
It builds a pipeline using TensorFlow Extended (TFX) components and integrates
them for end-to-end machine learning workflows.

Main Components:
- CsvExampleGen: Ingests data from CSV files.
- StatisticsGen: Computes dataset statistics.
- SchemaGen: Infers the schema of the dataset.
- ExampleValidator: Detects anomalies and missing values.
- Transform: Applies feature engineering transformations.
- Tuner: Optimizes hyperparameters using a tuning module.
- Trainer: Trains a machine learning model.
- Resolver: Fetches the latest blessed model for evaluation comparison.
- Evaluator: Analyzes model performance using predefined metrics.
- Pusher: Deploys the trained model if it passes evaluation.
"""

from dataclasses import dataclass
import os
import tensorflow_model_analysis as tfma

from tfx.components import (
    CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator,
    Transform, Tuner, Trainer, Evaluator, Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)


@dataclass
class ComponentConfig:
    """
    Configuration class for initializing TFX components.

    Attributes:
        data_dir (str): Directory where the input data is stored.
        transform_module (str): Path to the transformation module.
        tuner_module (str): Path to the hyperparameter tuning module.
        training_module (str): Path to the model training module.
        training_steps (int): Number of steps for training.
        eval_steps (int): Number of steps for evaluation.
        serving_model_dir (str): Directory to store the trained model for deployment.
    """
    data_dir: str
    transform_module: str
    tuner_module: str
    training_module: str
    training_steps: int
    eval_steps: int
    serving_model_dir: str


def init_components(config: ComponentConfig):
    """
    Initializes TFX components required for building a pipeline for training and deploying a model.
    """
    # Disable pylint no-member as some attributes are dynamically generated
    # and not recognized by pylint. Source:
    # https://stackoverflow.com/a/54156144

    # pylint: disable=no-member
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2)
            ]
        )
    )
    # pylint: enable=no-member

    example_gen = CsvExampleGen(
        input_base=config.data_dir,
        output_config=output_config)
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(config.transform_module)
    )

    # pylint: disable=no-member
    tuner = Tuner(
        module_file=os.path.abspath(
            config.tuner_module),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(
            splits=["train"],
            num_steps=config.training_steps),
        eval_args=trainer_pb2.EvalArgs(
            splits=["eval"],
            num_steps=config.eval_steps),
    )
    # pylint: enable=no-member

    # pylint: disable=no-member
    trainer = Trainer(
        module_file=os.path.abspath(
            config.training_module),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=config.training_steps),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=config.eval_steps))
    # pylint: enable=no-member

    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='Depression')],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name='FalsePositives'),
                tfma.MetricConfig(class_name='TruePositives'),
                tfma.MetricConfig(class_name='FalseNegatives'),
                tfma.MetricConfig(class_name='TrueNegatives'),
                tfma.MetricConfig(class_name='BinaryAccuracy',
                                  threshold=tfma.MetricThreshold(
                                      value_threshold=tfma.GenericValueThreshold(
                                          lower_bound={'value': 0.5}
                                      ),
                                      change_threshold=tfma.GenericChangeThreshold(
                                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                          absolute={'value': 0.0001}
                                      )
                                  )
                                  )
            ])
        ]
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )

    # pylint: disable=no-member
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=config.serving_model_dir
            )
        ),
    )
    # pylint: enable=no-member

    return (
        example_gen, statistics_gen, schema_gen, example_validator,
        transform, tuner, trainer, model_resolver, evaluator, pusher
    )
