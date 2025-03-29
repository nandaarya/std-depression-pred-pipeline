"""
Pipeline definition for the student depression prediction model.

This script initializes and runs a TFX pipeline for data transformation,
hyperparameter tuning, model training, and evaluation.
"""

import os
from typing import Text

from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from modules.components import ComponentConfig, init_components

PIPELINE_NAME = "nandaaryaputra-pipeline"

DATA_ROOT = "/kaggle/input/student-depression-dataset"
COMPONENTS_FILE = "modules/components.py"
PIPELINE_FILE = "modules/pipeline.py"
TRANSFORM_MODULE_FILE = "modules/transform.py"
TUNER_MODULE_FILE = "modules/tuner.py"
TRAINER_MODULE_FILE = "modules/trainer.py"

OUTPUT_BASE = "output"
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")


def init_local_pipeline(
    components_list, root_dir: Text
) -> pipeline.Pipeline:
    """
    Initialize a local TFX pipeline.

    Args:
        components_list: A list of TFX components to be included in the pipeline.
        root_dir: Root directory for pipeline output artifacts.

    Returns:
        A TFX pipeline.
    """
    logging.info(f'Pipeline root set to: {root_dir}')
    beam_args = [
        '--direct_running_mode=multi_processing',
        '--direct_num_workers=0'
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=root_dir,
        components=components_list,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_args
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)

    config = ComponentConfig(
        data_dir=DATA_ROOT,
        training_module=TRAINER_MODULE_FILE,
        transform_module=TRANSFORM_MODULE_FILE,
        tuner_module=TUNER_MODULE_FILE,
        training_steps=800,
        eval_steps=400,
        serving_model_dir=serving_model_dir
    )

    pipeline_comps = init_components(config)

    student_pipeline = init_local_pipeline(pipeline_comps, pipeline_root)
    BeamDagRunner().run(pipeline=student_pipeline)
