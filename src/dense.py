import os
from pathlib import Path

from absl import app
from absl import flags
from absl import logging

import gin

import numpy as np
from scipy.interpolate import CubicSpline

import tensorflow as tf

# import tensorflow_datasets as tfds
from kerastuner.tuners import BayesianOptimization

# from num2tex import num2tex
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# import cosmoplotian.colormaps
# string_cmap = "div yel grn"
# cmap = mpl.cm.get_cmap(string_cmap)
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[cmap(0.2), "k", "red"]) 

from datasets import LoadDataset, CAMBPowerspectrum
from plot_eval import Eval

plt.rcParams['text.usetex'] = True

tfkl = tf.keras.layers
tfk = tf.keras

FLAGS = flags.FLAGS

#MODEL_DIR = "/global/cscratch1/sd/bthorne/NeuralBoltzmann"
# DATA_DIR = Path(os.environ["NEURALBOLTZMANN_DATA_DIR"])

@gin.configurable
def BuildDenseNetwork(input_dim, output_dim, units, activation='relu', dropout_rate=0.5):
    inputs = tfkl.Input(input_dim)
    x_ = tfkl.Dense(2 * input_dim)(inputs)
    for num_units in units:
        x_ = tfkl.Dense(num_units, activation=activation)(x_)
    x_ = tfkl.Dropout(dropout_rate)(x_)
    x_ = tfkl.Dense(output_dim, activation=activation)(x_)
    return tfk.Model(inputs=inputs, outputs=x_)

@gin.configurable
def Train(model, x_train, y_train, val_data=None, batch_size=100, epochs=10, learning_rate=1e-3, loss=tfk.losses.MSE):
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=val_data, shuffle=True)
    return model 


def main(argv):
    del argv
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
        tf.get_logger().setLevel('DEBUG')
    else:
        logging.set_verbosity(logging.INFO)
        tf.get_logger().setLevel('INFO')
    # get config file path
    config_path = Path(FLAGS.config_file)
    # use file name as identifier for saved files
    stem = config_path.stem
    results_dir = Path(FLAGS.results_dir) / stem
    results_dir.mkdir(exist_ok=True, parents=True)

    # set data directory
    data_dir = DATA_DIR / stem
    data_dir.mkdir(exist_ok=True, parents=True)

    # parse configuration and lock it in
    logging.debug(f"Using Gin config {config_path}")
    gin.parse_config_file(str(config_path))
    logging.debug(gin.config_str())

    # setup checkpointing
    checkpoint_filepath = DATA_DIR / stem / "checkpoints" / "checkpoint"
    checkpoint_filepath.parent.mkdir(exist_ok=True, parents=True)

    # setup hyperparameter search directory
    hyper_filepath = DATA_DIR / stem / "hyperparameters" / "bayesianoptimization"
    hyper_filepath.parent.mkdir(exist_ok=True, parents=True)

    # The following is intended to be structured in a way that we do not need
    # to keep retraining networks, or recomputing large sets of spectra, during
    # the development of new applications. 
    # Keep training of NNs to the 'train' step, or additional new step at the beginning.
    # Save and reload for next applications.
    
    # These datasets are used in many different tasks so just unpack here.
    dset = LoadDataset()
    (x_train, y_train) = dset["train"]
    (x_val, y_val) = dset["val"]
    (x_test, y_test) = dset["test"]
    (raw_x_test, raw_y_test) = dset["raw_test"]

    # do training of NN
    if FLAGS.mode == "train":
        model = BuildDenseNetwork()
        model = Train(model, x_train, y_train, val_data=dset["test"])
        model.save(str(checkpoint_filepath))

    # Load a previously defined model and make plots 
    if FLAGS.mode =='eval':
        model = tfk.models.load_model(str(checkpoint_filepath))

    with open(results_dir / "operative_gin_config.txt", "w") as f:
        f.write(gin.operative_config_str())
    return

if __name__ == "__main__":
    flags.DEFINE_enum("mode", "train", ["train", "eval"], "Mode in which to run script.")
    flags.DEFINE_string("results_dir", "./results", "Path to results directory where plots will be saved.")
    flags.DEFINE_string("config_file", "./configs/dense_128_tt.gin", "Path to configuration file.")
    flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
    app.run(main)
