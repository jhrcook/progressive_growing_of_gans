#!/usr/bin/env python3

import argparse
import os

import numpy as np
import tensorflow as tf

import dataset
import config
import misc
import tfutil
import train
import util_scripts

from generate_and_evaluate_gan import generate_fake_images_with_input_vectors


def check_input_array_dimensions(a, d):
    if not a.shape[0] == d:
        raise ValueError(
            f"Input array will create {a.shape[0]} images, but {d} are expected."
        )
    return True


def main(run_id, input_array, output_dir, check_dims=-1):
    arr = np.loadtxt(input_array)
    if check_dims > 0:
        _ = check_input_array_dimensions(arr, check_dims)
    generate_fake_images_with_input_vectors(
        run_id, num_pngs=arr.shape[0], input_array=arr, output_dir=output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_id",
        type=int,
        help="number corresponding to the GAN to use (in './results/' dir)",
    )
    parser.add_argument("input_array", type=str, help="path to file with input array")
    parser.add_argument("output_dir", type=str, help="output directory")
    parser.add_argument(
        "-n",
        "--check-input-dimensions",
        type=int,
        default=-1,
        help="provide a positive number to check the number expected images before running",
    )
    parser.add_argument("-g", "--num-gpus", type=int, default=1, help="number of GPUs")
    args = parser.parse_args()

    # Override number of GPUs in config.py.
    config.num_gpus = args.num_gpus
    print(f"num. GPUS: {config.num_gpus}")

    # Configuration (copied from 'train.py').
    misc.init_output_logging()
    np.random.seed(config.random_seed)
    print("Initializing TensorFlow...")
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)

    main(args.run_id, args.input_array, args.output_dir, args.check_input_dimensions)
    print("done")
