#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import scipy.ndimage

import config
from util_scripts import generate_interpolation_video
import misc
import tfutil

# Set up logging.
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter("[%(levelname)s]:%(funcName)s:%(lineno)d - %(message)s")
c_handler.setFormatter(c_format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(c_handler)


def interpolate_two_points(a, b, n_steps):
    c = b - a
    pts = []
    for i in range(n_steps - 1):
        pts.append(a + c * i / n_steps)
    pts.append(b)
    return np.vstack(pts)


def make_interpolation_array(arr, frames_per_interpolation):
    interps = []
    for i in range(arr.shape[0] - 1):
        a = arr[i, :].reshape((1, -1))
        b = arr[i + 1, :].reshape((1, -1))
        interp = interpolate_two_points(a, b, n_steps=frames_per_interpolation)
        interps.append(interpolate_two_points(a, b, n_steps=frames_per_interpolation))
    return np.vstack(interps)


def main(run_id, input_file, output_dir, duration, mp4_fps=30):
    if not input_file.exists():
        raise FileNotFoundError(
            "Could not locate input file: '{input_file.as_posix()}'"
        )
    else:
        logger.info("Input file exists.")

    if not output_dir.exists():
        logger.info("Creating output directory: '{output_dir.as_posix()}'")
        output_dir.mkdir()
    else:
        logger.info("Output directory already exists")

    input_array = np.loadtxt(input_file)
    logger.info(f"Input array shape: {input_array.shape}")
    num_frames = int(np.rint(duration * mp4_fps))
    frames_per_interpolation = int(np.rint(num_frames / (input_array.shape[0] - 1)))
    movie_array = make_interpolation_array(input_array, frames_per_interpolation)
    movie_array = movie_array[:, None, :]
    logger.info(f"Final interpolation array shape: {movie_array.shape}")

    # random_state = np.random.RandomState(123)
    # G, D, Gs = misc.load_network_pkl(run_id, None)
    # shape = [num_frames, np.prod([1,1])] + Gs.input_shape[1:]
    # logger.info(f"shape: {shape}")
    # raise Exception("Hey there big boi")
    # all_latents = random_state.randn(*shape)
    # logger.info(f"latents shape: {all_latents.shape}")
    # all_latents = scipy.ndimage.gaussian_filter(all_latents, [1.0 * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    # logger.info(f"latents shape after filter: {all_latents.shape}")

    # raise Exception("on purpose")
    logger.info("Passing interpolating array to video making function.")
    generate_interpolation_video(
        run_id,
        duration_sec=duration,
        mp4_fps=mp4_fps,
        all_latents=movie_array,
        output_dir=output_dir,
        apply_smoothing_operations=False,
    )


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=int)
    parser.add_argument("input_array", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("-d", "--duration", type=float, default=30.0)
    parser.add_argument("-g", "--num-gpus", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()

    config.num_gpus = args.num_gpus
    logger.info(f"num. GPUS: {config.num_gpus}")

    # Setup from 'train.py'.
    misc.init_output_logging()
    np.random.seed(config.random_seed)
    logger.info("Initializing TensorFlow...")
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)

    main(args.run_id, args.input_array, args.output_dir, args.duration)
