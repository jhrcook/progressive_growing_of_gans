#!/usr/bin/env python3

import argparse
import bisect
from collections import OrderedDict
import os
import re
import time

import numpy as np
import scipy.ndimage
import scipy.misc
import tensorflow as tf

import config
import dataset
import misc
import tfutil
import train
import util_scripts


def generate_fake_images_with_input_vectors(run_id, num_pngs=1000):
    print(f"Generating {num_pngs} images")
    network_pkl = misc.locate_network_pkl(run_id, snapshot=None)

    png_prefix = misc.get_id_string_for_network_pkl(network_pkl) + "-"
    random_state = np.random.RandomState(818)

    print(f"Loading network from {network_pkl}...")
    G, D, Gs = misc.load_network_pkl(run_id, snapshot=None)

    result_subdir = misc.create_result_subdir(
        config.result_dir, f"fake-images-{run_id}"
    )
    latent_vectors = []
    for png_idx in range(num_pngs):
        print(f"Generating png {png_idx} / {num_pngs}...")
        latents = misc.random_latents(1, Gs, random_state=random_state)
        labels = np.zeros([latents.shape[0], 0], np.float32)
        latent_vectors.append(latents.copy().flatten())
        images = Gs.run(
            latents,
            labels,
            minibatch_size=8,
            num_gpus=config.num_gpus,
            out_mul=127.5,
            out_add=127.5,
            out_shrink=1,
            out_dtype=np.uint8,
        )
        misc.save_image_grid(
            images,
            os.path.join(result_subdir, "%s%06d.png" % (png_prefix, png_idx)),
            [0, 255],
            [1, 1],
        )

    latent_vectors = np.vstack(latent_vectors)
    np.savetxt(os.path.join(result_subdir, "latent_vectors.txt"), latent_vectors)
    open(os.path.join(result_subdir, "_done.txt"), "wt").close()
    return None


def generate_interpolation_video(run_id, duration_sec):
    original_desc = config.desc
    config.desc = f"interpolation-video-{run_id}"
    util_scripts.generate_interpolation_video(run_id, duration_sec=duration_sec)
    config.desc = original_desc
    return None

def generate_training_video(run_id, duration_sec):
    original_desc = config.desc
    config.desc = f"training-video-{run_id}"
    util_scripts.generate_training_video(run_id, duration_sec=duration_sec)
    config.desc = original_desc
    return None


def evaluate_metrics(run_id):
    print("Evaluation: SWD")
    util_scripts.evaluate_metrics(run_id=run_id, log="metric-swd-16k.txt", metrics=['swd'], num_images=16384, real_passes=2)
    reset_pggan_logger()
    print("Evaluation: FID")
    util_scripts.evaluate_metrics(run_id=run_id, log="metric-fid-10k.txt", metrics=["fid"], num_images=50000, real_passes=1)
    reset_pggan_logger()
    print("Evaluation: IS")
    util_scripts.evaluate_metrics(run_id=run_id, log='metric-is-50k.txt', metrics=['is'], num_images=50000, real_passes=1)
    reset_pggan_logger()
    print("Evaluation: MS-SSIM")
    util_scripts.evaluate_metrics(run_id=run_id, log='metric-msssim-20k.txt', metrics=['msssim'], num_images=20000, real_passes=1)
    return None


def reset_pggan_logger():
    misc.output_logger = None
    misc.init_output_logging()


def main(run_id):
    generate_fake_images_with_input_vectors(run_id, num_pngs=10)

    reset_pggan_logger()
    generate_interpolation_video(run_id, duration_sec=30.0)

    #reset_pggan_logger()
    #generate_training_video(run_id, duration_sec=10.0)

    reset_pggan_logger()
    evaluate_metrics(run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=int)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()
    run_id = args.run_id

    config.num_gpus = args.num_gpus
    print(f"num. GPUS: {config.num_gpus}")

    # Configuration (copied from 'train.py').
    misc.init_output_logging()
    np.random.seed(config.random_seed)
    print("Initializing TensorFlow...")
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)

    main(run_id)
    print("done")
