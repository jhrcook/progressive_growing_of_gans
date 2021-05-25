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


def generate_fake_images_with_input_vectors(
    run_id, num_pngs=1000, input_array=None, output_dir=None
):
    print(f"Generating {num_pngs} images")
    network_pkl = misc.locate_network_pkl(run_id, snapshot=None)

    png_prefix = misc.get_id_string_for_network_pkl(network_pkl) + "-"
    random_state = np.random.RandomState(818)

    print(f"Loading network from {network_pkl}...")
    G, D, Gs = misc.load_network_pkl(run_id, snapshot=None)

    if output_dir is None:
        result_subdir = misc.create_result_subdir(
            config.result_dir, f"fake-images-{run_id}"
        )
    else:
        result_subdir = output_dir

    latent_vectors = []
    discriminator_values = []
    for png_idx in range(num_pngs):
        print(f"Generating png {png_idx} / {num_pngs}...")
        if input_array is not None:
            latents = input_array[png_idx, :].reshape(1, -1)
        else:
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
        scaled_images = (images - 127.5) / 127.5
        D_pred = D.run(scaled_images, resolution=1024)
        discriminator_values.append(D_pred[0][0])
        misc.save_image_grid(
            images,
            os.path.join(result_subdir, f"{png_prefix}{png_idx:08d}.png"),
            [0, 255],
            [1, 1],
        )

    latent_vectors = np.vstack(latent_vectors)
    print("Dimensions of latent vectors:")
    print(latent_vectors.shape)
    np.savetxt(os.path.join(result_subdir, "latent_vectors.txt"), latent_vectors)
    np.savetxt(
        os.path.join(result_subdir, "discriminator_values.txt"),
        np.hstack(discriminator_values),
    )
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


def evaluate_metrics(run_id, metric_name):
    if metric_name == "swd":
        print("Evaluation: SWD")
        util_scripts.evaluate_metrics(
            run_id=run_id,
            log="metric-swd-1k.txt",
            metrics=["swd"],
            num_images=1000,
            real_passes=2,
        )
        return None
    elif metric_name == "fid":
        print("Evaluation: FID")
        util_scripts.evaluate_metrics(
            run_id=run_id,
            log="metric-fid-10k.txt",
            metrics=["fid"],
            num_images=10000,
            real_passes=2,
        )
        return None
    elif metric_name == "is":
        print("Evaluation: IS")
        util_scripts.evaluate_metrics(
            run_id=run_id,
            log="metric-is-5k.txt",
            metrics=["is"],
            num_images=5000,
            real_passes=2,
        )
    elif metric_name == "msssim":
        print("Evaluation: MS-SSIM")
        util_scripts.evaluate_metrics(
            run_id=run_id,
            log="metric-msssim-5k.txt",
            metrics=["msssim"],
            num_images=5000,
            real_passes=2,
        )
        return None
    else:
        raise Exception(f"Unknown metric '{metric_name}'")


def reset_pggan_logger():
    misc.output_logger = None
    misc.init_output_logging()


def main(run_id, num_images, video_length, eval_metric):
    if num_images > 0:
        generate_fake_images_with_input_vectors(run_id, num_pngs=num_images)

    if video_length > 0:
        reset_pggan_logger()
        generate_interpolation_video(run_id, duration_sec=video_length)

    # reset_pggan_logger()
    # generate_training_video(run_id, duration_sec=10.0)

    if eval_metric != "none":
        reset_pggan_logger()
        evaluate_metrics(run_id, eval_metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=int)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--video-length", type=int, default=120)
    parser.add_argument("--eval-metric", type=str, default="none")
    args = parser.parse_args()

    config.num_gpus = args.num_gpus
    print(f"num. GPUS: {config.num_gpus}")

    # Configuration (copied from 'train.py').
    misc.init_output_logging()
    np.random.seed(config.random_seed)
    print("Initializing TensorFlow...")
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)

    main(args.run_id, num_images=args.num_images, video_length=args.video_length, eval_metric=args.eval_metric)
    print("done")
