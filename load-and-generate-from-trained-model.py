#!/usr/bin/env python3

import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
from pathlib import Path

# Initialize TensorFlow session.
# tf.InteractiveSession()
tf_session = tf.Session().__enter__()
# tf_session.as_default()

trained_model_path = "results/001-pgan-mnist-preset-v2-2gpus-fp16/network-final.pkl"

with open(trained_model_path, "rb") as file:
    G, D, Gs = pickle.load(file)

latents = np.random.RandomState(1000).randn(100, *Gs.input_shapes[0][1:]) # 100 random latents

# Generate dummy labels (not used by the official networks).
labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

# Run the generator to produce a set of images.
images = Gs.run(latents, labels)

# Convert images to PIL-compatible format.
images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

# Save images as PNG.
image_dir = Path("example-images", "mnist")
if not image_dir.exists():
    image_dir.mkdir(parents=True)

for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx, :, :, 0]).save((image_dir / f"img{int(idx):04d}.png").as_posix())
