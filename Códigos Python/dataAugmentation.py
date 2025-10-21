import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf # type: ignore
from tensorflow.keras import layers # type: ignore



baseDir = 'C:/Users/majos/Documents/cabus'

# Load train/validation datasets
trainDs = tf.keras.utils.image_dataset_from_directory(
    baseDir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(1080, 1080)
)

valDs = tf.keras.utils.image_dataset_from_directory(
    baseDir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(1080, 1080)
)

# Normalize + augment
dataAugmentation = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomRotation(0.4),
    layers.RandomFlip("horizontal"),
    layers.Lambda(lambda x: tf.tile(tf.image.rgb_to_grayscale(x), [1, 1, 1, 3]))
])

# Take one batch from the dataset
for images, labels in trainDs.take(1):
    sample_images = images[:5]  # take 5 example images
    break

# Apply augmentation
augmented_images = dataAugmentation(sample_images)

# Plot original vs augmented
plt.figure(figsize=(10, 6))

for i in range(5):
    # Original
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i].numpy().astype("uint8"))
    plt.title("Original")
    plt.axis("off")

    # Augmented
    plt.subplot(2, 5, i + 6)
    plt.imshow((augmented_images[i].numpy() * 255).astype("uint8"))
    plt.title("Augmented")
    plt.axis("off")

plt.tight_layout()
plt.show()
