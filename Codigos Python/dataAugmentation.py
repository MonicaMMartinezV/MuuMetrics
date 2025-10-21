import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf # type: ignore
from tensorflow.keras import layers # type: ignore
from pathlib import Path


baseDir = "../Codigos Python/Batches para correr el codigo/04. Batch Imagenes Clasificadas DataAugmentation"

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
    sampleImages = images[:5]  # take 5 example images
    break

# Apply augmentation
augmentedImages = dataAugmentation(sampleImages)

# Plot original vs augmented
plt.figure(figsize=(10, 6))

for i in range(5):
    # Original
    plt.subplot(2, 5, i + 1)
    plt.imshow(
        sampleImages[i].numpy().astype("uint8")) # Fixed: Access i-th image from sampleImages
    plt.title("Original")
    plt.axis("off")

    # Augmented
    plt.subplot(2, 5, i + 6)
    plt.imshow((augmentedImages[i].numpy() * 255).astype("uint8"))
    plt.title("Augmented")
    plt.axis("off")

plt.tight_layout()
plt.show()

import os
import shutil
import tensorflow as tf # type: ignore

# Define directories to save images
outputDir = str(Path(baseDir) / 'output')
trainOutputDir = os.path.join(outputDir, 'train')
valOutputDir = os.path.join(outputDir, 'val')
augmentedOutputDir = os.path.join(outputDir, 'augmented')

# Create directories if they don't exist
os.makedirs(trainOutputDir, exist_ok=True)
os.makedirs(valOutputDir, exist_ok=True)
os.makedirs(augmentedOutputDir, exist_ok=True)

# Function to save images from a dataset
def saveImagesFromDataset(dataset, outputDirectory, maxImages=None, applyAugmentation=False, augmentationLayer=None):
    count = 0
    # Create a mapping from original labels to new labels
    originalLabels = sorted(trainDs.class_names) # Assuming trainDs contains all class names
    labelMapping = {originalLabels[i]: 2 + i * 0.25 for i in range(len(originalLabels))}


    for images, labels in dataset:
        if applyAugmentation and augmentationLayer is not None:
            images = augmentationLayer(images)

        for i in range(images.shape[0]):
            if maxImages is not None and count >= maxImages:
                return
            # Ensure image data is in the correct format before converting to PIL Image
            if images[i].dtype != tf.uint8:
                imgData = tf.cast(images[i] * 255, tf.uint8)
            else:
                imgData = images[i]

            img = tf.keras.utils.array_to_img(imgData)
            originalLabelIndex = labels[i].numpy()
            originalLabelName = trainDs.class_names[originalLabelIndex]
            newLabel = labelMapping[originalLabelName]

            # Use the new label for the directory name, converting to string
            classDir = os.path.join(outputDirectory, str(newLabel))
            os.makedirs(classDir, exist_ok=True)
            imgPath = os.path.join(classDir, f'image_{count}.png')
            img.save(imgPath)
            count += 1
        if maxImages is not None and count >= maxImages:
            break



# Save augmented images from the entire train dataset
print("Saving augmented train images...")
saveImagesFromDataset(trainDs, augmentedOutputDir, applyAugmentation=True, augmentationLayer=dataAugmentation)


print("Image saving complete.")