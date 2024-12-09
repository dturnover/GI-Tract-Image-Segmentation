""" Import statements and check for GPU """

import os
import re
import glob
import math
import cv2
import csv 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from transunet import TransUNet

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from tensorflow import keras
from tensorflow.keras import layers

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("GPUs: ", gpus)

if gpus:
    print("TensorFlow is using the GPU.")
else:
    print("TensorFlow is not using the GPU.")





""" Function Definitions """

def rle_to_binary(rle, shape):
    """
    Decodes run length encoded masks into a binary image

    Parameters:
        rle (list): list containing the starts and lengths that make up each RLE mask
        shape (tuple): the original shape of the associated image
    """

    # Initialize a flat mask with zeros
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    if rle == '' or rle == '0':  # Handle empty RLE
        return mask.reshape(shape, order='C')

    # Decode RLE into mask
    rle_numbers = list(map(int, rle.split()))
    for i in range(0, len(rle_numbers), 2):
        start = rle_numbers[i] - 1  # Convert to zero-indexed
        length = rle_numbers[i + 1]
        mask[start:start + length] = 1

    # Reshape flat mask into 2D
    return mask.reshape(shape, order='C')



def custom_generator(gdf, dir, batch_size, target_size=(224, 224), test_mode=False):
    """
    Custom data generator that dynamically aligns images and masks using RLE decoding.
    
    Parameters:
        gdf (GroupBy): Grouped dataframe containing image IDs and RLEs.
        dir (str): Root directory of the dataset.
        batch_size (int): Number of samples per batch.
        target_size (tuple): Target size for resizing (default=(224, 224)).
        test_mode (bool): If True, yields one image and mask at a time.
    """

    ids = list(gdf.groups.keys())
    dir2 = 'train'

    while True:
        sample_ids = np.random.choice(ids, size=batch_size, replace=False)
        images, masks = [], []

        for id_num in sample_ids:
            # Get the dataframe rows for the current image
            img_rows = gdf.get_group(id_num)
            rle_list = img_rows['segmentation'].tolist()
            
            # Construct the file path for the image
            sections = id_num.split('_')
            case = sections[0]
            day = sections[0] + '_' + sections[1]
            slice_id = sections[2] + '_' + sections[3]
            
            pattern = os.path.join(dir, dir2, case, day, "scans", f"{slice_id}*.png")
            filelist = glob.glob(pattern)
            
            if filelist:
                file = filelist[0]
                image = cv2.imread(file, cv2.IMREAD_COLOR)
                if image is None:
                    print(f"Image not found: {file}")
                    continue  # Skip if the image is missing
                
                # Original shape of the image
                original_shape = image.shape[:2]

                # Resize the image
                resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

                # Decode and resize the masks
                mask = np.zeros((target_size[0], target_size[1], len(rle_list)), dtype=np.uint8)
                for i, rle in enumerate(rle_list):
                    if rle != '0':  # Check if the RLE is valid
                        decoded_mask = rle_to_binary(rle, original_shape)
                        resized_mask = cv2.resize(decoded_mask, target_size, interpolation=cv2.INTER_NEAREST)
                        mask[:, :, i] = resized_mask

                if test_mode:
                    # Return individual samples in test mode
                    yield resized_image[np.newaxis], mask[np.newaxis], pattern
                else:
                    images.append(resized_image)
                    masks.append(mask)

        if not test_mode:
            x = np.array(images)
            y = np.array(masks)
            yield x, y, None

       



""" Loss function: dice loss ignores negative class thus negating class imbalance issues """     

def dice_coef(y_true, y_pred, smooth=1e-6):
    # Ensure consistent data types
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return 1 - dice_coef(y_true, y_pred)





""" Construct pipeline """

# dir = '../path/Dataset'
dir = './Dataset'

target_size = 224
batch_size = 24
epochs = 124

# read the csv file into a dataframe. os.path.join makes code executable across operating systes
df = pd.read_csv(os.path.join('.', dir, 'train.csv'))
df['segmentation'] = df['segmentation'].fillna('0')

# split into training, testing and validation sets
train_ids, temp_ids = train_test_split(df.id.unique(), test_size=0.25, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

# convert dfs into groupby objects to make sure rows are grouped by id
train_grouped_df = df[df.id.isin(train_ids)].groupby('id')
val_grouped_df = df[df.id.isin(val_ids)].groupby('id')
test_grouped_df = df[df.id.isin(test_ids)].groupby('id')


# steps per epoch is typically train length / batch size to use all training examples
train_steps_per_epoch = math.ceil(len(train_ids) / batch_size)
val_steps_per_epoch = math.ceil(len(val_ids) / batch_size)
test_steps_per_epoch = math.ceil(len(test_ids) / batch_size)

# create the training and validation datagens
train_generator = custom_generator(train_grouped_df, dir, batch_size, (target_size, target_size))
val_generator = custom_generator(val_grouped_df, dir, batch_size, (target_size, target_size))
test_generator = custom_generator(test_grouped_df, dir, batch_size, (target_size, target_size), test_mode=True)





""" Build the model or load the trained model """

loading = True

if loading:
    weights_path = './impmodels/model_weights.h5'
    model = TransUNet(image_size=224, pretrain=False)
    model.load_weights(weights_path)
    model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])
else:
    # create the optimizer and learning rate scheduler
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate = 1e-3,
        # decay_steps=train_steps_per_epoch * epochs,
        decay_steps=epochs+2,
        alpha=1e-2 
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # create the U-net neural network
    model = TransUNet(image_size=target_size, freeze_enc_cnn=False, pretrain=True)
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=['accuracy'])

    # set up model checkpoints and early stopping
    checkpoints_path = os.path.join('Checkpoints', 'model_weights.h5')
    model_checkpoint = ModelCheckpoint(filepath=checkpoints_path, save_best_only=True, monitor='val_loss')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=8)

    # log the training to a .csv for reference
    csv_logger = CSVLogger('training_log.csv', append=True)

    history = model.fit(train_generator, validation_data=val_generator, steps_per_epoch=train_steps_per_epoch, validation_steps=val_steps_per_epoch, epochs=epochs, callbacks=[model_checkpoint, early_stopping, csv_logger])





""" Display some predictions """

preds = []
ground_truths = []
num_samples = 50

# Generate predictions and ground truths
for i in range(num_samples):
    # Fetch a batch from the test generator
    batch = next(test_generator)
    image, mask = batch  
    
    preds.append(model.predict(image))  # Predict using the model
    ground_truths.append(mask)

best_threshold = 0.99

# Apply the best threshold to all predictions
final_preds = [(pred >= best_threshold).astype(int) for pred in preds]

# Compute Dice loss for each prediction
for i in range(len(final_preds)):
    loss = dice_loss(ground_truths[i], final_preds[i])  
    print(f"Image {i + 1}: Dice Loss = {loss:.4f}")



def visualize_predictions(generator, model, num_samples=8, target_size=(224, 224)):
    """
    Visualize predictions vs. ground truths overlaid on original images.

    Parameters:
        generator (generator): Data generator
        model (Model): Trained segmentation model
        num_samples (int): Number of samples to visualize
        target_size (tuple): Target size for resizing (default=(224, 224)).
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        # Fetch one image and mask from the generator
        image_batch, mask_batch = next(generator)
        image = image_batch[0]  # Single image
        ground_truth = mask_batch[0]  # Corresponding ground truth mask

        # Ensure image is RGB
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB

        # Ensure ground truth is a single-channel binary mask
        if ground_truth.ndim == 3 and ground_truth.shape[-1] == 3:
            ground_truth = ground_truth[:, :, 0]  # Extract the first channel

        # Generate prediction
        raw_prediction = model.predict(image[np.newaxis])[0]  # Add batch dimension for prediction

        # Ensure prediction is single-channel
        if raw_prediction.ndim == 3 and raw_prediction.shape[-1] == 3:
            prediction = raw_prediction[:, :, 0]  # Extract the first channel
        else:
            prediction = raw_prediction
        prediction = (prediction >= 0.99).astype(np.uint8)  # Threshold prediction

        # Create overlays
        gt_overlay = image.copy()
        pred_overlay = image.copy()

        # Overlay ground truth in red
        gt_overlay[ground_truth == 1] = [255, 0, 0]

        # Overlay prediction in green
        pred_overlay[prediction == 1] = [0, 255, 0]

        # Plot original image, ground truth overlay, and prediction overlay
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Image {i + 1}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_overlay)
        axes[i, 1].set_title(f"Ground Truth Overlay {i + 1}")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_overlay)
        axes[i, 2].set_title(f"Prediction Overlay {i + 1}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


# Call the function with your test generator and trained model
visualize_predictions(test_generator, model, num_samples=24)





def binary_to_rle(binary_mask):
    """
    Converts a binary mask to RLE (Run-Length Encoding).
    """
    # Flatten mask in column-major order
    flat_mask = binary_mask.T.flatten()

    rle = []
    start = -1
    for i, val in enumerate(flat_mask):
        if val == 1 and start == -1:
            start = i
        elif val == 0 and start != -1:
            rle.extend([start + 1, i - start])
            start = -1
    if start != -1:
        rle.extend([start + 1, len(flat_mask) - start])
    
    return ' '.join(map(str, rle))



def save_predictions_to_csv(test_generator, model, output_csv_path):
    """
    Generates predictions using the trained model and writes them to a CSV file in RLE format.
    
    Parameters:
        test_generator: The data generator for the test set.
        model: The trained segmentation model.
        output_csv_path: Path to save the CSV file.
    """
    
    with open(output_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id', 'segmentation'])  # Header row

        for image, masks, ids in test_generator:
            predictions = model.predict(image)
            predictions = (predictions > 0.99).astype(int)  

            for pred_mask, mask_id in zip(predictions, ids):
                rle = binary_to_rle(pred_mask.squeeze())
                csv_writer.writerow([mask_id, rle])

            print(f"Processed {len(ids)} predictions...")



save_predictions_to_csv(test_generator, model, 'model_output.csv')