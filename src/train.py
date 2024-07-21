import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.callbacks import EarlyStopping

# Load data
def load_data(image_dir, mask_file):
    masks = pd.read_csv(mask_file)
    images = []
    masks_list = []

    for img_id in masks['ImageId'].unique():
        img = cv2.imread(os.path.join(image_dir, img_id))
        img = cv2.resize(img, (256, 256))
        images.append(img)

        rle = masks[masks['ImageId'] == img_id]['EncodedPixels'].values
        mask = np.zeros((256, 256), dtype=np.uint8)

        for r in rle:
            if pd.isna(r):
                continue
            mask += rle_decode(r, (256, 256))

        masks_list.append(mask)

    return np.array(images), np.array(masks_list)

# RLE decoding
def rle_decode(mask_rle, shape):
    # Implement RLE decoding here
    pass

# U-Net model
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv_final = Conv2D(1, (1, 1), activation='sigmoid')(pool2)
    model = Model(inputs, conv_final)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Main function
if __name__ == "__main__":
    image_dir = 'data/train/'
    mask_file = 'train_ship_segmentations_v2.csv'

    images, masks = load_data(image_dir, mask_file)

    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    model = unet_model()
    early_stopping = EarlyStopping(patience=5, verbose=1)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, callbacks=[early_stopping])
