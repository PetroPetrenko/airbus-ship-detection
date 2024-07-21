# Airbus Ship Detection Challenge

## Overview
This project aims to detect ships in satellite images using deep learning techniques. The model is trained on a dataset provided by Airbus and evaluated based on the F2 Score.

## Data
- Training images are located in the `data/train/` directory.
- The ground truth segmentation masks are provided in `train_ship_segmentations_v2.csv`.

## Model
The model used is a U-Net architecture, which is well-suited for image segmentation tasks.

## Training
To train the model, run the following command:

python src/train.py


## Requirements
Install the required packages using:

pip install -r requirements.txt

DA
The exploratory data analysis is documented in the Jupyter notebook located in the notebooks/ directory.
text

### 5. `notebooks/EDA.ipynb`

In this Jupyter Notebook, you can perform exploratory data analysis by visualizing images and their corresponding masks, analyzing the distribution of ship sizes, and understanding the dataset better.

### Conclusion

This structure and code provide a solid foundation for participating in the Airbus Ship Detection Challenge. Make sure to implement the RLE encoding and decoding functions, as well as any additional preprocessing steps as needed. This setup will also facilitate easy deployment and testing.


### Project Structure




airbus-ship-detection/
├── data/
│   ├── train/
│   └── test/
├── notebooks/
│   └── EDA.ipynb  
├── src/
│   ├── train.py
│   └── infer.py
├── utils/
│   ├── preprocessing.py
│   └── metrics.py
├── models/
│   └── unet.py
├── requirements.txt
└── README.md
