# CancerScanAI-DesktopPredictor 🎗️

Welcome to the CancerScanAI-DesktopPredictor repository. This project is dedicated to assisting in the early detection of breast cancer through an intuitive desktop application. Comprehensive Breast Cancer Detection suite combining advanced deep learning models with a user-friendly PyQT5 desktop interface for image-based diagnosis and visualization. This project showcases a full-stack approach from model training to practical application deployment.

## Screenshots
![app0_800x555](https://github.com/alpellario/CancerScanAI-DesktopPredictor/assets/74739828/6c1f4315-1091-4e11-97a0-2714c5d8d97e)
![app1_800x555](https://github.com/alpellario/CancerScanAI-DesktopPredictor/assets/74739828/b26f44c5-da42-426f-947a-3c2b1471c270)
![app2_800x555](https://github.com/alpellario/CancerScanAI-DesktopPredictor/assets/74739828/91081844-8c1c-421d-8a4c-88909d51737c)
![app3_800x555](https://github.com/alpellario/CancerScanAI-DesktopPredictor/assets/74739828/96d9f05a-a6f1-46eb-82e4-c329705fd981)
![app4_800x555](https://github.com/alpellario/CancerScanAI-DesktopPredictor/assets/74739828/b7397c70-e5b7-4696-92a0-9de8d9262cbd)
![app5_800x555](https://github.com/alpellario/CancerScanAI-DesktopPredictor/assets/74739828/7bc12f3b-c400-42dc-950e-d5e945165554)



## Description

CancerScanAI is developed using Python and PyQt5, with a model trained on a dataset from Kaggle, featuring 277,524 histopathology images (198,738 non-cancerous and 78,786 cancerous). The dataset preparation, model creation, and training process are detailed in the `Breast-Cancer-Model-Training.ipynb` file. For practical purposes, this repository includes only a sample of 40 images, with a link to the full dataset available [here](https://drive.google.com/file/d/1Vjup229O_VNnTIYEvXWas8SJx_EyIm0I/view?usp=sharing).

## Installation

1. **Setup Environment:** Begin by downloading Anaconda Navigator. Create a new environment in the Environments tab, selecting Python version 2.11.7. Wait for the default components to install.
2. **Install Dependencies:** Open a terminal in the newly created environment and run:
   ```
   pip install tensorflow==2.12.0 opencv-python matplotlib numpy PyQt5
   ```
3. **Launch Editor:** Ensure the new environment is selected in Anaconda Navigator's Home tab, then launch VSCode. In the editor, navigate to the root directory of CancerScanAI and start the application with:
   ```
   python ./app.py
   ```

## Customization

For UI customizations, download QT Designer from [this link](https://build-system.fman.io/qt-designer-download). Open the `breastCancerDetectionUI.ui` file for modifications and save your changes. To apply these changes to the project, convert the UI file to a Python file using the command:
```
pyuic5 -x breastCancerDetectionUI.ui -o breastCancerDetectionUI.py
```
Note: If you change the name of the `.py` file, remember to update the import statement in `app.py`.

## Usage

- **Initial Settings:** Begin by selecting the `cancer_detection_model.h5` file in the project root.
- **Dataset Tab:** Test images from `prepared_dataset_final/test` are displayed in a gallery, with pagination options. Images will have a green border if non-cancerous and red if cancerous.
- **Making Predictions:** Double-click an image or select and press "Make Prediction" to predict. The "Random 36 Images from Test Dataset" button allows for bulk predictions across random images.
- **Prediction Tab:** Shows the image, actual label, predicted label, and prediction confidence.
- **Charts:** View the model's Accuracy, Loss, and Confusion Matrix charts.


