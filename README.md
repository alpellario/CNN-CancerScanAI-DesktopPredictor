# CancerScanAI-DesktopPredictor 🎗️

Welcome to the CancerScanAI-DesktopPredictor repository. This project is dedicated to assisting in the early detection of breast cancer through an intuitive desktop application. Comprehensive Breast Cancer Detection suite combining advanced deep learning models with a user-friendly PyQT5 desktop interface for image-based diagnosis and visualization. This project showcases a full-stack approach from model training to practical application deployment.

## Screenshots
<img src="https://github.com/alpellario/CancerScanAI-DesktopPredictor/assets/74739828/37ad0df0-0619-4784-bea3-7e25188fa2d9" width="800">

<img src="https://github.com/alpellario/CancerScanAI-DesktopPredictor/assets/74739828/0e588d93-e5b1-4008-8d67-fc21b4887c2b" width="800">

<img src="https://github.com/alpellario/CancerScanAI-DesktopPredictor/assets/74739828/36e832ba-cd92-4a62-a0a7-e52eea2c2566" width="800">

<img src="https://github.com/alpellario/CancerScanAI-DesktopPredictor/assets/74739828/be600afe-7c18-4f55-ab57-6c1562589098" width="800">

<img src="https://github.com/alpellario/CancerScanAI-DesktopPredictor/assets/74739828/5182d036-f1de-4603-bbf5-a1a2ccf34459" width="800">

<img src="https://github.com/alpellario/CancerScanAI-DesktopPredictor/assets/74739828/4544f08d-9359-4cd2-9c60-5280ee393eef" width="800">




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


