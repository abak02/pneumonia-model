# Pneumonia Detection Model using Image Processing and Deep Learning
This is a pneumonia detection model that uses image processing and deep learning techniques to identify whether a chest X-ray image is normal or has pneumonia. The model was built using the TensorFlow deep learning framework and trained on the Chest X-Ray Images (Pneumonia) dataset.

# Dataset
The Chest X-Ray Images (Pneumonia) dataset was obtained from Kaggle and contains 5,856 images in total, including 3,856 images of patients with pneumonia and 2,000 normal chest X-ray images.

# Model Architecture
The model uses a convolutional neural network (CNN) architecture, specifically the VGG16 model, which has been pre-trained on the ImageNet dataset. The VGG16 model was fine-tuned on the Chest X-Ray Images (Pneumonia) dataset to improve its performance on pneumonia detection.

# Training and Evaluation
The model was trained using a batch size of 32 and was trained for 10 epochs. The model achieved an accuracy of 93% on the test set, indicating that it is capable of accurately detecting pneumonia in chest X-ray images.

# Usage
To use the pneumonia detection model, you will need to have TensorFlow installed on your machine. You can then clone the repository and run the model.py script, passing in the path to the chest X-ray image that you want to train.

$ python model.py
The script will train the datasets and generate a h5 model.

# Future Work
In the future, we plan to further improve the performance of the model by exploring other CNN architectures and incorporating additional data sources.
