# 3-kinds-of-pneumonia-classification
classifying chest-Xray data in https://www.kaggle.com/datasets/artyomkolas/3-kinds-of-pneumonia
# Classification of X-Ray Images into Four Classes
# Introduction
- This project utilizes a dataset sourced from Kaggle, specifically designed for the classification of pneumonia types in chest X-ray images. The dataset, titled "3 Kinds of Pneumonia," contains a diverse collection of X-ray images that have been categorized into three classes of pneumonia: bacterial, viral, and healthy. For this project, I expanded the classification task to include a fourth class, which represents a specific category of pneumonia or a different lung condition. The objective of this study is to develop a robust machine learning model capable of accurately classifying X-ray images into these four distinct classes.
# Methodology
- The classification process involved several key steps:
- # Data Preparation: The dataset was downloaded from Kaggle, and the images were preprocessed to ensure uniformity in size and format. Data augmentation techniques were applied to enhance the model's ability to generalize by artificially expanding the dataset with variations of the existing images.
- # Model Selection: A convolutional neural network (CNN) architecture was chosen for this classification task due to its effectiveness in image recognition tasks. The model was trained using a labeled dataset, with a focus on optimizing accuracy and minimizing loss.
- # Training and Evaluation: The model was trained on a subset of the dataset, with a portion reserved for validation and testing. Performance metrics, including accuracy, precision, recall, and F1-score, were calculated to evaluate the model's effectiveness in classifying the X-ray images.
- # Results: The trained model demonstrated a high level of accuracy in classifying the X-ray images into the four designated classes. Detailed analysis of the confusion matrix and classification report highlighted areas of strength and potential improvement in the model's performance.
![CM{BASE_MODEL_NAME}](https://github.com/user-attachments/assets/c6c8b242-e9e8-4ac0-8b55-bbdb12bd5865)
![training_validation_loss_curvevit_32_simpleViT](https://github.com/user-attachments/assets/b5cac3e5-1b37-4b9f-b1d3-fc0a1cf827f9)
