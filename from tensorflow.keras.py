from tensorflow.keras.preprocessing import image
import numpy as np
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from pyod.models.iforest import IForest
from Readfile_folder import ReadImage

data_folder_path="oathfansdfm;"
obj = ReadImage(data_folder_path)
img_arrays,image_path_all = obj.folder()


model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features using ResNet-50
features = model.predict(img_arrays)

# Flatten the features to feed into PyOD models
features_flatten = features.reshape(features.shape[0], -1)

# Standardize the feature vectors
scaler = StandardScaler()
features_norm = scaler.fit_transform(features_flatten)


# Initialize and process of building your outlier detection model
clf = IForest()
clf.fit(features_norm)

# Predict outliers
outlier_predictions = clf.predict(features_norm)

# Train a k-NN detector
clf = KNN()
clf.fit(features_norm)

# Get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

Isoutlier=[]
# Print the results
for i, (pred, score) in enumerate(zip(y_train_pred, y_train_scores)):
    print(f"Image {image_path_all[i]} is {'an outlier' if pred == 1 else 'normal'} with outlier score: {score}")
    if pred==1:
        Isoutlier.append(image_path_all[i])

