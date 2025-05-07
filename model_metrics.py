import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

plt.switch_backend('agg')   # resolves a pyplot bug

def create_df(image_path):
    classes, class_paths = zip(*[(label, os.path.join(image_path, label, image))
                                 for label in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, label))
                                 for image in os.listdir(os.path.join(image_path, label))])

    image_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return image_df

# Set the directories in the project
train_df = create_df("Training")
test_df = create_df("Testing")

train2_df, valid_df = train_test_split(train_df, train_size=0.8, random_state=42, stratify=train_df['Class'])

# Use data augmentation on the training set to help the model generalize
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# For test data, just rescale
eval_datagen = ImageDataGenerator(rescale=1. / 255)

# Create iterators
batch_size = 32
img_size = (512, 512)

train_eval = eval_datagen.flow_from_dataframe(
    train_df,
    x_col='Class Path',
    y_col='Class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle = False
)

train2_generator = train_datagen.flow_from_dataframe(
    train2_df,
    x_col='Class Path',
    y_col='Class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = eval_datagen.flow_from_dataframe(
    valid_df,
    x_col='Class Path',
    y_col='Class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # important so predictions and labels align
)

test_generator = eval_datagen.flow_from_dataframe(
    test_df,
    x_col='Class Path',
    y_col='Class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # important so predictions and labels align
)


# Load a CNN Model
model = tf.keras.models.load_model('23CNN.keras')


# Evaluate on Test/Train Set

test_metrics = model.evaluate(test_generator, return_dict = True)
print(f"Test Loss: {test_metrics['loss']:.4f}")
print(f"Test Recall: {test_metrics['recall']:.4f}")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n")

train_metrics = model.evaluate(train_eval, return_dict = True)
print(f"Train Loss: {train_metrics['loss']:.4f}")
print(f"Train Recall: {train_metrics['recall']:.4f}")
print(f"Train Accuracy: {train_metrics['accuracy']:.4f}\n")

# get class labels
labels = list(train2_generator.class_indices.keys())

# Predict classes test set
Y_pred_test = model.predict(test_generator)
y_pred_test = np.argmax(Y_pred_test, axis=1)
y_true_test = test_generator.classes

# Predict classes train set
Y_pred_train = model.predict(train_eval)
y_pred_train = np.argmax(Y_pred_train, axis=1)
y_true_train = train_eval.classes

# Print classification report
print("Classification Report (Test Set):")
print(classification_report(y_true_test, y_pred_test, target_names=labels))
print("Classification Report (Train Set):")
print(classification_report(y_true_train, y_pred_train, target_names=labels))

# Confusion matrix test set
cm_test = confusion_matrix(y_true_test, y_pred_test)
print("Confusion Matrix 23CNN (Test Set):")
print(cm_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels = labels)
disp.plot()
#plt.savefig(f'temp/23CNN/test_23cnn_cm_display.png')

# Confusion matrix train set
cm_train = confusion_matrix(y_true_train, y_pred_train)
print("Confusion Matrix 23CNN (Train set):")
print(cm_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels = labels)
disp.plot()
#plt.savefig(f'temp/23CNN/train_23cnn_cm_display.png')

