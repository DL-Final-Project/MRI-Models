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
from tensorflow.keras.applications import VGG16

#tf.config.threading.set_intra_op_parallelism_threads(8)
#tf.config.threading.set_inter_op_parallelism_threads(4)

def create_df(image_path):
    classes, class_paths = zip(*[(label, os.path.join(image_path, label, image))
                                 for label in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, label))
                                 for image in os.listdir(os.path.join(image_path, label))])

    image_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return image_df

# Set the directories in the project
harvard_df = create_df("HarvardDataset")


train_df, test_df = train_test_split(harvard_df, test_size=0.2, random_state=42, stratify=harvard_df['Class'], random_state=42) # 19.5% to match 30 in test set of original paper
train2_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Class'])

# -----------------------------------------------------
# 1) Set Up Image Generators
# -----------------------------------------------------
# Use data augmentation on the training set to help the model generalize
train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    vertical_flip=True)

# For test data, just rescale
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Create iterators
batch_size = 10
img_size = (256, 256)

train_generator = train_datagen.flow_from_dataframe(
    train2_df,
    x_col='Class Path',
    y_col='Class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = test_datagen.flow_from_dataframe(
    valid_df,
    x_col='Class Path',
    y_col='Class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # important so predictions and labels align
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='Class Path',
    y_col='Class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # important so predictions and labels align
)

# -----------------------------------------------------
# 2) Define a CNN Model
# -----------------------------------------------------
base_model = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(256,256,3))

base_model.trainable = False

model = Sequential([
    base_model, # VGG16
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax')  # 2 classes: Normal, Abnormal
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['recall','accuracy']
)

model.summary()

# -----------------------------------------------------
# 3) Set Callbacks (Optional)
# -----------------------------------------------------
# EarlyStopping: stop if validation loss does not improve for patience epochs
# ReduceLROnPlateau: reduce the learning rate when validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

# -----------------------------------------------------
# 4) Train the Model
# -----------------------------------------------------
epochs = 40

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs
)

# -----------------------------------------------------
# 5) Evaluate on Test Set
# -----------------------------------------------------
'''
test_metrics = model.evaluate(test_generator, return_dict = True)
print(f"Test Loss: {test_metrics['loss']:.4f}")
print(f"Test Recall: {test_metrics['recall']:.4f}")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
'''

# -----------------------------------------------------
# 6) Plot Training Curves
# -----------------------------------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Training vs Validation Accuracy')
plt.legend()
os.makedirs('temp', exist_ok=True)
os.makedirs('temp/harvard', exist_ok=True)
os.makedirs('temp/harvard/VGG16', exist_ok=True)
#plt.savefig(f'temp/harvard/VGG16/VGG16_accuracy.png')
plt.close()

plt.figure()
plt.plot(history.history['recall'], label='train_recall')
plt.plot(history.history['val_recall'], label='val_recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.ylim(0, 1)
plt.title('Training vs Validation Recall')
plt.legend()
#plt.savefig(f'temp/harvard/VGG16/VGG16_recall.png')
plt.close()

plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
#plt.savefig(f'temp/harvard/VGG16/VGG16_val_loss.png')
plt.close()

# -----------------------------------------------------
# 7) Generate Classification Report & Confusion Matrix
# -----------------------------------------------------
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Predict classes
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

# Print classification report
labels = list(train_generator.class_indices.keys())
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = labels)
disp.plot()
#plt.savefig(f'temp/harvard/VGG16/VGG16_cm_display.png')

model.save('VGG16_Harvard.keras')
