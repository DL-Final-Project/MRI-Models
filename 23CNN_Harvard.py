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

# Limit TensorFlow to use up to 8 threads
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(4)

def create_df(image_path):
    classes, class_paths = zip(*[(label, os.path.join(image_path, label, image))
                                 for label in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, label))
                                 for image in os.listdir(os.path.join(image_path, label))])

    image_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return image_df

# Set the directories in the project
harvard_df = create_df("HarvardDataset")

train_df, test_df = train_test_split(harvard_df, test_size=0.195, random_state=42, stratify=harvard_df['Class']) # 19.5% to match 30 in test set of original paper
train2_df, valid_df = train_test_split(train_df, test_size = 0.2, random_state=42, stratify=train_df['Class'])

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
img_size = (512, 512)

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
model = Sequential([
    # Block 1
    Conv2D(64, (22, 22), strides=2, input_shape=(512, 512, 3)),
    MaxPooling2D(pool_size=(4, 4)),
    BatchNormalization(),


    # Block 2
    Conv2D(128, (11, 11), strides=2, padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Block 3
    Conv2D(256, (7, 7), strides=2, padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Block 4
    Conv2D(512, (3, 3), strides=2, padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    GlobalAveragePooling2D(),
    Activation("relu"),

    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(2, activation='softmax')  # 2 classes: normal, abnormal
])

model.compile(
    optimizer=Adam(),
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
test_metrics = model.evaluate(test_generator, return_dict = True)
print(f"Test Loss: {test_metrics['loss']:.4f}")
print(f"Test Recall: {test_metrics['recall']:.4f}")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

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
os.makedirs('temp/harvard/23CNN', exist_ok=True)
plt.savefig(f'temp/harvard/23CNN/23cnn_accuracy.png')
plt.close()

plt.figure()
plt.plot(history.history['recall'], label='train_recall')
plt.plot(history.history['val_recall'], label='val_recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.ylim(0, 1)
plt.title('Training vs Validation Recall')
plt.legend()
plt.savefig(f'temp/harvard/23CNN/23cnn_recall.png')
plt.close()

plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.savefig(f'temp/harvard/23CNN/23cnn_val_loss.png')
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
plt.savefig(f'temp/harvard/23CNN/23cnn_cm_display.png')

model.save('23CNN_Harvard.keras')
