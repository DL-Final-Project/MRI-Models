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
from tensorflow.keras.applications import ResNet50V2

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
train_df = create_df("Training")
test_df = create_df("Testing")

train2_df, valid_df = train_test_split(train_df, random_state=42, stratify=train_df['Class'])

# -----------------------------------------------------
# 1) Set Up Image Generators
# -----------------------------------------------------
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
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Create iterators
batch_size = 32
img_size = (224, 224)

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
# 2) Load the Resnet Model
# -----------------------------------------------------
base_model = ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224,3))

base_model.trainable = False

model = Sequential([
    # Block 1
    base_model,

    GlobalAveragePooling2D(),
    Activation("relu"),

    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(4, activation='softmax')  # 4 classes: glioma, meningioma, pituitary, no_tumor
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['recall', 'accuracy']
)

model.summary()

# -----------------------------------------------------
# 3) Set Callbacks
# -----------------------------------------------------
# EarlyStopping: stop if validation loss does not improve for patience epochs
# ReduceLROnPlateau: reduce the learning rate when validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

# -----------------------------------------------------
# 4) Train the Model Frozen then Unfrozen
# -----------------------------------------------------
epochs = 10

history_frozen = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

# unfreeze model and continue training at low LR
base_model.trainable = True
epochs = 25

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['recall', 'accuracy']
)

history_unfreeze = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

# Combine the histories
history = {}
for keys in history_frozen.history:
    print(history_frozen.history[keys])
    print(history_unfreeze.history[keys])
    history[keys] = history_frozen.history[keys] + history_unfreeze.history[keys]
    print(history[keys])

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
plt.plot(history['accuracy'], label='train_accuracy')
plt.plot(history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
os.makedirs('temp', exist_ok=True)
plt.savefig(f'temp/accuracy_plot.png')
plt.close()

plt.figure()
plt.plot(history['recall'], label='train_recall')
plt.plot(history['val_recall'], label='val_recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Training vs Validation Recall')
plt.legend()
os.makedirs('temp', exist_ok=True)
plt.savefig(f'temp/recall_plot.png')
plt.close()

plt.figure()
plt.plot(history['loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.savefig(f'temp/val_loss_plot.png')
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
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = list(train_generator.class_indices.keys()))
disp.plot()
plt.savefig(f'temp/cm_display.png')

# save the model for future use
model.save('resmodel.keras')
