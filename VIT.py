import numpy as np
import tensorflow as tf
from keras import layers, models
from vit_keras import vit
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# handles possible mem issues for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled memory growth on GPU")
    except RuntimeError as e:
        print(e)

'''
create_df - take a directory and return a dataframe with data and labels

parameters  image path  the directory of images to parse
returns     image_df    dataframe with data and labels

'''
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
img_size = (224, 224)
num_classes = 4

# evaluation generator for train
train_generator_for_pred = eval_datagen.flow_from_dataframe(
    train_df,
    x_col='Class Path',
    y_col='Class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# training set using data augmentation
train2_generator = train_datagen.flow_from_dataframe(
    train2_df,
    x_col='Class Path',
    y_col='Class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# evaluation generator for validation
valid_generator = eval_datagen.flow_from_dataframe(
    valid_df,
    x_col='Class Path',
    y_col='Class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # important so predictions and labels align
)

# evaluation generator for test
test_generator = eval_datagen.flow_from_dataframe(
    test_df,
    x_col='Class Path',
    y_col='Class',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # important so predictions and labels align
)

# load ViT model backbone
vit_base = vit.vit_b16(
    image_size=img_size[0],
    activation='softmax',
    pretrained=True,
    include_top=False,
    pretrained_top=False
)

# build full model
inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
x = vit_base(inputs)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
vit_model = models.Model(inputs=inputs, outputs=outputs)

vit_model.compile(
    optimizer=Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
)

vit_model.summary()

# early stopping and learning rate reduction — monitor training loss now
callbacks = [
    EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, verbose=1)
]

# train without validation
history = vit_model.fit(
    train2_generator,
    epochs=40,
    callbacks=callbacks,
    validation_data = valid_generator
)

# save models for future use
os.makedirs('models', exist_ok=True)
vit_model.save('models/vit_model.h5')


# evaluate on test set
test_loss, test_acc, test_recall = vit_model.evaluate(test_generator)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# evaluate on train set
train_loss, train_acc, train_recall = vit_model.evaluate(train_generator_for_pred)
print(f"\nTrain Loss: {train_loss:.4f}")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Train Recall: {train_recall:.4f}")

# prediction on test set
Y_pred_test = vit_model.predict(test_generator)
y_pred_test = np.argmax(Y_pred_test, axis=1)
y_true_test = test_generator.classes


Y_pred_train = vit_model.predict(train_generator_for_pred)
y_pred_train = np.argmax(Y_pred_train, axis=1)
y_true_train = train_generator_for_pred.classes

# labels
labels = list(train2_generator.class_indices.keys())

# training classification report and confusion matrix
print("\n--- Classification Report: TRAINING DATA ---")
print(classification_report(y_true_train, y_pred_train, target_names=labels))

print("--- Confusion Matrix: TRAINING DATA ---")
print(confusion_matrix(y_true_train, y_pred_train))

# test classification report and confusion matrix
print("\n--- Classification Report: TEST DATA ---")
print(classification_report(y_true_test, y_pred_test, target_names=labels))

print("--- Confusion Matrix: TEST DATA ---")
print(confusion_matrix(y_true_test, y_pred_test))

# macro recall score for test set
macro_recall = recall_score(y_true_test, y_pred_test, average='macro')
print(f"Macro Recall (sklearn): {macro_recall:.4f}")

# plot Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('ViT Accuracy')
plt.legend()
os.makedirs('temp', exist_ok=True)
os.makedirs('temp/vit', exist_ok=True)
plt.savefig(f'temp/vit/vit_accuracy.png')
plt.close()

# plot Loss
plt.figure()
plt.plot(history.history['loss'], label='ViT Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ViT Loss')
plt.legend()
plt.savefig(f'temp/vit/vit_loss.png')
plt.close()

# plot Recall
plt.figure()
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['recall'], label='Val Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('ViT Recall over Epochs')
plt.legend()
plt.savefig(f'temp/vit/vit_recall.png')
plt.close()

# Confusion matrix
cm_test = confusion_matrix(y_true_test, y_pred_test)
print("Confusion Matrix ViT (Test Set):")
print(cm_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels = labels)
disp.plot()
plt.savefig(f'temp/vit/test_vit_cm_display.png')

# Confusion matrix
cm_train = confusion_matrix(y_true_train, y_pred_train)
print("Confusion Matrix ViT (Train set):")
print(cm_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels = labels)
disp.plot()
plt.savefig(f'temp/vit/train_vit_cm_display.png')