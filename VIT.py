import numpy as np
import tensorflow as tf
from keras import layers, models
from vit_keras import vit
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import matplotlib.pyplot as plt

#paths and parameters
train_dir = "Training"
test_dir = "Testing"
img_size = (224, 224)
batch_size = 32
num_classes = 4

# data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
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
    train_generator,
    epochs=20,
    callbacks=callbacks
)

# evaluate on test set
test_loss, test_acc, test_recall = vit_model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# prediction on test set
Y_pred_test = vit_model.predict(test_generator)
y_pred_test = np.argmax(Y_pred_test, axis=1)
y_true_test = test_generator.classes

# prediction on training set
train_generator_for_pred = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
Y_pred_train = vit_model.predict(train_generator_for_pred)
y_pred_train = np.argmax(Y_pred_train, axis=1)
y_true_train = train_generator_for_pred.classes

# labels
labels = list(train_generator.class_indices.keys())

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
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('ViT Accuracy')
plt.legend()
plt.show()

# plot Loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ViT Loss')
plt.legend()
plt.show()

# plot Recall
plt.figure()
plt.plot(history.history['recall'], label='Train Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('ViT Recall over Epochs')
plt.legend()
plt.show()
