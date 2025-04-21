import numpy as np
import tensorflow as tf
from keras import layers, models
from vit_keras import vit
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import matplotlib.pyplot as plt

#paths and params
train_dir = "Training"
test_dir = "Testing"
img_size = (224, 224)
batch_size = 32
num_classes = 4

#data gen
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15, #make some small changes to images 
    width_shift_range=0.1, #make some small changes to images 
    height_shift_range=0.1, #make some small changes to images 
    zoom_range=0.1, #make some small changes to images 
    horizontal_flip=True, #make some small changes to images 
    fill_mode='nearest',
    validation_split=0.2 #add validation (required for project)
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
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

#load ViT model backbone from vit_keras
vit_base = vit.vit_b16(
    image_size=img_size[0],
    activation='softmax',
    pretrained=True,
    include_top=False,
    pretrained_top=False
)

#build class model
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

#train
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
]

history = vit_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks
)

#eval
test_loss, test_acc, test_recall = vit_model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Recall: {test_recall:.4f}")

#predict
Y_pred = vit_model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

macro_recall = recall_score(y_true, y_pred, average='macro')
print(f"Macro Recall (sklearn): {macro_recall:.4f}")

#plot
plt.figure()
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('ViT Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ViT Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['recall'], label='train_recall')
plt.plot(history.history['val_recall'], label='val_recall') #using recall as our primary indicator
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('ViT Recall')
plt.legend()
plt.show()

#class report
labels = list(train_generator.class_indices.keys())
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
