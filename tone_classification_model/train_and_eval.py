import numpy as np
import matplotlib
import math
import os
from matplotlib import pyplot as plt
import librosa
import librosa.display
import matplotlib.cm as cm
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.utils import class_weight
import sklearn.metrics as metrics

def get_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(2, 2), activation='relu'),
        BatchNormalization(),
        Conv2D(48, kernel_size=(2, 2), activation='relu'),
        BatchNormalization(),
        Conv2D(120, kernel_size=(2, 2), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy']
    )
    
    return model

def mp3tomfcc(file_path, max_pad):
    audio, sample_rate = librosa.core.load(file_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
    pad_width = max_pad - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

DATA_PATH = 'chinese_tone_data'
FILE_MFCCS = 'mfccs_all.npy'
FILE_LABELS = 'mfccs_all_labels.npy'
OUTPUT_DIR = "weights"

if os.path.exists(FILE_MFCCS) and os.path.exists(FILE_LABELS):
    mfccs = np.load(FILE_MFCCS)
    labels = np.load(FILE_LABELS)
    print("Files already exist. Loading existing data.")
else:
    mfccs = []
    labels = []
    for f in os.listdir(DATA_PATH):
        if f.endswith('.mp3'):
            mfccs.append(mp3tomfcc(os.path.join(DATA_PATH, f), 60))
            labels.append(int(f.split('_')[0][-1]))

    mfccs = np.asarray(mfccs)
    np.save('mfccs_all.npy', mfccs)
    labels = to_categorical(labels, num_classes=None)
    np.save('mfccs_all_labels.npy', labels)

dim_1, dim_2 = mfccs.shape[1], mfccs.shape[2]
num_channels = 1
num_classes = 5

X = mfccs
print(f"Shape of X: {X.shape}")
X = X.reshape((mfccs.shape[0], dim_1, dim_2, num_channels))
print(f"Reshaped X into: {X.shape}")

y = labels
input_shape = (dim_1, dim_2, num_channels)

# Training code starts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model = get_cnn_model(input_shape, num_classes)
y_ints = np.argmax(y, axis=1)
# [y.argmax() for y in y_train]
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_ints),
    y=y_ints
)
class_weights = dict(enumerate(class_weights))
history = model.fit(X_train, y_train, batch_size=20, epochs=15, verbose=1, validation_split=0.2, class_weight=class_weights)
print(history)

model.save("tone_cnn.keras")

# Evaluation
print(f"Evaluate Test Set: {model.evaluate(X_test, y_test, batch_size = 3, verbose = 1)}")
print(f"Evaluate Train Set: {model.evaluate(X_train, y_train, batch_size = 3, verbose = 1)}")

y_pred = model.predict(X_test).ravel()
y_pred_ohe = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_ohe, axis=1)

y_true_labels = np.argmax(y_test, axis=1)

confusion_matrix = metrics.confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels)
print(confusion_matrix)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_plot.png')
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss_plot.png')
plt.close()
