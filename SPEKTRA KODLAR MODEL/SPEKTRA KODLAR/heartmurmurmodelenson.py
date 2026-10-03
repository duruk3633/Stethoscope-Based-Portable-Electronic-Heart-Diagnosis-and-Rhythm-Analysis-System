import numpy as np
import librosa
import glob
import os
import tensorflow as tf
from tensorflow.python.keras import layers, Sequential

def ozellikcikart(dosyayolu):
    signal, sr = librosa.load(dosyayolu, sr=None)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfccs = np.expand_dims(mfccs, axis=-1)  # Conv1D için kanal boyutu ekledik
    return mfccs

EPOCH = 50
BATCHSIZE = 32
train_data_dir = r"C:\Users\adaas\Downloads\heart_sounds_thechallenge\heart_sound\train"
val_data_dir = r"C:\Users\adaas\Downloads\heart_sounds_thechallenge\heart_sound\val"

def veriyukle(data_dir):
    features = []
    labels = []
    for label_dir in glob.glob(os.path.join(data_dir, '*')):
        label = os.path.basename(label_dir)
        for dosyayolu in glob.glob(os.path.join(label_dir, '*.wav')):
            features.append(ozellikcikart(dosyayolu))
            labels.append(label)
    return np.array(features), np.array(labels)

X_train, y_train = veriyukle(train_data_dir)
X_val, y_val = veriyukle(val_data_dir)

unique_labels, y_train_encoded = np.unique(y_train, return_inverse=True)
_, y_val_encoded = np.unique(y_val, return_inverse=True)

# Model Tanımlama
model = Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),

    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    
    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_encoded, epochs=EPOCH, batch_size=BATCHSIZE, validation_data=(X_val, y_val_encoded))

loss, accuracy = model.evaluate(X_val, y_val_encoded)
print(f"Doğruluk Oranı: {accuracy:.2f}")

model.save('kalpufurum.h5')

donusturucu = tf.lite.TFLiteConverter.from_keras_model(model)
donusturucu.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = donusturucu.convert()

with open('kalpufurummodel.tflite', 'wb') as f:
    f.write(tflite_model)
