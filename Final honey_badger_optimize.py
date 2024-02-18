import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import wfdb
import numpy as np
import os
from scipy import fftpack
import scipy.signal as signal
import tensorflow as tf
import random
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import psutil

def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory Used: {memory_info.rss / (1024 * 1024):.2f} MB")
print("Memory usage before processing:")
print_memory_usage()
# Ensure you are using TensorFlow 2.x
print("TensorFlow version:", tf.__version__)
assert "2." in tf.__version__, "This code requires TensorFlow 2.x"

data_directory = 'C:\\Users\\RayanSabz\\Desktop\\cccccc\\mit-bih-normal-sinus-rhythm-database-1.0.0'
# Load ECG records
ecg_records = {}
for filename in os.listdir(data_directory):
    if filename.endswith('.dat') or filename.endswith('.hea') or filename.endswith('.atr')  :
        record_name = os.path.splitext(filename)[0]
        record_path = os.path.join(data_directory, record_name)

        record = wfdb.rdrecord(record_path)
        signals = record.p_signal
        ecg_records[record_name] = signals

for record_name, signals in ecg_records.items():
    print(f"Record Name: {record_name}")
    print(f"Signals shape: {signals.shape}")
    print(signals)
    break  

##########

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y
def highpass_filter(data, cutoff=0.5, fs=360, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y
def lowpass_filter(data, cutoff=50, fs=360, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y
# Filtering
fs = 360 
lowcut = 0.5
highcut = 50
for record_name in ecg_records:
    for channel in range(ecg_records[record_name].shape[1]):
        ecg_records[record_name][:, channel] = bandpass_filter(ecg_records[record_name][:, channel], lowcut, highcut, fs)
        ecg_records[record_name][:, channel] = highpass_filter(ecg_records[record_name][:, channel], cutoff=lowcut, fs=fs)
        ecg_records[record_name][:, channel] = lowpass_filter(ecg_records[record_name][:, channel], cutoff=highcut, fs=fs)

for record_name, signals in ecg_records.items():
    print(f"Record Name: {record_name}")
    print(f"Signals shape: {signals.shape}")
    print(signals[:10])  
    break  

########

def extract_features_dct(signal):
    return fftpack.dct(signal, norm='ortho')

def extract_features_fft(signal):
    return np.abs(fftpack.fft(signal))
# Feature extraction
ecg_features = {}
for record_name, signals in ecg_records.items():
    features_dct = np.array([extract_features_dct(channel) for channel in signals.T])
    features_fft = np.array([extract_features_fft(channel) for channel in signals.T])
    ecg_features[record_name] = {
        'dct': features_dct.T, 
        'fft': features_fft.T
    }
for record_name, features in ecg_features.items():
    print(f"Record Name: {record_name}")
    print("DCT Features Shape:", features['dct'].shape)
    print("FFT Features Shape:", features['fft'].shape)
    break 

############
# DNLMS Algorithm
def dnlms_algorithm(input_signal, desired_signal, step_size=0.01, filter_length=10, delay=1, eps=1e-6):
    n_samples = len(input_signal)
    filter_weights = np.zeros(filter_length)
    error_signal = np.zeros(n_samples)

    for i in range(delay, n_samples):
        input_vector = input_signal[i-delay:i-delay-filter_length:-1]
        predicted_signal = np.dot(input_vector, filter_weights)
        error = desired_signal[i] - predicted_signal
        error_signal[i] = error
        power = np.dot(input_vector, input_vector) + eps
        filter_weights += 2 * step_size * error * input_vector / power

    return filter_weights, error_signal

def create_faster_rcnn_model(input_shape):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    x = Dense(6, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def evaluate_model(model, data, labels):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    loss, accuracy = model.evaluate(data, labels, verbose=1)
    return accuracy

def honey_badger_optimize(model, data, labels, iterations=10):
    best_model = tf.keras.models.clone_model(model)
    best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    best_score = evaluate_model(best_model, data, labels)

    for i in range(iterations):
        candidate_model = tf.keras.models.clone_model(model)
        candidate_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        candidate_model.set_weights(model.get_weights())

        # Randomly perturb the weights of the model
        for layer in candidate_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                weights, biases = layer.get_weights()
                weights += np.random.normal(0, 0.1, weights.shape)
                biases += np.random.normal(0, 0.1, biases.shape)
                layer.set_weights([weights, biases])

        candidate_score = evaluate_model(candidate_model, data, labels)

        if candidate_score > best_score:
            best_model = candidate_model
            best_score = candidate_score

    return best_model

def split_ecg_data_to_images(ecg_data, segment_length=1000, image_height=40):

    num_samples, num_channels = ecg_data.shape
    num_segments = num_samples // segment_length

    segments = np.array(np.split(ecg_data[:num_segments * segment_length], num_segments))

    images = segments.reshape(num_segments, image_height, segment_length // image_height, num_channels)
    return images

record_labels = {
    "16483": 0,  # Abnormal
    "16773": 1,  # Normal
    "17453": 1,  # Normal
    "16273": 0,  # Abnormal
    "18184": 0,  # Abnormal
    "19093": 1,  # Normal
    "16265": 1,  # Normal
    "17052": 0,  # Abnormal
    "19090": 1,  # Normal
    "19830": 0,  # Abnormal
    "16795": 1,  # Normal
    "18177": 0,  # Abnormal
    "16539": 1,  # Normal
    "19140": 0,  # Abnormal
    "16420": 1,  # Normal
    "16272": 0,  # Abnormal
}

# Preparing data and labels
data_list = []
labels_list = []

for record_name, signals in ecg_records.items():
    ecg_images = split_ecg_data_to_images(signals)
    data_list.append(ecg_images)

    label = record_labels.get(record_name, 0) # default label if not found in the dictionary
    labels = np.full(ecg_images.shape[0], label)
    labels_list.append(labels)

data = np.concatenate(data_list, axis=0)
labels = np.concatenate(labels_list, axis=0)

# Model creation and optimization
input_shape = data.shape[1:]
model = create_faster_rcnn_model(input_shape)
model.summary()

optimized_model = honey_badger_optimize(model, data, labels)


# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train the model
optimized_model.fit(train_data, train_labels, epochs=12, batch_size=16, verbose=1)

predicted_labels = optimized_model.predict(test_data)
predicted_labels = (predicted_labels > 0.5).astype(int)


# Evaluate the model
accuracy = accuracy_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels)
recall = recall_score(test_labels, predicted_labels)
f1 = f1_score(test_labels, predicted_labels)

import matplotlib.pyplot as plt

def plot_history(history):
    """Plot the accuracy and loss graphs for the training history."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')  
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1])
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')  
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


history = model.fit(train_data, train_labels, epochs=12, batch_size=16, verbose=1,
                    validation_data=(test_data, test_labels))



plot_history(history)

predicted_labels = optimized_model.predict(test_data)
predicted_labels = (predicted_labels > 0.5).astype(int)

accuracy = accuracy_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels)
recall = recall_score(test_labels, predicted_labels)
f1 = f1_score(test_labels, predicted_labels)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
conf_matrix = confusion_matrix(test_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


n_splits = 10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_accuracy = []
fold_precision = []
fold_recall = []
fold_f1 = []
fold_loss = []

for fold, (train_index, test_index) in enumerate(kfold.split(data, labels)):
    print(f"Training on fold {fold+1}/{n_splits}...")

    train_data, test_data = data[train_index], data[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

    model = create_faster_rcnn_model(input_shape)

    history = model.fit(train_data, train_labels, epochs=12, batch_size=16, verbose=1,
                        validation_data=(test_data, test_labels))

    predicted_labels = model.predict(test_data)
    predicted_labels = (predicted_labels > 0.5).astype(int)

    fold_accuracy.append(accuracy_score(test_labels, predicted_labels))
    fold_precision.append(precision_score(test_labels, predicted_labels))
    fold_recall.append(recall_score(test_labels, predicted_labels))
    fold_f1.append(f1_score(test_labels, predicted_labels))
    fold_loss.append(history.history['val_loss'][-1])

    plot_history(history)

avg_accuracy = np.mean(fold_accuracy)
avg_precision = np.mean(fold_precision)
avg_recall = np.mean(fold_recall)
avg_f1 = np.mean(fold_f1)
avg_loss = np.mean(fold_loss)

print(f'Average Accuracy: {avg_accuracy:.2f}')
print(f'Average Precision: {avg_precision:.2f}')
print(f'Average Recall: {avg_recall:.2f}')
print(f'Average F1 Score: {avg_f1:.2f}')
print(f'Average Loss: {avg_loss:.2f}')
print("Memory usage after processing:")
print_memory_usage()

