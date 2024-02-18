import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import wfdb
import numpy as np
import os
from scipy import fftpack
import scipy.signal as signal
import tensorflow as tf
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure you are using TensorFlow 2.x
print("TensorFlow version:", tf.__version__)
assert "2." in tf.__version__, "This code requires TensorFlow 2.x"

data_directory = 'C:\\Users\\RayanSabz\\Desktop\\New folder (2)\\mit-bih-normal-sinus-rhythm-database-1.0.0'
# Load ECG records
ecg_records = {}
for filename in os.listdir(data_directory):
    if filename.endswith('.dat') or filename.endswith('.hea') or filename.endswith('.atr'):
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

def dnlms_algorithm(input_signal, desired_signal, step_size=0.01, filter_length=10, delay=1, eps=1e-6):
    n_samples = len(input_signal)
    filter_weights = np.zeros(filter_length)
    error_signal = np.zeros(n_samples)

    for i in range(delay, n_samples):
        if i < filter_length + delay:  # اطمینان از اینکه input_vector اندازه کافی دارد
            continue

        input_vector = input_signal[i-delay:i-delay-filter_length:-1]
        predicted_signal = np.dot(input_vector, filter_weights)
        error = desired_signal[i] - predicted_signal
        error_signal[i] = error
        power = np.dot(input_vector, input_vector) + eps
        filter_weights += 2 * step_size * error * input_vector / power

    return filter_weights, error_signal

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
    x = Conv2D(3, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(6, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def evaluate_model(model, data, labels):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    loss, accuracy = model.evaluate(data, labels, verbose=0)
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

        # Evaluate the candidate model
        candidate_score = evaluate_model(candidate_model, data, labels)

        # Update the best model if the candidate model is better
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

    # Assign labels based on record name
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
optimized_model.fit(train_data, train_labels, epochs=25, batch_size=32, verbose=1)

# Predict labels for the test set
predicted_labels = optimized_model.predict(test_data)

# Convert probabilities to binary labels (threshold can be adjusted)
predicted_labels = (predicted_labels > 0.5).astype(int)
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# تعریف مدل SVM
svm_model = SVC(kernel='rbf', probability=True)

# آماده سازی داده‌ها برای SVM
# ممکن است لازم باشد داده‌ها را تغییر دهیم تا با مدل SVM سازگار باشند
svm_train_data = train_data.reshape(train_data.shape[0], -1)
svm_test_data = test_data.reshape(test_data.shape[0], -1)

# آموزش مدل SVM
svm_model.fit(svm_train_data, train_labels)

# ارزیابی مدل SVM
svm_predicted_labels = svm_model.predict(svm_test_data)
svm_accuracy = accuracy_score(test_labels, svm_predicted_labels)
svm_precision = precision_score(test_labels, svm_predicted_labels)
svm_recall = recall_score(test_labels, svm_predicted_labels)
svm_f1 = f1_score(test_labels, svm_predicted_labels)

print(f'SVM Accuracy: {svm_accuracy:.2f}')
print(f'SVM Precision: {svm_precision:.2f}')
print(f'SVM Recall: {svm_recall:.2f}')
print(f'SVM F1 Score: {svm_f1:.2f}')

# Evaluate the model
accuracy = accuracy_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels)
recall = recall_score(test_labels, predicted_labels)
f1 = f1_score(test_labels, predicted_labels)

import matplotlib.pyplot as plt

def plot_history(history):
    """Plot the accuracy and loss graphs for the training history."""
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Assuming `optimized_model` is already defined and compiled
# Fit the model and capture the history
history = optimized_model.fit(train_data, train_labels, 
                              epochs=25, batch_size=32, verbose=1, 
                              validation_split=0.2)  # Using part of the training data for validation

# Plot the training history
plot_history(history)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy by Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss by Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
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

# ارزیابی 10 برابری تاشو برای مدل CNN
cnn_scores = cross_val_score(model, data.reshape(data.shape[0], -1), labels, cv=10)

plt.figure(figsize=(10, 6))
plt.plot(cnn_scores, label='CNN Cross-Validation Scores')
plt.title('10-Fold Cross-Validation Scores for CNN')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.xticks(range(10), [f'Fold {i+1}' for i in range(10)])
plt.legend()
plt.show()
