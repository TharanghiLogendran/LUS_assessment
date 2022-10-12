""" Training script for LUS videos
Tutorial from https://keras.io/examples/vision/video_classification/"""

from tensorflow import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras import backend
#from tensorflow_docs.vis import embed
from imutils import paths

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
from keras import metrics
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import graphviz

# define hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 1

MAX_SEQ_LENGTH = 80
NUM_FEATURES = 2048

# storing the name of train and test videos in a dataframe
# open the .txt file that has names of training videos
f = open("I:/Onderzoek/Sabien Heisterkamp/Tharanghi-PedLUS/Data_per_score/test_batch/trainlist.txt", "r")
temp = f.read()
videos = temp.split('\n')

# creating a dataframe having video names
train_df = pd.DataFrame()
train_df['video_name'] = videos
train_df.head()

# open the .txt file that has names of test videos
f = open("I:/Onderzoek/Sabien Heisterkamp/Tharanghi-PedLUS/Data_per_score/test_batch/testlist.txt", "r")
temp = f.read()
videos = temp.split('\n')

# creating a dataframe having video names
test_df = pd.DataFrame()
test_df['video_name'] = videos
test_df.head()

#creating tags for training videos
train_video_tag = []
for i in range(train_df.shape[0]):
    train_video_tag.append(train_df['video_name'][i].split('_')[1])

train_df['tag'] = train_video_tag

# creating tags for test videos
test_video_tag = []
for i in range(test_df.shape[0]):
    test_video_tag.append(test_df['video_name'][i].split('_')[1])

test_df['tag'] = test_video_tag

print(f"Total videos for training: {len(train_df)}")

train_df.sample(10)

# Capture the frames of a video.
# Extract frames from the videos until a maximum frame count is reached.
# In the case where a video's frame count is less than the maximum frame count we will pad the video with zeros.
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

# use a pre-trained network to extract meaningful features from the extracted frames
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)

    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
)
print(label_processor.get_vocabulary())

def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video:
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels

# Path containing all videos
train_data, train_labels = prepare_all_videos(train_df, "I:/Onderzoek/Sabien Heisterkamp/Tharanghi-PedLUS/Data_per_score/")
test_data, test_labels = prepare_all_videos(test_df, "I:/Onderzoek/Sabien Heisterkamp/Tharanghi-PedLUS/Data_per_score/")

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")


# Utility for our sequence model.
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # mask: Binary tensor of shape [samples, timesteps] indicating whether a given timestep should be masked
    # (optional, defaults to None). An individual True entry indicates that the corresponding timestep should be
    # utilized, while a False entry indicates that the corresponding timestep should be ignored.
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    x = keras.layers.Dense(1, use_bias=False)(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)
    opt = keras.optimizers.Adam(learning_rate=0.01)
    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]

    )
    plot_model(rnn_model, to_file='RNNmodel_plot.png', show_shapes=False, show_layer_names=True)
    return rnn_model

# Utility for running experiments.
def run_experiment():
    filepath = "I:/Onderzoek/Sabien Heisterkamp/Tharanghi-PedLUS/Data_per_score/tmp/video_classifier" # this is where the weights will be stored
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


_, sequence_model = run_experiment()

# Confusion matrix
x_test = test_data
y_true = test_labels
y_pred = sequence_model.predict(x_test)
y_pred = np.argmax(y_pred, axis=-1)

cm = confusion_matrix(y_true, y_pred)
ax = sns.heatmap(cm, annot=True, cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted LUS score')
ax.set_ylabel('True LUS score');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['NTB','Score 0', 'Score 1', 'Score 2', 'Score 3'])
ax.yaxis.set_ticklabels(['NTB','Score 0', 'Score 1', 'Score 2', 'Score 3'])

## Display the visualization of the Confusion Matrix.
plt.show()



