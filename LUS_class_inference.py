""" Testing script for LUS videos
Tutorial from https://keras.io/examples/vision/video_classification/"""

from tensorflow import keras
from tensorflow_docs.vis import embed
from imutils import paths

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
from keras import metrics
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# These are still required to create labels
# define hyperparameters
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# storing the name of train and test videos in a dataframe
# open the .txt file that has names of training videos
f = open("D:/TGM3/Video_classification/Data/Small_batch/trainlist.txt", "r")
temp = f.read()
videos = temp.split('\n')

# creating a dataframe having video names
train_df = pd.DataFrame()
train_df['video_name'] = videos
train_df.head()

# open the .txt file that has names of test videos
f = open("D:/TGM3/Video_classification/Data/Small_batch/testlist.txt","r")
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

# creating tags for test videos
test_video_tag = []
for i in range(test_df.shape[0]):
    test_video_tag.append(test_df['video_name'][i].split('_')[1])

test_df['tag'] = test_video_tag
train_df['tag'] = train_video_tag

# Capture the frames of a video.
# Extract frames from the videos until a maximum frame count is reached.
# In the case, where a video's frame count is lesser than the maximum frame count we will pad the video with zeros.
# let op: we nemen dus enkele frames uit de video
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

# Path to all videos (train and test)
train_data, train_labels = prepare_all_videos(train_df, "D:/TGM3/Video_classification/Data/Small_batch/")
test_data, test_labels = prepare_all_videos(test_df, "D:/TGM3/Video_classification/Data/Small_batch/")


# Utility for our sequence model.
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)
    opt = keras.optimizers.Adam(learning_rate=0.001)
    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]

    )

    return rnn_model

sequence_model = get_sequence_model()
# Load model weights for testing
sequence_model.load_weights("D:/TGM3/Video_classification/Data/Small_batch/tmp/video_classifier")

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()
    # Path to all videos
    # Change to path of single test video for prediction
    frames = load_video(os.path.join("I:/Onderzoek/Sabien Heisterkamp/Tharanghi-PedLUS/Data_per_score/testvideo/", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames

#%% inference
# Select random test video from test batch
test_video = np.random.choice(test_df["video_name"].values.tolist())


# Or select single test video from path
f = open("I:/Onderzoek/Sabien Heisterkamp/Tharanghi-PedLUS/Data_per_score/testvideo/testlist.txt","r")
temp = f.read()
videos = temp.split('\n')

testvideo_df = pd.DataFrame()
testvideo_df['video_name'] = videos
testvideo_df.head()

testvideo_video_tag = []
for i in range(testvideo_df.shape[0]):
    testvideo_video_tag.append(testvideo_df['video_name'][i].split('_')[1])

testvideo_df['tag'] = testvideo_video_tag

test_video = np.random.choice(testvideo_df["video_name"].values.tolist())
#%%
def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=10)
    return embed.embed_file("animation.gif")

print(f"Test video path: {test_video}")
test_frames = sequence_prediction(test_video) # make sure to change path in def sequence prediction!
to_gif(test_frames)

#%%
# Confusion matrix
x_test = test_data
y_true = test_labels
y_pred = sequence_model.predict(x_test)
y_pred = np.argmax(y_pred, axis=-1)

cm = confusion_matrix(y_true, y_pred)
ax = sns.heatmap(cm, annot=True, fmt="d")
ax.set_title('Seaborn Confusion Matrix with labels')
ax.set_xlabel('Predicted LUS score')
ax.set_ylabel('True LUS score')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['NTB','Score 0', 'Score 1', 'Score 2', 'Score 3'])
ax.yaxis.set_ticklabels(['NTB','Score 0', 'Score 1', 'Score 2', 'Score 3'])

## Display the visualization of the Confusion Matrix.
#plt.savefig("CM_all.jpg")
plt.show()

# Confusion matrix per class
mcm = multilabel_confusion_matrix(y_true, y_pred)
mcm_ntb = mcm[0]
mcm_score0 = mcm[1]
mcm_score1 = mcm[2]
mcm_score2 = mcm[3]
mcm_score3 = mcm[4]

# Plots
ax = sns.heatmap(mcm_ntb, annot=True)
ax.set_title('Confusion Matrix NTB')
ax.set_xlabel('Predicted LUS score')
ax.set_ylabel('True LUS score')
ax.xaxis.set_ticklabels(['Else','NTB'])
ax.yaxis.set_ticklabels(['Else','NTB'])
plt.show()

ax = sns.heatmap(mcm_score0, annot=True)
ax.set_title('Confusion Matrix Score 0')
ax.set_xlabel('Predicted LUS score')
ax.set_ylabel('True LUS score')
ax.xaxis.set_ticklabels(['Else','Score 0'])
ax.yaxis.set_ticklabels(['Else','Score 0'])
plt.show()

ax = sns.heatmap(mcm_score1, annot=True)
ax.set_title('Confusion Matrix Score 1')
ax.set_xlabel('Predicted LUS score')
ax.set_ylabel('True LUS score')
ax.xaxis.set_ticklabels(['Else','Score 1'])
ax.yaxis.set_ticklabels(['Else','Score 1'])
plt.show()

ax = sns.heatmap(mcm_score2, annot=True)
ax.set_title('Confusion Matrix Score 2')
ax.set_xlabel('Predicted LUS score')
ax.set_ylabel('True LUS score')
ax.xaxis.set_ticklabels(['Else','Score 2'])
ax.yaxis.set_ticklabels(['Else','Score 2'])
plt.show()

ax = sns.heatmap(mcm_score3, annot=True)
ax.set_title('Confusion Matrix Score 3')
ax.set_xlabel('Predicted LUS score')
ax.set_ylabel('True LUS score')
ax.xaxis.set_ticklabels(['Else','Score 3'])
ax.yaxis.set_ticklabels(['Else','Score 3'])
plt.show()
