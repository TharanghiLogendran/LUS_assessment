""" Plot data distribution figures
Tharanghi Logendran, tharanghi@gmail.com"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data overview
f = open("D:/TGM3/Video_classification/Data/Sessie_3/all_sessie_3.txt", "r")
temp = f.read()
videos = temp.split('\n')

# creating a dataframe having video names
all_df = pd.DataFrame()
all_df['video_name'] = videos
all_df.head()

#creating tags for training videos
all_video_tag = []
for i in range(all_df.shape[0]):
    all_video_tag.append(all_df['video_name'][i].split('_')[1])

all_df['tag'] = all_video_tag
frequency = all_df['tag'].value_counts()
frequency = frequency.reindex(["ntb", "score0", "score1", "score2", "score3"])

classes = ["ID", "Score 0", "Score 1", "score 2", "Score 3"]

# creating the bar plot
plt.bar(classes, frequency, color='maroon',
        width=0.4)
plt.xlabel("LUS classes")
plt.ylabel("Number of videos")
plt.title("LUS videos for each score")
plt.show()
#%%
x = np.array([100,200,300,400,500,600,700,800,900,1000,1100,1200,])
y = np.array([7.626,3.549,2.951,3.250,2.675,2.360,2.004,1.939,1.795,2.297,2.684,1.932])

plt.scatter(x, y, c='maroon')
plt.xlabel('Batch number')
plt.ylabel('Generator loss')
plt.show()
