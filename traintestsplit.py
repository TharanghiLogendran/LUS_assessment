""" Split data in train and test data"""

import random

def split_file(file, out1, out2, percentage=0.75, isShuffle=True, seed=123):
    """Splits a file in 2 given the `percentage` to go in the large file."""
    random.seed(seed)
    with open(file, 'r', encoding="utf-8") as fin, \
         open(out1, 'w') as foutBig, \
         open(out2, 'w') as foutSmall:

        nLines = sum(1 for line in fin) # if didn't count you could only approximate the percentage
        fin.seek(0)
        nTrain = int(nLines*percentage)
        nValid = nLines - nTrain

        i = 0
        for line in fin:
            r = random.random() if isShuffle else 0  # so that always evaluated to true when not isShuffle
            if (i < nTrain and r < percentage) or (nLines - i > nValid):
                foutBig.write(line)
                i += 1
            else:
                foutSmall.write(line)

# .txt file containing all video names
split_file("D:/TGM3/Video_classification/Data/Sessie_3/shuffled_all_sessie_3.txt", "train_3.txt", "test_3.txt", percentage=0.75, isShuffle=True, seed=123)

# for extra random shuffle before train/test split
with open('D:/TGM3/Video_classification/Data/Sessie_3/all_sessie_3.txt','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('D:/TGM3/Video_classification/Data/Sessie_3/shuffled_all_sessie_3.txt','w') as target:
    for _, line in data:
        target.write( line )
