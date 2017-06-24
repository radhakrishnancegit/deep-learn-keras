"""
Simple MLP nueral net to count the number of vowels from set of sequences.
Nueral net will classify a sentence into a class #num_vowels in the sentence.

abcd => 1
prwkld => 0
aaaiiiooo => 9

Results:
epoch_size: 100
batch_size: 100

python count_vowels_mlp.py --optimzer=adam
Optimzer function: adam (Why adam works better than others?)
Time taken to train the model with single core GPU: 269.73893404 seconds
Test Accuracy : 0.9457

python count_vowels_mlp.py --optimzer=rmsprop
Optimzer function: rmsprop
Time taken to train the model with single core GPU: 256.734019041 seconds
Test Accuracy : 0.8693

python count_vowels_mlp.py --optimzer=sgd
Optimzer function: sgd
Time taken 241.566617012
Test Accuracy : 0.7149

"""
import random
import string
import sys
import time
import gflags
import numpy as np
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten

FLAGS = gflags.FLAGS

gflags.DEFINE_string('optimizer', '', 'optimizer function to optimize the loss function')

def generateData(samples, max_string_length = 100):
    data = []
    labels = []
    for i in range(samples):
        tmp_string = ''.join(random.choice(string.lowercase) for x in range(random.randint(1, max_string_length)))
        num_vowels = sum(v for v in map(tmp_string.count, "aeiou"))
        tmp_array = [(ord(x) - ord('a') + 1) for x in tmp_string]
        tmp_data_value = []
        for x in tmp_array:
            tmp_val = [0]*27
            tmp_val[x] = 1
            tmp_data_value.append(tmp_val)
        data.append(tmp_data_value)
        labels.append(num_vowels)
    return data, labels
    
def TrainAndTestModel():
    MAX_LENGTH_OF_STRING = 50
    NUM_TRAIN_SAMPLES = 40000
    NUM_TEST_SAMPLES = 10000
    train_data, train_labels = generateData(NUM_TRAIN_SAMPLES, max_string_length=MAX_LENGTH_OF_STRING)
    train_input = sequence.pad_sequences(train_data, maxlen=MAX_LENGTH_OF_STRING)
    train_one_hot_labels = to_categorical(train_labels, num_classes=MAX_LENGTH_OF_STRING + 1)
    
    
    test_data, test_labels = generateData(NUM_TEST_SAMPLES, max_string_length=MAX_LENGTH_OF_STRING)
    test_input = sequence.pad_sequences(test_data, maxlen=MAX_LENGTH_OF_STRING)
    test_one_hot_labels = to_categorical(test_labels, num_classes=MAX_LENGTH_OF_STRING + 1)
    
    model = Sequential()
    model.add(Dense(MAX_LENGTH_OF_STRING, input_shape=[MAX_LENGTH_OF_STRING, 27], activation='relu'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(MAX_LENGTH_OF_STRING + 1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer=FLAGS.optimzer,
                  metrics=['accuracy'])
    
    print(model.summary())
    print(train_input.shape)
    print(train_one_hot_labels.shape)
    
    train_start_time = time.time()
    model.fit(train_input, train_one_hot_labels, epochs=100, batch_size=100)
    print("Time taken " + str(time.time() - train_start_time))
    print("Optimzer function: " + FLAGS.optimzer)
    print("Test Accuracy : " + str(model.evaluate(test_input, test_one_hot_labels)[1]))


def main(argv):
    try:
        argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError, e:
        print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)
    TrainAndTestModel()
if __name__ == '__main__':
    main(sys.argv)