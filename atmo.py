import tensorflow as tf
import pretty_midi as pm
import numpy as np
import os
import keras as kr
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.activations import softmax
from keras.optimizers import RMSprop
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

network_input = []
network_output = []
sequenceLength = 4
count = 0
input_data_path = "C:/Users/Josh/PycharmProjects/Bobs/Songs/"
output_data_path = ""
input_directory_listing = os.listdir(input_data_path)
songs = ([])
instruments = np.empty([13, 4, 200])
for file in input_directory_listing:
    midi_data = pm.PrettyMIDI('Songs/'+file)

    for instrument in midi_data.instruments:
        parts = np.empty((4, 200))
        start_in = np.empty((200))
        pitch_in = np.empty((200))
        end_in = np.empty((200))
        velocity_in = np.empty((200))
        note_count = 0
        count = count + 1
        if "gui" in instrument.name or "GUIT" in instrument.name or "Guit" in instrument.name:
            temp = instrument.name
            for i in instrument.notes:
                note_count = note_count + 1
                if not note_count >= 200:
                    #print(i.pitch)
                    np.append(pitch_in, i.pitch)
                    np.append(velocity_in, i.velocity)
                    np.append(start_in, int(i.start*1000))
                    np.append(end_in, int(i.end*1000))
                np.append(parts, [velocity_in, start_in, end_in, pitch_in])
        np.append(instruments, parts)
network_output = ([4, 200])
print(len(instruments))
np.reshape(instruments, (len(instruments), 4, 200), 3)
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(4, 200)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(1))
model.add(Activation("relu"))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(instruments, network_output, epochs=200, batch_size=13, callbacks='callbacks_list')

model = Sequential()
model.add(LSTM(
    512,
    input_shape=(network_input.shape[1], network_input.shape[2]),
    return_sequences=True
))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

start = np.random.randint(0, len(network_input)-1)
pattern = network_input[start]
prediction_output = []
# generate 500 notes
for note_index in range(500):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(128)
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    prediction_output.append(index)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
