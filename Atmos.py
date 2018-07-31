import tensorflow as tf
import pretty_midi as pm
import numpy as np
import os
from matplotlib import pyplot as plt
import keras as kr
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.activations import softmax
from keras.optimizers import RMSprop
from keras.layers import LSTM
from keras import utils as kr_utils
from keras.callbacks import ModelCheckpoint

input_data_path = "C:/Users/Josh/PycharmProjects/Bobs/Songs/"
output_data_path = ""
input_directory_listing = os.listdir(input_data_path)
instruments = np.empty([13, 4, 200])
output_instruments = np.empty([13,4])
for file in input_directory_listing:
    midi_data = pm.PrettyMIDI('Songs/'+file)
    for instrument in midi_data.instruments:
        parts = np.empty((4, 200))

        output_parts = np.empty((4,1))
        distance_from_last_note_in = np.empty((200))
        pitch_in = np.empty((200))
        note_lengths_in = np.empty((200))
        velocity_in = np.empty((200))
        velocity_out = np.empty((1))
        pitch_out = np.empty((1))
        distance_from_last_note_out = np.empty((1))
        velocity_out = np.empty((1))
        note_count = 0
        if "gui" in instrument.name or "GUIT" in instrument.name or "Guit" in instrument.name:
            temp = instrument.name
            for i in instrument.notes:
                note_count = note_count + 1
                if not note_count >= 200:
                    np.append(pitch_in, i.pitch)
                    np.append(velocity_in, i.velocity)
                    length = i.start - i.end
                    np.append(note_lengths_in, length)
                    #CREATE output file with single level of notes for each.
                    if note_count == 1:
                        temporary_end = 0
                        start_time = temporary_end - i.start
                        np.append(distance_from_last_note_in, start_time)
                        temporary_end = i.end
                    else:
                        start_time = temporary_end - i.start
                        print(start_time)
                        np.append(distance_from_last_note_in, start_time)
                        temporary_end = i.end
                elif note_count == 201:
                    np.append(pitch_out, i.pitch)
                    np.append(velocity_out, i.velocity)
                    np.append(distance_from_last_note_out, i.start)
                    length = i.start - i.end
                    np.append(note_lengths_in, length)
                    start_time_out = temporary_end - i.start
                    np.append(distance_from_last_note_out, start_time)
                np.append(output_parts,[velocity_out,distance_from_last_note_out,distance_from_last_note_out,pitch_out])
                np.append(parts, [velocity_in, distance_from_last_note_in, note_lengths_in, pitch_in])
    #print(parts[1])
    np.append(output_instruments, output_parts)
    np.append(instruments, parts)
y_train = np.array(len(instruments))
y_train = np.append(y_train, len(instruments))
#print(output_instruments.shape)
output_instruments = kr_utils.to_categorical(output_instruments)
instruments = instruments.reshape(instruments.shape[0], 4, 200)
#print(instruments.shape)
instruments = instruments.astype('float32')
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(1, 4, 200)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(200))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

