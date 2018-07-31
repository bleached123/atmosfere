import tensorflow as tf
import pretty_midi as pm
import numpy as np
import os
import keras as kr
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers import LSTM

songs = []
for file in os.listdir('C:/Users/Josh/PycharmProjects/Bobs/Songs/'):
    midi_data = pm.PrettyMIDI('Songs/'+file)
    notes = []
    instruments = []
    for instrument in midi_data.instruments:
        starts = []
        ends = []
        pitches = []
        velocities = []
        if not instrument.is_drum: #STILL TO DO create an array for the drumming notes.
            for note in instrument.notes:
                #normalise to be an integer
                tempstart = int(100 * note.start)
                tempend = int(100 * note.end)
                #place these integers into arrays
                starts.append(tempstart)
                ends.append(tempend)
                pitches.append(note.pitch)
                velocities.append(note.velocity)
        #add these arrays to an array for that instrument
        notes = np.array([starts, ends, pitches, velocities])

        n_instruments = len(notes)

#add these arrays to an array for the whole song.
        instruments.extend([notes])
    #print(len(instruments))
    songs.append(instruments)
n_patterns = len(songs)
print(notes)
#print(songs[1])
print(instruments[0][1])
print(pitches)
network_input = np.reshape(pitches, (instruments[0:1],  2500, 1))

print('Building model...')
model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
#increase number of nodes in this layer to 512
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(256))
#add the number of different types of notes there are
model.add(Dense(127))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#load the weights to each node.
model.load_weights('weights.hdf5')

