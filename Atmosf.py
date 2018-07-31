import tensorflow as tf
import pretty_midi as pm
import numpy as np
import os
import h5py as hp
from matplotlib import pyplot as plt
import keras as kr
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.activations import softmax
from keras.optimizers import RMSprop
from keras.layers import LSTM
from keras.utils import np_utils as np_utils
from keras.callbacks import ModelCheckpoint, CallbackList
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def create_midi(output_array):

    # Create a PrettyMIDI object
    created_song = pm.PrettyMIDI()
    # Create an Instrument instance for a cello instrument
    guitar = pm.Instrument(program=25)
    # Iterate over note names, which will be converted to note number later
    for note in output_array:
        # Create a Note instance, starting at 0s and ending at .5s

        note = pm.Note(
            velocity=int(note[0]), pitch=int(note[1]), start=note[2], end=note[3])
        # Add it to our cello instrument
        guitar.notes.append(note)
    # Add the cello instrument to the PrettyMIDI object
    created_song.instruments.append(guitar)
    # Write out the MIDI data
    created_song.write('Generated_Song.mid')

input_data_path = "C:/Users/Josh/PycharmProjects/Bobs/Songs/"
input_directory_listing = os.listdir(input_data_path)
temp_notes = np.empty([])
temp_notes_output = np.empty([])
num = 0
for file in input_directory_listing:
    midi_data = pm.PrettyMIDI('Songs/'+file)
    for instrument in midi_data.instruments:
        note_start = np.empty((2000))
        pitch_in = np.empty((2000))
        note_lengths_in = np.empty((2000))
        velocity_in = np.empty((2000))
        note_start_out = np.empty((2000))
        pitch_out = np.empty((2000))
        note_lengths_out = np.empty((2000))
        velocity_out = np.empty((2000))
        note_count = 0
        num = num + 1
        if "gui" in instrument.name or "GUIT" in instrument.name or "Guit" in instrument.name:
            temp = instrument.name
            for i in instrument.notes:
                note_count = note_count + 1
                if note_count == 1:
                    np.append(velocity_out, i.velocity)
                    np.append(pitch_out, i.pitch)
                    np.append(note_start_out, i.start)
                    np.append(note_lengths_out, i.end)
                if not note_count <= 2000:
                    np.append(pitch_in, i.pitch)
                    np.append(velocity_in, i.velocity)
                    np.append(note_start, i.start)
                    np.append(note_lengths_in, i.end)
                np.append(temp_notes, [velocity_in, pitch_in, note_start, note_lengths_in])
    np.append(temp_notes_output, [velocity_out, pitch_out, note_start_out, note_lengths_out])
notes = np.empty([480, 4, 2000])
notes_out = np.zeros([480, 4])
np.append(notes, temp_notes)
np.append(notes_out, temp_notes_output)
notes_out = np_utils.to_categorical(notes_out)
notes = notes.reshape(notes.shape[0], 4, 2000)
np.set_printoptions(suppress=True)
model = Sequential()
model.add(LSTM(512, input_shape=(4, 2000), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(1000, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)

callbacks_list = [checkpoint]
model.fit(notes, notes_out, epochs=1, batch_size=1000)

# Generate notes from the neural network based on a sequence of notes
prediction_input = np.empty([1, 4, 2000])
prediction_output = np.empty([4, 2000])
np.append(prediction_input[0], [7, 7, 0.0, 0.7])
print(prediction_input.shape)
print(prediction_input)
np.set_printoptions(suppress=True)

for note in range(1000):
    #prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    np.append(prediction_output, index)
    print(index)

create_midi(prediction_output)
#create_midi(prediction_output)
#model.save_weights("model_weight.h5")



