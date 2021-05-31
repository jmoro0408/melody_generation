import tensorflow.keras as keras
import json
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
import numpy as np
import music21 as m21
import os

MELODY_SAVE_PATH = r"Melodies"


class MelodyGenerator:
    def __init__(self, model_path="model.h5"):

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):

        # create seed with start symbol
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to integers
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # limit ther seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # one hot encode the seed

            onehot_seed = keras.utils.to_categorical(
                seed, num_classes=len(self._mappings)
            )
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map in to mappign encoding
            output_symbol = [
                key for key, value in self._mappings.items() if value == output_int
            ][0]

            # Check if we are at the end of a melody
            if output_symbol == "/":
                break

            # update the melody
            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilities, temperature):
        predictions = np.log(probabilities) / temperature
        probablities = np.exp(predictions) / np.sum(
            np.exp(predictions)
        )  # applying softmax
        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(
        self,
        melody,
        step_duration=0.25,
        format="midi",
        file_name="mel.mid",
        save_path=MELODY_SAVE_PATH,
    ):
        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = (
                        step_duration * step_counter
                    )  # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(
                            int(start_symbol), quarterLength=quarter_length_duration
                        )

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, fp=os.path.join(save_path, file_name))


if __name__ == "__main__":
    mg = MelodyGenerator()
    seeds = [
        "55 55 60 _ _ _ 67 _ _ 65 64",
        "60 _ _ _ _ _ _ _ 60 _ _ _ 64 _ _ _ _ _ _ _ 65 _ _ _ 67 _ _ _ _ _ _ _ 67 _ _ _",
        "60 _ _ 55 55 _ 55 _ 57 _ 59 _",
        "55 _ 60 _ _ _ 60 _ 64 _ 62 _ 60 _",
    ]
    for i, seed in enumerate(seeds):
        melody = mg.generate_melody(
            seed, num_steps=500, max_sequence_length=SEQUENCE_LENGTH, temperature=0.85
        )
        melody_file_name = f"mel_{i}.mid"
        mg.save_melody(melody,file_name = melody_file_name )
