import os
import music21 as m21

KERN_DATASET_PATH = r"/Users/James/Documents/Python/MachineLearningProjects/Generating_melodies/PreProcessing/deutschl/test"
SAVE_DIR = r"/Users/James/Documents/Python/MachineLearningProjects/Generating_melodies/PreProcessing/Dataset"
ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]


def encode_song(song, time_step=0.25):
    # pitch = 60, duration = 1 -> [60, "_". "_", "_"] "_" = 1/16th note
    encoded_song = []
    for event in song.flat.notesAndRests:
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi  # midi note version of note
        elif isinstance(event, m21.note.Rest):
            symbol = "r"  # rest

        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to string
    encoded_song = " ".join([str(i) for i in encoded_song])
    return encoded_song


def load_songs_in_kern(dataset_path):
    songs = []
    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".krn"):
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
        return True


def transpose(song):
    # Get key from score
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    # estimate key using music21 if key is not included in score
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    # Get interval for transposition. E.G Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    # Transpose song by calculated interval
    transposed_song = song.transpose(interval)

    return transposed_song


def preprocess(dataset_path):
    pass
    # Load dataset
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs")

    for i, song in enumerate(songs):
        # filter out songs that have non-acceptable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
        # transpose songs to Cmaj/Amin
        song = transpose(song)
        # encode songs with music time series representation
        encoded_song = encode_song(song)
        # save songs to text file
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


if __name__ == "__main__":
    preprocess(KERN_DATASET_PATH)
    print(f"Finished preprocessing")
