import pathlib
import os
import music21 as m21

KERN_DATASET_PATH = r"/Users/James/Documents/Python/Machine Learning Projects/Generating_melodies/PreProcessing/deutschl/test"
ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]


def current_working_dir():
    return pathlib.Path().absolute()


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

    for song in songs:
        # filter out songs that have non-acceptable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
        # transpose songs to Cmaj/Amin
        song = transpose(song)
        # encode songs with music time series representation

        # save songs to text file


if __name__ == "__main__":

    songs = load_songs_in_kern(KERN_DATASET_PATH)
    print(f"Loaded {len(songs)} songs")
    song = songs[1]

    print(
        f"Has aceptable duration? {has_acceptable_durations(song, ACCEPTABLE_DURATIONS)}"
    )
    transposed_song = transpose(song)

    song.show()
    transposed_song.show()
