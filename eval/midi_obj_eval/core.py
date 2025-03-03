import numpy as np
import mido
import pretty_midi

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def extract_pretty_midi_features(midi_filepath):
    return pretty_midi.PrettyMIDI(midi_filepath)

def extract_pretty_midi_features_multiple(midi_filepaths):
    return [extract_pretty_midi_features(midi_filepath) for midi_filepath in midi_filepaths]

def get_num_notes(pretty_midi_features):
    get_onsets = pretty_midi_features.get_onsets()
    return len(get_onsets)

def get_note_density(pretty_midi_features, interval=5.0):
    # Total duration of the song
    total_duration = pretty_midi_features.get_end_time()
    
    # Define the start times for each interval
    interval_starts = np.arange(0, total_duration, interval)
    
    # List to store note density for each interval
    note_densities = []
    
    # Calculate note density in each interval
    for start in interval_starts:
        end = start + interval
        note_count = 0
        # Iterate over all instruments and notes
        for instrument in pretty_midi_features.instruments:
            for note in instrument.notes:
                # Count notes that start within the interval
                if start <= note.start < end:
                    note_count += 1
        # Append the density for this interval
        note_densities.append(note_count)
    
    # Calculate the average note density
    average_note_density = np.mean(note_densities)
    return average_note_density

# def get_note_density(pretty_midi_features, fs = 100, hop_sec = 5): # num notes per secend
#     piano_roll = pretty_midi_features.instruments[0].get_piano_roll(fs=fs)
#     onset_roll = (piano_roll[:, :-1] - piano_roll[:, 1:]) > 0
#     num_chunks = onset_roll.shape[-1] // (hop_sec * fs)
#     onset_roll_chunked = np.array_split(onset_roll, num_chunks, axis=1)
#     note_density_chunked = [
#         np.sum(onset_roll_segment) / onset_roll_segment.shape[1] * fs for onset_roll_segment in onset_roll_chunked
#     ]
#     return sum(note_density_chunked) / len(note_density_chunked)

def get_used_pitch(pretty_midi_features):
    """
    total_used_pitch (Pitch count): The number of different pitches within a sample.

    Returns:
    'used_pitch': pitch count, scalar for each sample.
    """
    piano_roll = pretty_midi_features.instruments[0].get_piano_roll(fs=100)
    sum_notes = np.sum(piano_roll, axis=1)
    used_pitch = np.sum(sum_notes > 0)
    return used_pitch

def get_used_pitch_multiple(list_of_pretty_midi_features):
    sum_notes = 0
    for pretty_midi_features in list_of_pretty_midi_features:
        piano_roll = pretty_midi_features.instruments[0].get_piano_roll(fs=100)
        sum_notes += np.sum(piano_roll, axis=1)
    used_pitch = np.sum(sum_notes > 0)
    return used_pitch

def get_pitch_class_histogram(pretty_midi_features):
    """
    total_pitch_class_histogram (Pitch class histogram):
    The pitch class histogram is an octave-independent representation of the pitch content with a dimensionality of 12 for a chromatic scale.
    In our case, it represents to the octave-independent chromatic quantization of the frequency continuum.

    Returns:
    'histogram': histrogram of 12 pitch, with weighted duration shape 12
    """
    piano_roll = pretty_midi_features.instruments[0].get_piano_roll(fs=100)
    histogram = np.zeros(12)
    for i in range(0, 128):
        pitch_class = i % 12
        histogram[pitch_class] += np.sum(piano_roll, axis=1)[i]
    histogram = histogram / sum(histogram)
    return histogram

def get_pitch_class_transition_matrix(pretty_midi_features, normalize=0):
    """
    pitch_class_transition_matrix (Pitch class transition matrix):
    The transition of pitch classes contains useful information for tasks such as key detection, chord recognition, or genre pattern recognition.
    The two-dimensional pitch class transition matrix is a histogram-like representation computed by counting the pitch transitions for each (ordered) pair of notes.

    Args:
    'normalize' : If set to 0, return transition without normalization.
                  If set to 1, normalizae by row.
                  If set to 2, normalize by entire matrix sum.
    Returns:
    'transition_matrix': shape of [12, 12], transition_matrix of 12 x 12.
    """
    transition_matrix = pretty_midi_features.get_pitch_class_transition_matrix()

    if normalize == 0:
        return transition_matrix

    elif normalize == 1:
        sums = np.sum(transition_matrix, axis=1)
        sums[sums == 0] = 1
        return transition_matrix / sums.reshape(-1, 1)

    elif normalize == 2:
        return transition_matrix / sum(sum(transition_matrix))

    else:
        print("invalid normalization mode, return unnormalized matrix")
        return transition_matrix

def get_avg_ioi(pretty_midi_features):
    """
    avg_IOI (Average inter-onset-interval):
    To calculate the inter-onset-interval in the symbolic music domain, we find the time between two consecutive notes.

    Returns:
    'avg_ioi': a scalar for each sample.
    """
    onset = pretty_midi_features.get_onsets()
    ioi = np.diff(onset)
    avg_ioi = np.mean(ioi)
    return avg_ioi