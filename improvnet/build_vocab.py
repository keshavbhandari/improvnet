import yaml
import os
import argparse
import pickle
import glob
import json
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer
from tqdm import tqdm
import random
import pickle

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.normpath("configs/configs_style_transfer.yaml"),
                    help="Path to the config file")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    configs = yaml.safe_load(f)

build_dataset = configs["raw_data"]["build_dataset"]
artifact_folder = configs["raw_data"]["artifact_folder"]
raw_data_folders = configs["raw_data"]["raw_data_folders"]

# Build the vocabulary
vocab = {}
# MIDI velocity range from 0 to 127
velocity = [0, 15, 30, 45, 60, 75, 90, 105, 120, 127]
# MIDI pitch range from 0 to 127
midi_pitch = list(range(0, 128))
# Onsets are quantized in 10 milliseconds up to 5 seconds
onset = list(range(0, 5001, 10))
duration = list(range(0, 5001, 10))

# Add the tokens to the vocabulary
for v in velocity:
    for p in midi_pitch:
        vocab[("piano", p, v)] = len(vocab) + 1
for o in onset:
    vocab[("onset", o)] = len(vocab) + 1
for d in duration:
    vocab[("dur", d)] = len(vocab) + 1

# Genre tokens
vocab["classical"] = len(vocab) + 1
vocab["pop"] = len(vocab) + 1
vocab["jazz"] = len(vocab) + 1

# Special tokens
vocab["O"] = len(vocab) + 1
vocab["D"] = len(vocab) + 1
vocab["PVM"] = len(vocab) + 1
vocab["mask"] = len(vocab) + 1
vocab["pitch_velocity_mask"] = len(vocab) + 1
vocab["onset_duration_mask"] = len(vocab) + 1
vocab["whole_mask"] = len(vocab) + 1
vocab["pitch_permutation"] = len(vocab) + 1
vocab["pitch_velocity_permutation"] = len(vocab) + 1
vocab["fragmentation"] = len(vocab) + 1
vocab["incorrect_transposition"] = len(vocab) + 1
vocab["skyline"] = len(vocab) + 1
vocab["note_modification"] = len(vocab) + 1

vocab[('prefix', 'instrument', 'piano')] = len(vocab) + 1
vocab["<T>"] = len(vocab) + 1
vocab["<D>"] = len(vocab) + 1
vocab["<U>"] = len(vocab) + 1
vocab["<S>"] = len(vocab) + 1
vocab["<E>"] = len(vocab) + 1
vocab["SEP"] = len(vocab) + 1

# Print the vocabulary length
print(f"Vocabulary length: {len(vocab)}")

# Save the vocabulary
# Directory path
fusion_folder = os.path.join(artifact_folder, "style_transfer")
# Make directory if it doesn't exist
os.makedirs(fusion_folder, exist_ok=True)
with open(os.path.join(fusion_folder, "vocab_corrupted.pkl"), 'wb') as f:
    pickle.dump(vocab, f)

if build_dataset:
    pre_training_dir = raw_data_folders['pre_training']
    pre_training_file_list = []
    folder_paths_list = pre_training_dir['folder_paths']
    for folder_path in folder_paths_list:
        pre_training_file_list += glob.glob(folder_path + '/**/*.mid', recursive=True)
        pre_training_file_list += glob.glob(folder_path + '/**/*.midi', recursive=True)
    print(f"Number of Pre-training MIDI files: {len(pre_training_file_list)}")
    pre_training_file_dict = {file: "classical" for file in pre_training_file_list}

    fine_tuning_dir = raw_data_folders['fine_tuning']
    fine_tuning_file_dict = {}
    classical_folder_paths_list = fine_tuning_dir['classical_folder_paths']
    for folder_path in classical_folder_paths_list:
        file_list = glob.glob(folder_path + '/**/*.mid', recursive=True) + glob.glob(folder_path + '/**/*.midi', recursive=True)
        fine_tuning_file_dict.update({file: "classical" for file in file_list})
        print(f"Number of Classical Fine-tuning MIDI files: {len(file_list)}")
    jazz_folder_paths_list = fine_tuning_dir['jazz_folder_paths']
    for folder_path in jazz_folder_paths_list:
        file_list = glob.glob(folder_path + '/**/*.mid', recursive=True) + glob.glob(folder_path + '/**/*.midi', recursive=True)
        fine_tuning_file_dict.update({file: "jazz" for file in file_list})
        print(f"Number of Jazz Fine-tuning MIDI files: {len(file_list)}")
    pop_folder_paths_list = fine_tuning_dir['pop_folder_paths']
    for folder_path in pop_folder_paths_list:
        file_list = glob.glob(folder_path + '/**/*.mid', recursive=True) + glob.glob(folder_path + '/**/*.midi', recursive=True)
        file_list = [file for file in file_list if "versions" not in file]
        fine_tuning_file_dict.update({file: "pop" for file in file_list})
        print(f"Number of Pop Fine-tuning MIDI files: {len(file_list)}")
    print(f"Number of Fine-tuning MIDI files: {len(fine_tuning_file_dict)}")

    # Create a function that shuffles the file_list and creates a train validation split
    # def shuffle_and_split(file_dict, split=0.9):
    #     file_list = list(file_dict.keys())
    #     random.shuffle(file_list)
    #     train_data = {k: v for k, v in file_dict.items() if k in file_list[:int(split * len(file_list))]}
    #     val_data = {k: v for k, v in file_dict.items() if k in file_list[int(split * len(file_list)):]}

    #     return train_data, val_data
    def shuffle_and_split(file_dict, val_size=100):
        file_list = list(file_dict.keys())
        # Shuffle the file list with seed
        random.seed(42)
        random.shuffle(file_list)
        
        # Ensure we do not exceed the number of files in the dictionary
        val_size = min(val_size, len(file_list))
        
        train_files = file_list[val_size:]
        val_files = file_list[:val_size]
        
        train_data = {k: file_dict[k] for k in train_files}
        val_data = {k: file_dict[k] for k in val_files}

        return train_data, val_data
    
    def filter_midi_files(file_dict):
        filtered_file_dict = {}
        for file in tqdm(file_dict.keys()):
            try:
                mid = MidiDict.from_midi(file)
                mid.note_msgs = [msg for msg in mid.note_msgs if msg['channel'] != 9]
                if len(mid.note_msgs) == 0:
                    continue
            except:
                continue
            unique_channels = list(set([msg['channel'] for msg in mid.note_msgs]))
            all_instrument_numbers = [msg['data'] for msg in mid.instrument_msgs]
            # If the MIDI file has only one channel and any instrument number is between 0 and 8
            if any([0 <= instrument_number < 8 for instrument_number in all_instrument_numbers]):
                filtered_file_dict[file] = file_dict[file]
        return filtered_file_dict
    
    # Filter out the MIDI files that have multiple channels
    pre_training_file_dict = filter_midi_files(pre_training_file_dict)
    print(f"Number of Pre-training MIDI files after filtering: {len(pre_training_file_dict)}")
    # Split the pre-training data into train and validation sets
    pre_training_train_data, pre_training_val_data = shuffle_and_split(pre_training_file_dict)
    print(f"Number of Pre-training training files: {len(pre_training_train_data)}, Number of Pre-training validation files: {len(pre_training_val_data)}")

    # Split the classical fine-tuning data into train and validation sets
    fine_tuning_file_dict = filter_midi_files(fine_tuning_file_dict)
    fine_tuning_train_data, fine_tuning_val_data = shuffle_and_split(fine_tuning_file_dict)
    print(f"Number of Fine-tuning training files: {len(fine_tuning_train_data)}, Number of Fine-tuning validation files: {len(fine_tuning_val_data)}")

    # Save the training and validation file list as json
    with open (os.path.join(fusion_folder, "pre_training_train_file_list.json"), "w") as f:
        json.dump(pre_training_train_data, f)
    with open (os.path.join(fusion_folder, "pre_training_valid_file_list.json"), "w") as f:
        json.dump(pre_training_val_data, f)

    with open (os.path.join(fusion_folder, "fine_tuning_train_file_list.json"), "w") as f:
        json.dump(fine_tuning_train_data, f)
    with open (os.path.join(fusion_folder, "fine_tuning_valid_file_list.json"), "w") as f:
        json.dump(fine_tuning_val_data, f)

    # Read and store the MIDI files as pkl files
    def store_files_as_json(file_dict, pickle_folder, file_name="train"):
        aria_tokenizer = AbsTokenizer()
        list_of_tokens = []
        
        for idx, file_path in tqdm(enumerate(file_dict.keys()), total=len(file_dict), desc="Processing files"):
            genre = file_dict[file_path]
            mid = MidiDict.from_midi(file_path)
            mid.note_msgs = [msg for msg in mid.note_msgs if msg['channel'] != 9]
            for i, msg in enumerate(mid.instrument_msgs):
                if msg['data'] != 0:
                    mid.instrument_msgs[i]['data'] = 0
            tokenized_sequence = aria_tokenizer.tokenize(mid)
            final_seq = [tokenized_sequence, genre]
            list_of_tokens.append(final_seq)
        
        # Save the list_of_tokens as a pickle file
        pickle_file_path = os.path.join(pickle_folder, f"{file_name}.pkl")
        with open(pickle_file_path, "wb") as f:
            pickle.dump(list_of_tokens, f)

    # Store the training and validation files as pkl after reading them
    store_files_as_json(pre_training_train_data, fusion_folder, file_name="pre_training_train")
    store_files_as_json(pre_training_val_data, fusion_folder, file_name="pre_training_valid")

    store_files_as_json(fine_tuning_train_data, fusion_folder, file_name="fine_tuning_train")
    store_files_as_json(fine_tuning_val_data, fusion_folder, file_name="fine_tuning_valid")