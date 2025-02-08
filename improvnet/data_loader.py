import yaml
import jsonlines
import glob
import random
import os
import sys
import pickle
import json
import argparse
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
import torch
from torch.nn import functional as F
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer

from corruptions import DataCorruption

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import flatten, unflatten, unflatten_corrupted


class Fusion_Dataset(Dataset):
    def __init__(self, configs, data_list, mode="train", shuffle = False):
        self.mode = mode
        self.data_list = data_list
        if shuffle:
            random.shuffle(self.data_list)

        # Artifact folder
        self.artifact_folder = configs['raw_data']['artifact_folder']
        # Load encoder tokenizer json file dictionary
        tokenizer_filepath = os.path.join(self.artifact_folder, "style_transfer", "vocab_corrupted.pkl")

        self.aria_tokenizer = AbsTokenizer()
        # Load the pickled tokenizer dictionary
        with open(tokenizer_filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)

        self.corruption_obj = DataCorruption()

        # Get the maximum sequence length
        self.encoder_max_sequence_length = configs['model']['encoder_max_sequence_length']
        self.decoder_max_sequence_length = configs['model']['decoder_max_sequence_length']

        # Print length of dataset
        print("Length of dataset: ", len(self.data_list))

    def __len__(self):
        return len(self.data_list)
    
    def augmentation(self, sequence, change_pitch_by=5, static_velocity=True):
        # Apply pitch augmentation to the sequence
        sequence_copy = deepcopy(sequence)
        for n, note in enumerate(sequence_copy):
            if note[0] == "piano":
                new_pitch = note[1] + change_pitch_by
                if static_velocity:
                    velocity = 90
                else:
                    velocity = note[2]
                new_note = "<U>" if new_pitch < 0 or new_pitch > 127 else ("piano", new_pitch, velocity)
                sequence_copy[n] = new_note               

        return sequence_copy
    
    
    def get_corrupted_sequence(self, sequence, meta_data):
        # Take the 3rd token as the start token until the 2nd last token
        sequence = sequence[2:-1]

        # Call the flatten function
        flattened_sequence = flatten(sequence, add_special_tokens=True)

        # Corrupt the flattened sequence
        context_before = random.randint(1, 5)
        context_after = random.randint(1, 5)
        output_dict = self.corruption_obj.apply_random_corruption(flattened_sequence, 
                                                                  context_before=context_before, 
                                                                  context_after=context_after, 
                                                                  meta_data=meta_data, 
                                                                  t_segment_ind=None,
                                                                  inference=False,
                                                                  corruption_type=None)
        corrupted_sequence = output_dict['corrupted_sequence']
        original_segment = output_dict['original_segment']

        corrupted_sequence, original_segment = unflatten_corrupted(corrupted_sequence), unflatten(original_segment)

        return corrupted_sequence, original_segment


    def __getitem__(self, idx):
        sequence_info = self.data_list[idx]
        genre = sequence_info[-1]
        tokenized_sequence = sequence_info[0]

        meta_tokens = [genre]

        # Apply augmentations
        pitch_aug_function = self.aria_tokenizer.export_pitch_aug(12)
        tokenized_sequence = pitch_aug_function(tokenized_sequence)

        input_tokens, original = self.get_corrupted_sequence(tokenized_sequence, meta_tokens)
        
        # Add the start and end tokens
        original = ["<S>"] + original + ["<E>"]

        # Tokenize the sequences
        input_tokens = [self.tokenizer[tuple(token)] if isinstance(token, list) else self.tokenizer[token] for token in input_tokens]
        original = [self.tokenizer[tuple(token)] if isinstance(token, list) else self.tokenizer[token] for token in original]

        # Pad the sequences
        if len(original) < self.decoder_max_sequence_length:
            original = F.pad(torch.tensor(original), (0, self.decoder_max_sequence_length - len(original))).to(torch.int64)
        else:
            original = torch.tensor(original[0:self.decoder_max_sequence_length]).to(torch.int64)
        if len(input_tokens) < self.encoder_max_sequence_length:
            input_tokens = F.pad(torch.tensor(input_tokens), (0, self.encoder_max_sequence_length - len(input_tokens))).to(torch.int64)
        else:
            input_tokens = torch.tensor(input_tokens[0:self.encoder_max_sequence_length]).to(torch.int64)

        # Attention mask based on non-padded tokens of the phrase
        attention_mask = torch.where(input_tokens != 0, 1, 0).type(torch.bool)

        train_data = {"input_ids": input_tokens, "labels": original, "attention_mask": attention_mask}

        return train_data


class Genre_Classifier_Dataset(Dataset):
    def __init__(self, configs, data_list, mode="train", shuffle = False):
        self.mode = mode
        self.data_list = data_list
        if shuffle:
            random.shuffle(self.data_list)

        # Artifact folder
        self.artifact_folder = configs['raw_data']['artifact_folder']
        # Load encoder tokenizer json file dictionary
        tokenizer_filepath = os.path.join(self.artifact_folder, "style_transfer", "vocab_corrupted.pkl")

        self.aria_tokenizer = AbsTokenizer()
        # Load the pickled tokenizer dictionary
        with open(tokenizer_filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)

        self.corruption_obj = DataCorruption()

        # Get the maximum sequence length
        self.encoder_max_sequence_length = configs['classifier_model']['encoder_max_sequence_length']

        # Print length of dataset
        print("Length of dataset: ", len(self.data_list))

    def __len__(self):
        return len(self.data_list)
    
    def augmentation(self, sequence, change_pitch_by=5, static_velocity=True):
        # Apply pitch augmentation to the sequence
        sequence_copy = deepcopy(sequence)
        for n, note in enumerate(sequence_copy):
            if note[0] == "piano":
                new_pitch = note[1] + change_pitch_by
                if static_velocity:
                    velocity = 90
                else:
                    velocity = note[2]
                new_note = "<U>" if new_pitch < 0 or new_pitch > 127 else ("piano", new_pitch, velocity)
                sequence_copy[n] = new_note               

        return sequence_copy
    
    
    def get_cropped_sequence(self, sequence, meta_data, t_segment_ind=None):
        # Take the 3rd token as the start token until the 2nd last token
        sequence = sequence[2:-1]

        # Call the flatten function
        flattened_sequence = flatten(sequence, add_special_tokens=True)

        # Corrupt the flattened sequence
        context_before = 2
        context_after = 2
        output_dict = self.corruption_obj.apply_random_corruption(flattened_sequence, 
                                                                  context_before=context_before, 
                                                                  context_after=context_after, 
                                                                  meta_data=meta_data, 
                                                                  t_segment_ind=t_segment_ind,
                                                                  inference=False,
                                                                  corruption_type=None,
                                                                  run_corruption=False)
        # Actually this is not a corrupted sequence, but the cropped original sequence 
        cropped_sequence = output_dict['corrupted_sequence']

        cropped_sequence = unflatten_corrupted(cropped_sequence)

        return cropped_sequence


    def __getitem__(self, idx):
        sequence_info = self.data_list[idx]
        genre = sequence_info[-1]
        tokenized_sequence = sequence_info[0]

        meta_tokens = [genre]

        # Apply augmentations
        pitch_aug_function = self.aria_tokenizer.export_pitch_aug(12)
        tokenized_sequence = pitch_aug_function(tokenized_sequence)

        input_tokens = self.get_cropped_sequence(tokenized_sequence, meta_tokens, t_segment_ind=None)

        # Tokenize the sequences
        input_tokens = [self.tokenizer[tuple(token)] if isinstance(token, list) else self.tokenizer[token] for token in input_tokens]
        
        # Define labels based on genre
        if genre == "classical":
            label = 0
        elif genre == "pop":
            label = 1
        elif genre == "jazz":
            label = 2

        # Pad the sequences
        if len(input_tokens) < self.encoder_max_sequence_length:
            input_tokens = F.pad(torch.tensor(input_tokens), (0, self.encoder_max_sequence_length - len(input_tokens))).to(torch.int64)
        else:
            input_tokens = torch.tensor(input_tokens[0:self.encoder_max_sequence_length]).to(torch.int64)

        # Attention mask based on non-padded tokens of the phrase
        attention_mask = torch.where(input_tokens != 0, 1, 0).type(torch.bool)

        train_data = {"input_ids": input_tokens, "labels": label, "attention_mask": attention_mask}

        return train_data



if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.normpath("configs/configs_style_transfer.yaml"),
                        help="Path to the config file")
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)
        
    artifact_folder = configs["raw_data"]["artifact_folder"]
    raw_data_folders = configs["raw_data"]["raw_data_folders"]
    
    run_fusion = False

    if run_fusion:
        with open(os.path.join(artifact_folder, "style_transfer", "pre_training_train.pkl"), "rb") as f:
            train_sequences = pickle.load(f)
        # Call the Fusion_Dataset class
        data_loader = Fusion_Dataset(configs, train_sequences, mode="val", shuffle=False)
        # Get the first item
        for n, data in enumerate(data_loader):
            print(data["input_ids"], '\n')
            print(data["labels"], '\n')
            print(data["input_ids"].shape)
            print(data["labels"].shape)
            print(data["attention_mask"].shape)
            if n == 1000:
                break
    else:
        with open(os.path.join(artifact_folder, "style_transfer", "fine_tuning_train.pkl"), "rb") as f:
            train_sequences = pickle.load(f)
        # Call the Genre_Classifier_Dataset class
        data_loader = Genre_Classifier_Dataset(configs, train_sequences, mode="train", shuffle=False)
        # Get the first item
        for n, data in enumerate(data_loader):
            print(data["input_ids"], '\n')
            print(data["labels"], '\n')
            print(data["input_ids"].shape)
            if n == 1000:
                break