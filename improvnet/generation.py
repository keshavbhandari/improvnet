import yaml
import json
import pickle
import os
import random
import numpy as np
import copy
import sys
import argparse
from tqdm import tqdm
import pretty_midi
import torch
from torch.nn import functional as F
from transformers import EncoderDecoderModel
from torch.cuda import is_available as cuda_available
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer

from corruptions import DataCorruption

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import flatten, unflatten_corrupted, parse_generation, unflatten_for_aria, Segment_Novelty


def refine_sequence(corrupted_sequence, tokenizer, decode_tokenizer, model, encoder_max_sequence_length, decoder_max_sequence_length, temperature=1.0):
    # Tokenize the sequence
    # input_tokens = [tokenizer[tuple(token)] if isinstance(token, list) else tokenizer[token] for token in corrupted_sequence]
    input_tokens = [tokenizer[token] for token in corrupted_sequence if token in tokenizer.keys()]
    # Pad the sequences
    if len(input_tokens) < encoder_max_sequence_length:
        input_tokens = F.pad(torch.tensor(input_tokens), (0, encoder_max_sequence_length - len(input_tokens))).to(torch.int64)
    else:
        input_tokens = torch.tensor(input_tokens[0:encoder_max_sequence_length]).to(torch.int64)

    # Attention mask based on non-padded tokens of the phrase
    attention_mask = torch.where(input_tokens != 0, 1, 0).type(torch.bool)

    # Generate the output sequence
    output_tokens = model.generate(input_ids=input_tokens.unsqueeze(0).to("cuda" if cuda_available() else "cpu"),
                                    attention_mask=attention_mask.unsqueeze(0).to("cuda" if cuda_available() else "cpu"),
                                    max_length=decoder_max_sequence_length,
                                    num_beams=1,
                                    early_stopping=False,
                                    do_sample=True,
                                    temperature=temperature,
                                    top_k=50,
                                    top_p=1.0,
                                    pad_token_id=0,
                                    eos_token_id=tokenizer["<E>"],
                                    bos_token_id=tokenizer["<S>"])

    # Decode the output tokens
    output_sequence = [decode_tokenizer[token.item()] for token in output_tokens[0]]

    generated_sequences = parse_generation(output_sequence, add_special_tokens = True)

    # Remove special tokens
    generated_sequences = [token for token in generated_sequences if token not in ["<S>", "<E>", "<SEP>"]]

    return generated_sequences


def generate_one_pass(tokenized_sequence, fusion_model, configs, 
                      t_segment_start, convert_to, context_before, 
                      context_after, corruption_type, corruption_rate, 
                      tokenizer, decode_tokenizer, quiet, temperature=1.0, end_original=True, t_segment_stop=-1):

    # Get the encoder and decoder max sequence length
    encoder_max_sequence_length = configs['model']['encoder_max_sequence_length']
    decoder_max_sequence_length = configs['model']['decoder_max_sequence_length']
    
    corruption_obj = DataCorruption()
    separated_sequence = corruption_obj.seperateitems(tokenized_sequence)
    # Get the indices of the novelty tokens
    novelty_segments = [n for n, i in enumerate(separated_sequence) if '<N>' in i]
    all_segment_indices, _, _, _ = corruption_obj.get_segment_to_corrupt(separated_sequence, t_segment_ind=t_segment_start, exclude_idx=novelty_segments)

    if end_original:
        n_iterations = len(all_segment_indices) - 1
    else:
        n_iterations = len(all_segment_indices)

    t_segment_ind = t_segment_start
    if t_segment_stop <= t_segment_start:
        t_segment_stop = n_iterations
        # print("t_segment_stop is less than or equal to t_segment_start. Setting t_segment_stop to the end of the sequence.")

    # Initialize tqdm
    progress_bar = tqdm(total=n_iterations, disable=quiet)
    jump_every = 1

    while t_segment_ind < n_iterations:

        if t_segment_ind >= t_segment_stop:
            break

        if random.random() < corruption_rate:
            if corruption_type == "random":
                corruption_type_tmp = random.choice(list(corruption_obj.corruption_functions.keys()))
                output_dict = corruption_obj.apply_random_corruption(tokenized_sequence, context_before=context_before, context_after=context_after, meta_data=[convert_to], t_segment_ind=t_segment_ind, inference=False, corruption_type=corruption_type_tmp, run_corruption=True, exclude_idx=novelty_segments)
            else:
                output_dict = corruption_obj.apply_random_corruption(tokenized_sequence, context_before=context_before, context_after=context_after, meta_data=[convert_to], t_segment_ind=t_segment_ind, inference=False, corruption_type=corruption_type, run_corruption=True, exclude_idx=novelty_segments)
            index = output_dict['index']
            corrupted_sequence = output_dict['corrupted_sequence']
            corrupted_sequence = unflatten_corrupted(corrupted_sequence)
            refined_segment = refine_sequence(corrupted_sequence, tokenizer, decode_tokenizer, fusion_model, encoder_max_sequence_length, decoder_max_sequence_length, temperature=temperature)
            flattened_refined_segment = flatten(refined_segment, add_special_tokens=True)
            separated_sequence[index] = flattened_refined_segment
            tokenized_sequence = corruption_obj.concatenate_list(separated_sequence)
            if not quiet:
                print("Corrupted t_segment_ind:", t_segment_ind, "Corruption type:", output_dict['corruption_type'])
        
        t_segment_ind += jump_every
        # Update the progress bar
        progress_bar.update(jump_every)

    progress_bar.close()

    return tokenized_sequence


def get_midi_notes_from_tick(midi_dict, midi_data, peak_times, quiet):
    novel_note_numbers = []
    novel_notes = []
    for n, note in enumerate(midi_dict.note_msgs):
        if len(peak_times) > 0:
            tick = note['tick']
            # tick_to_time
            time_in_sec = pretty_midi.PrettyMIDI.tick_to_time(midi_data, tick)

            if time_in_sec > peak_times[0]:
                if not quiet:
                    print(f"Note {n} is at {time_in_sec} seconds")
                novel_note_numbers.append(n)
                novel_notes.append(note)
                peak_times = np.delete(peak_times, 0)
        else:
            return novel_note_numbers, novel_notes


def add_novelty_segment_token(tokenized_sequence, novel_note_numbers):
    note_number = 0
    insertion_points = []
    for note_idx, note in enumerate(tokenized_sequence):
        if type(note) == list:
            if note_number in novel_note_numbers:
                insertion_points.append(note_idx)
            note_number += 1
    
    new_tokenized_sequence = []
    for idx, token in enumerate(tokenized_sequence):
        if idx in insertion_points:
            new_tokenized_sequence.append("<N>")
        new_tokenized_sequence.append(token)
    
    return new_tokenized_sequence


def write_file(midi_file_path, output_folder, tokenized_sequence, aria_tokenizer):
    sequence = copy.deepcopy(tokenized_sequence)
    # Unflatten the tokenized sequence
    sequence = unflatten_for_aria(sequence)

    # Filter out <N> tokens
    sequence = [token for token in sequence if token != "<N>"]

    # Write the generated sequences to a MIDI file
    generated_sequence = [('prefix', 'instrument', 'piano'), "<S>"] + sequence + ["<E>"]
    mid_dict = aria_tokenizer.detokenize(generated_sequence)
    generated_mid = mid_dict.to_midi()
    filename = os.path.basename(midi_file_path)
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    generated_mid.save(os.path.join(output_folder, "generated_" + filename))


def generate(midi_file_path, audio_file_path, fusion_model, configs, novel_peaks_pct,
             t_segment_start, convert_to, context_before, context_after, 
             corruption_passes, tokenizer, decode_tokenizer, output_folder, 
             save_original=False, quiet=False, write_intermediate_passes=False, temperature=1.0, end_original=True, t_segment_stop=-1):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the MIDI file
    if not quiet:
        print("File path:", midi_file_path)
    filename = os.path.basename(midi_file_path)
    mid = MidiDict.from_midi(midi_file_path)
    aria_tokenizer = AbsTokenizer()
    tokenized_sequence = aria_tokenizer.tokenize(mid)

    if save_original:
        # Save the original MIDI file
        mid_dict = aria_tokenizer.detokenize(tokenized_sequence)
        original_mid = mid_dict.to_midi()
        original_mid.save(os.path.join(output_folder, "original_" + filename))

    # Get the instrument token
    instrument_token = tokenized_sequence[0]
    tokenized_sequence = tokenized_sequence[2:-1]

    # Novelty based segmentation
    if audio_file_path is None or audio_file_path == "":
        novel_note_numbers = []
    else:
        ssm_config_file = "configs/config_ssm.yaml"
        n_t_segments = len([t for t in tokenized_sequence if t == "<T>"])
        novel_peaks_pct = configs['generation']['novel_peaks_pct']
        n_novel_peaks = np.ceil(novel_peaks_pct * n_t_segments).astype(int)
        if not quiet:
            print("Number of novel peaks:", n_novel_peaks)
        segment_novelty = Segment_Novelty(ssm_config_file, audio_file_path)
        peak_times = segment_novelty.get_peak_timestamps(audio_file_path, n_novel_peaks)
        pretty_midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        novel_note_numbers, novel_notes = get_midi_notes_from_tick(mid, pretty_midi_data, peak_times, quiet)

    # Flatten the tokenized sequence
    tokenized_sequence = flatten(tokenized_sequence, add_special_tokens=True)

    # Add novelty segment token to the tokenized sequence
    tokenized_sequence = add_novelty_segment_token(tokenized_sequence, novel_note_numbers)

    # Generate by iterating with multiple passes
    passes = len(corruption_passes.keys())
    data_corruption_obj = DataCorruption()
    # corruptions = list(data_corruption_obj.corruption_functions.keys())
    for i in range(passes):
        if not quiet:
            print("Pass:", i + 1)
        corruption_type = corruption_passes['pass_' + str(i + 1)]['corruption_type']
        # if corruption_type == "random":
        #     corruption_type = random.choice(corruptions)
        corruption_rate = corruption_passes['pass_' + str(i + 1)]['corruption_rate']
        # Generate the sequence
        tokenized_sequence = generate_one_pass(tokenized_sequence, fusion_model, configs, 
                                               t_segment_start, convert_to, context_before, 
                                               context_after, corruption_type, corruption_rate, 
                                               tokenizer, decode_tokenizer, quiet, temperature=temperature, 
                                               end_original=end_original, t_segment_stop=t_segment_stop)

        if write_intermediate_passes:
            pass_number = f"pass_{passes}"
            if pass_number in output_folder:
                new_output_folder = output_folder.replace(pass_number, f"pass_{i + 1}")
            else:
                new_output_folder = os.path.join(output_folder, f"pass_{i + 1}")
            write_file(midi_file_path, new_output_folder, tokenized_sequence, aria_tokenizer)

    # Write the generated sequence to a MIDI file
    write_file(midi_file_path, output_folder, tokenized_sequence, aria_tokenizer)



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
    output_folder = os.path.join("output")
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get tokenizer
    tokenizer_filepath = os.path.join(artifact_folder, "style_transfer", "vocab_corrupted.pkl")
    # Load the tokenizer dictionary
    with open(tokenizer_filepath, "rb") as f:
        tokenizer = pickle.load(f)

    # Reverse the tokenizer
    decode_tokenizer = {v: k for k, v in tokenizer.items()}

    # Load the fusion model
    fusion_model = EncoderDecoderModel.from_pretrained(os.path.join(artifact_folder, "style_transfer", "fine_tuned_model"))
    fusion_model.eval()
    fusion_model.to("cuda" if cuda_available() else "cpu")
    print("Fusion model loaded")

    # Open JSON file
    with open(os.path.join(artifact_folder, "style_transfer", "fine_tuning_valid_file_list.json"), "r") as f:
        valid_sequences = json.load(f)

    # Choose a random key whose value is equivalent to convert_from arg from the valid sequences as file_path
    valid_file_paths = [key for key, value in valid_sequences.items() if value == configs['generation']['convert_from']]
    midi_file_path = configs['generation']['midi_file_path']
    audio_file_path = configs['generation']['wav_file_path']
    if midi_file_path is None or midi_file_path == "":
        midi_file_path = random.choice(valid_file_paths)
        audio_file_path = None

    convert_to = configs['generation']['convert_to']
    context_before = configs['generation']['context_before']
    context_after = configs['generation']['context_after']
    corruption_passes = configs['generation']['passes']
    t_segment_start = configs['generation']['t_segment_start']
    novel_peaks_pct = configs['generation']['novel_peaks_pct']
    write_intermediate_passes = configs['generation']['write_intermediate_passes']
    temperature = configs['generation']['temperature']
    end_original = configs['generation']['end_original']
    t_segment_stop = configs['generation']['t_segment_stop']

    generate(midi_file_path, audio_file_path, fusion_model, configs, novel_peaks_pct,
             t_segment_start, convert_to, context_before, context_after, 
             corruption_passes, tokenizer, decode_tokenizer, output_folder, 
             save_original=True, quiet=False, write_intermediate_passes=write_intermediate_passes, 
             temperature=temperature, end_original=end_original, t_segment_stop=t_segment_stop)