from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer
import pickle
import argparse
import yaml
from tqdm import tqdm
import torch
import torch.nn.functional as F
import librosa
import json
import numpy as np
import glob
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from transformers import AutoProcessor, ClapModel
from transformers import AutoModelForSequenceClassification
from data_loader import Genre_Classifier_Dataset
from corruptions import DataCorruption
import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
# os.sched_setaffinity(0, {0, 1, 2, 3, 4, 5})
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.utils import convert_midi_to_wav

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), "eval"))
from eval.midi_obj_eval.single_midi_eval import evaluate_single_midi, compare_single_midi_metrics
from eval.midi_obj_eval.single_midi_eval import plot_pitch_class_histogram_pair, plot_pitch_class_transition_matrix_pair
from eval.tonal_tension_muspy.metrics import compute_tonal_tension, compute_muspy_metrics
from eval.tonal_tension_muspy.plot import plot_tonal_tension_comparison, plot_muspy_comparison

torch.set_num_threads(6)
torch.set_num_interop_threads(6)

def get_genre_probabilities(midi_file_path, tokenizer, model, dataset_obj, encoder_max_sequence_length, aria_tokenizer, verbose=True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Read a midi file
    mid = MidiDict.from_midi(midi_file_path)
    # Tokenize the midi file
    tokens = aria_tokenizer.tokenize(mid)

    corruption_obj = DataCorruption()
    separated_sequence = corruption_obj.seperateitems(tokens)
    # Get the indices of the novelty tokens
    novelty_segments = [n for n, i in enumerate(separated_sequence) if '<N>' in i]
    all_segment_indices, _, _, _ = corruption_obj.get_segment_to_corrupt(separated_sequence, t_segment_ind=2, exclude_idx=novelty_segments)

    t_segment_ind = 0
    n_iterations = len(all_segment_indices) - 1
    jump_every = 1
    # Initialize tqdm
    disable = True if not verbose else False
    progress_bar = tqdm(total=n_iterations, disable=disable)
    genre_probs = {'classical': [], 'pop': [], 'jazz': []}

    while t_segment_ind < n_iterations:
        # Get the cropped sequence
        input_tokens = dataset_obj.get_cropped_sequence(tokens, meta_data=None, t_segment_ind=t_segment_ind)

        # Tokenize the sequences
        input_tokens = [tokenizer[tuple(token)] if isinstance(token, list) else tokenizer[token] for token in input_tokens]

        # Pad the sequences
        if len(input_tokens) < encoder_max_sequence_length:
            input_tokens = F.pad(torch.tensor(input_tokens), (0, encoder_max_sequence_length - len(input_tokens))).to(torch.int64)
        else:
            input_tokens = torch.tensor(input_tokens[0:encoder_max_sequence_length]).to(torch.int64)

        # Attention mask based on non-padded tokens of the phrase
        attention_mask = torch.where(input_tokens != 0, 1, 0).type(torch.bool)

        # Move the tensors to the device
        input_tokens = input_tokens.to(device)
        attention_mask = attention_mask.to(device)

        # Get the prediction
        outputs = model(input_tokens.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        logits = outputs.logits
        # Get probabilities from the logits
        probs = F.softmax(logits, dim=1)
        genre_probs['classical'].append(probs[0][0].item())
        genre_probs['jazz'].append(probs[0][2].item())

        t_segment_ind += jump_every
        # Update the progress bar
        progress_bar.update(jump_every)

    progress_bar.close()

    if verbose:
        # Print average probabilities by genre
        print(f"Average probability for classical: {sum(genre_probs['classical'])/len(genre_probs['classical'])}")
        print(f"Average probability for jazz: {sum(genre_probs['jazz'])/len(genre_probs['jazz'])}")

    avg_classical_prob = float(sum(genre_probs['classical'])/len(genre_probs['classical']))
    avg_jazz_prob = float(sum(genre_probs['jazz'])/len(genre_probs['jazz']))

    return {'Jazz': avg_jazz_prob, 'Classical': avg_classical_prob}


def generate_ssm(wav_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the audio file and take the first n minutes to fit into memory
    y, sr = librosa.load(wav_file, duration=60*5, sr=None, mono=True)

    # Duration of the audio file in minutes and seconds
    duration = librosa.get_duration(y=y, sr=sr)
    duration = f"{int(duration // 60)} minutes and {int(duration % 60)} seconds"

    #### Chromagram using librosa ####
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=1024)

    # Chromagram for shorter audio file
    shorter_samples = sr * 60 * 5 # First n minutes to fit into memory
    if len(y) > shorter_samples:    
        # Slice the audio signal to get the first 5 minutes
        y_short = y[:shorter_samples]
        chroma_cqt_short = librosa.feature.chroma_cqt(y=y_short, sr=sr, hop_length=1024)
    else:
        chroma_cqt_short = chroma_cqt

    ## norm of chroma ##
    chroma_norm = np.linalg.norm(chroma_cqt_short, axis=0)

    # Convert to torch tensor
    chroma = torch.from_numpy(chroma_cqt_short).float().to(device)
    chroma_norm = torch.from_numpy(chroma_norm).float().to(device)

    # Compute dot products
    dot_products = torch.matmul(chroma.T, chroma)

    # Compute norms product
    norms_product = chroma_norm[:, None] * chroma_norm[None, :]

    # Compute cosine similarity
    ssm = dot_products / norms_product

    #### normalize SSM ####
    ssm_normalized = (ssm - torch.min(ssm)) / (torch.max(ssm) - torch.min(ssm))

    #### Flip up-side-down ####
    ssm_flipped = torch.flipud(ssm_normalized)

    # Rescale ssm_normalized
    ssm_rescaled = F.interpolate(ssm_flipped.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear').squeeze().cpu().numpy()

    return ssm_rescaled, chroma_cqt


def compare_similarity_matrices(original_wav_file, generated_wav_file, verbose=True):
    # Generate the similarity matrices
    original_ssm, original_chroma_cqt = generate_ssm(original_wav_file)
    generated_ssm, generated_chroma_cqt = generate_ssm(generated_wav_file)

    # Ensure the matrices are numpy arrays
    original_ssm = np.array(original_ssm)
    generated_ssm = np.array(generated_ssm)

    # Check if the matrices have the same shape
    if original_ssm.shape != generated_ssm.shape:
        raise ValueError("Matrices must have the same dimensions")

    # Calculate Mean Squared Error (MSE)
    mse_value = float(mean_squared_error(original_ssm, generated_ssm))

    # Calculate Structural Similarity Index (SSIM)
    ssim_value = float(ssim(original_ssm, generated_ssm, data_range=generated_ssm.max() - generated_ssm.min()))

    # Calculate Pearson Correlation Coefficient
    pearson_corr, _ = pearsonr(original_ssm.flatten(), generated_ssm.flatten())
    pearson_corr = float(pearson_corr)

    # Calculate Frobenius Norm
    frobenius_norm = float(np.linalg.norm(original_ssm - generated_ssm, 'fro'))

    # Calculate Cosine Similarity
    cosine_sim = cosine_similarity(original_ssm.flatten().reshape(1, -1), generated_ssm.flatten().reshape(1, -1))
    cosine_sim = float(cosine_sim[0][0])

    # Calculate Chroma Cosine Similarity
    chroma_similarity = cosine_similarity(original_chroma_cqt.T, generated_chroma_cqt.T)
    # Averaged similarity across all time frames
    avg_chroma_similarity = float(np.mean(chroma_similarity))

    if verbose:
        print(f"SSM Mean Squared Error: {mse_value}")
        print(f"SSM SSIM: {ssim_value}")
        print(f"SSM Pearson Correlation: {pearson_corr}")
        print(f"SSM Frobenius Norm: {frobenius_norm}")
        print(f"SSM Cosine Similarity: {cosine_sim}")
        print(f"Average Chroma Cosine Similarity: {avg_chroma_similarity}")

    return {
        'SSM MSE': mse_value,
        'SSM SSIM': ssim_value,
        'SSM Pearson Correlation': pearson_corr,
        'SSM Frobenius Norm': frobenius_norm,
        'SSM Cosine Similarity': cosine_sim,
    }, avg_chroma_similarity


def clap_similarity_score(audio_sample, text, model, processor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Preprocess inputs
    text_input = processor(text=[text], return_tensors="pt").to(device)
    audio_input = processor(audios=audio_sample, return_tensors="pt", sampling_rate=48000).to(device)

    # Generate embeddings
    text_embeddings = model.get_text_features(**text_input)
    audio_embeddings = model.get_audio_features(**audio_input)

    # Normalize embeddings
    text_embeddings = text_embeddings / torch.norm(text_embeddings, dim=1, keepdim=True)
    audio_embeddings = audio_embeddings / torch.norm(audio_embeddings, dim=1, keepdim=True)

    # Compute similarity score
    similarity_score = torch.mm(text_embeddings, audio_embeddings.T)

    # Clear GPU memory
    torch.cuda.empty_cache()

    return similarity_score


def compare_clap_similarity_score(wav_filepath, model, processor, verbose=True):
    audio_sample, sr = librosa.load(wav_filepath, sr=None, mono=True, duration=60*3)
    total_duration = audio_sample.shape[0] / sr

    segment_duration = 5  # Duration of each segment in seconds
    segment_samples = int(segment_duration * sr)  # Number of samples in each segment

    classical_similarity_scores = []
    jazz_similarity_scores = []
    disable = True if not verbose else False
    for i in tqdm(range(0, int(total_duration / segment_duration)), desc="Processing segments", disable=disable):
        start_sample = i * segment_samples
        end_sample = start_sample + segment_samples

        segment_audio = audio_sample[start_sample:end_sample]

        classical_similarity_score = clap_similarity_score(segment_audio, "Classical piano music", model, processor)
        jazz_similarity_score = clap_similarity_score(segment_audio, "Jazz piano music", model, processor)
        classical_similarity_scores.append(classical_similarity_score)
        jazz_similarity_scores.append(jazz_similarity_score)
    
    # Average the similarity score values with numpy
    classical_similarity_scores = float(np.mean(torch.cat(classical_similarity_scores).detach().cpu().numpy()))
    jazz_similarity_scores = float(np.mean(torch.cat(jazz_similarity_scores).detach().cpu().numpy()))
    if verbose:
        print(f"Average similarity score for classical music: {classical_similarity_scores}")
        print(f"Average similarity score for jazz music: {jazz_similarity_scores}")

    clap_scores = {
        "Classical Similarity Score": classical_similarity_scores,
        "Jazz Similarity Score": jazz_similarity_scores
    }

    return clap_scores
        
    



if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.normpath("configs/config_style_transfer.yaml"),
                        help="Path to the config file")
    parser.add_argument("--experiment_name", type=str, default="all",
                    help="Name of the experiment")
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)
        
    artifact_folder = configs["raw_data"]["artifact_folder"]
    raw_data_folders = configs["raw_data"]["raw_data_folders"]
    eval_folder = configs["raw_data"]["eval_folder"]

    if args.experiment_name == "experiment_1" or args.experiment_name == "all":

        # Get the encoder max sequence length
        encoder_max_sequence_length = configs['classifier_model']['encoder_max_sequence_length']

        # Get tokenizer
        tokenizer_filepath = os.path.join(artifact_folder, "style_transfer", "vocab_corrupted.pkl")
        # Load the tokenizer dictionary
        with open(tokenizer_filepath, "rb") as f:
            tokenizer = pickle.load(f)

        aria_tokenizer = AbsTokenizer()

        # Load the model
        model_path = os.path.join(artifact_folder, "style_transfer", "classifier_model")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        print("Genre classifier model loaded")

        clap_model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
        clap_model.eval()
        processor = AutoProcessor.from_pretrained("laion/larger_clap_music_and_speech")
        print("CLAP model loaded")

        # Load the dataset
        dataset_obj = Genre_Classifier_Dataset(configs, data_list=[], mode="eval", shuffle=False)

        # Get file paths of the original and generated midi files from eval_folder
        all_midi_files = glob.glob(os.path.join(eval_folder, "**/*.mid"), recursive=True)
        original_midi_file_paths = [f for f in all_midi_files if "generated" not in f and "pop" not in f and "harmony" not in f and "experiment_5" not in f and "experiment_6" not in f]
        # # Split original_midi_file_paths into 5 parts
        # original_midi_file_paths_splits = [original_midi_file_paths[i:i + len(original_midi_file_paths)//4] for i in range(0, len(original_midi_file_paths), len(original_midi_file_paths)//4)]
        # original_midi_file_paths = original_midi_file_paths_splits[3]
        # Filter out all files from 0 until the specified file
        # filter_until = "evaluations/classical/34/original_34.mid"
        # original_midi_file_paths = original_midi_file_paths[original_midi_file_paths.index(filter_until):]
        generated_midi_file_paths = [f for f in all_midi_files if "generated" in f and "pop" not in f and "harmony" not in f and "experiment_5" not in f and "experiment_6" not in f]
        # generated_midi_file_paths = [f for f in generated_midi_file_paths if "experiment_3" in f or "experiment_4" in f]
        print("Number of original midi files: ", len(original_midi_file_paths))
        print("Number of generated midi files: ", len(generated_midi_file_paths))

        for original_midi_file_path in original_midi_file_paths:
            matching_generation_file_paths = [f for f in generated_midi_file_paths if os.path.join(os.path.dirname(original_midi_file_path), "experiment_") in f]
            original_genre_probs = get_genre_probabilities(original_midi_file_path, tokenizer, model, dataset_obj, encoder_max_sequence_length, aria_tokenizer, verbose=False)
            for generated_midi_file_path in tqdm(matching_generation_file_paths):

                generated_midi_folder = os.path.dirname(generated_midi_file_path)
                print("Processing: ", generated_midi_file_path)

                # Check if the metrics file already exists
                if os.path.exists(os.path.join(generated_midi_folder, "metrics.json")):
                    print("Metrics file already exists")
                    continue

                generated_genre_probs = get_genre_probabilities(generated_midi_file_path, tokenizer, model, dataset_obj, encoder_max_sequence_length, aria_tokenizer, verbose=False)

                # Convert the midi files to wav
                original_wav_file = original_midi_file_path.replace(".mid", ".wav")
                generated_wav_file = convert_midi_to_wav([generated_midi_file_path], "/homes/kb658/fusion/artifacts/soundfont.sf", max_workers=1, verbose=False)
                generated_wav_file = generated_wav_file[0]
                print("Wav file created")

                # Get the similarity matrices
                try:
                    ssm_score, chroma_score = compare_similarity_matrices(original_wav_file, generated_wav_file, verbose=False)
                except Exception as e:
                    ssm_score, chroma_score = None, None
                print("Similarity matrices calculated")

                # # Get the CLAP similarity scores
                # original_clap_score = compare_clap_similarity_score(original_wav_file, clap_model, processor, verbose=False)
                # generated_clap_score = compare_clap_similarity_score(generated_wav_file, clap_model, processor, verbose=False)
                # print("CLAP similarity scores calculated")

                metrics = {
                    "Original Genre Probabilities": original_genre_probs,
                    "Generated Genre Probabilities": generated_genre_probs,
                    "SSM Similarity Score": ssm_score,
                    "Chroma Frame Similarity Score": chroma_score,
                    # "Original CLAP Score": original_clap_score,
                    # "Generated CLAP Score": generated_clap_score
                }

                # Save the metrics as a JSON file
                with open(os.path.join(generated_midi_folder, "metrics.json"), "w") as f:
                    json.dump(metrics, f)

                # Delete the generated wav file
                os.remove(generated_wav_file)

                # Clear GPU memory
                torch.cuda.empty_cache()

        print("Evaluation completed")


    if args.experiment_name == "experiment_2" or args.experiment_name == "all":

        all_midi_files = glob.glob(os.path.join(eval_folder, "harmony", "**/*.mid"), recursive=True)
        original_midi_file_paths = [f for f in all_midi_files if "generated" not in f and "monophonic" not in f and "random_chords" not in f]
        monophonic_original_midi_file_paths = [f for f in all_midi_files if "generated" not in f and "original_monophonic" in f]
        generated_midi_file_paths = [f for f in all_midi_files if "generated" in f]
        random_chords_midi_file_paths = [f for f in all_midi_files if "random_chords" in f]

        print("Number of original midi files: ", len(original_midi_file_paths))
        print("Number of generated midi files: ", len(generated_midi_file_paths))
        print("Number of monophonic original midi files: ", len(monophonic_original_midi_file_paths))
        print("Number of random chords midi files: ", len(random_chords_midi_file_paths))

        for original_midi_file_path in original_midi_file_paths:
            matching_generation_file_paths = [f for f in generated_midi_file_paths if os.path.join(os.path.dirname(original_midi_file_path), "pass_") in f]
            matching_random_chords_file_paths = [f for f in random_chords_midi_file_paths if os.path.join(os.path.dirname(original_midi_file_path), "pass_") in f]
            matching_monophonic_original_midi_file_paths = [f for f in monophonic_original_midi_file_paths if os.path.dirname(original_midi_file_path) in f]

            original_metrics = evaluate_single_midi(original_midi_file_path, return_numpy=False)            
            original_monophonic_metrics = evaluate_single_midi(matching_monophonic_original_midi_file_paths[0], return_numpy=False)
            output_org_muspy = compute_muspy_metrics(original_midi_file_path, key="")
            output_org_tension = compute_tonal_tension(original_midi_file_path, key="")

            for generated_midi_file_path in tqdm(matching_generation_file_paths):

                generated_midi_folder = os.path.dirname(generated_midi_file_path)
                print("Processing: ", generated_midi_file_path)
                
                generated_metrics = evaluate_single_midi(generated_midi_file_path, return_numpy=False)
                metric_pairs = compare_single_midi_metrics(original_metrics, generated_metrics)
                monophonic_metric_pairs = compare_single_midi_metrics(original_monophonic_metrics, generated_metrics)
                output_gen_muspy = compute_muspy_metrics(generated_midi_file_path, key="")
                output_gen_tension = compute_tonal_tension(generated_midi_file_path, key="")

                # Combine the metrics dictionaries
                objective_metrics = {
                    "Metric Pairs": metric_pairs,
                    "Monophonic Metric Pairs": monophonic_metric_pairs,
                    "Original MusPy Metrics": output_org_muspy,
                    "Generated MusPy Metrics": output_gen_muspy,
                    "Original Tonal Tension Diameter": np.mean(output_org_tension['diameter']),
                    "Generated Tonal Tension Diameter": np.mean(output_gen_tension['diameter']),
                    "Original Tonal Tension Tensile": np.mean(output_org_tension['tensile']),
                    "Generated Tonal Tension Tensile": np.mean(output_gen_tension['tensile']),
                    "Original Tonal Tension Centroid Diff": np.mean(output_org_tension['centroid_diff']),
                    "Generated Tonal Tension Centroid Diff": np.mean(output_gen_tension['centroid_diff']),
                }
                
                # Save the metrics as a JSON file
                with open(os.path.join(generated_midi_folder, "metrics.json"), "w") as f:
                    json.dump(objective_metrics, f)
                print("Metrics saved in: ", generated_midi_folder)

            for random_chords_midi_file_path in matching_random_chords_file_paths:
                generated_metrics = evaluate_single_midi(random_chords_midi_file_path, return_numpy=False)
                metric_pairs = compare_single_midi_metrics(original_metrics, generated_metrics)
                monophonic_metric_pairs = compare_single_midi_metrics(original_monophonic_metrics, generated_metrics)
                output_gen_muspy = compute_muspy_metrics(random_chords_midi_file_path, key="")
                output_gen_tension = compute_tonal_tension(random_chords_midi_file_path, key="")

                # Combine the metrics dictionaries
                objective_metrics = {
                    "Metric Pairs": metric_pairs,
                    "Monophonic Metric Pairs": monophonic_metric_pairs,
                    "Original MusPy Metrics": output_org_muspy,
                    "Generated MusPy Metrics": output_gen_muspy,
                    "Original Tonal Tension Diameter": np.mean(output_org_tension['diameter']),
                    "Generated Tonal Tension Diameter": np.mean(output_gen_tension['diameter']),
                    "Original Tonal Tension Tensile": np.mean(output_org_tension['tensile']),
                    "Generated Tonal Tension Tensile": np.mean(output_gen_tension['tensile']),
                    "Original Tonal Tension Centroid Diff": np.mean(output_org_tension['centroid_diff']),
                    "Generated Tonal Tension Centroid Diff": np.mean(output_gen_tension['centroid_diff']),
                }
                
                # Save the metrics as a JSON file
                with open(os.path.join(os.path.dirname(random_chords_midi_file_path), "random_chords_metrics.json"), "w") as f:
                    json.dump(objective_metrics, f)
                print("Metrics saved in: ", os.path.dirname(random_chords_midi_file_path))

        print("Evaluation completed")
    
    if args.experiment_name == "experiment_3" or args.experiment_name == "all":

        all_midi_files = glob.glob(os.path.join(eval_folder, "**/*.mid"), recursive=True)
        original_midi_file_paths = [f for f in all_midi_files if "generated" not in f and ("experiment_5" in f or "experiment_6" in f) and "original_segment" in f]
        generated_midi_file_paths = [f for f in all_midi_files if "generated" in f and ("experiment_5" in f or "experiment_6" in f) and "amt" not in f]
        amt_generated_midi_file_paths = [f for f in all_midi_files if "amt" in f and ("experiment_5" in f or "experiment_6" in f)]
        print("Number of original midi files: ", len(original_midi_file_paths))
        print("Number of generated midi files: ", len(generated_midi_file_paths))
        print("Number of amt generated midi files: ", len(amt_generated_midi_file_paths))
              
        for original_midi_file_path in original_midi_file_paths:
            matching_generation_file_paths = [f for f in generated_midi_file_paths if os.path.dirname(original_midi_file_path) in f and "generated" in f]
            matching_amt_generation_file_paths = [f for f in amt_generated_midi_file_paths if os.path.dirname(original_midi_file_path) in f and "amt" in f]
            generated_midi_folder = os.path.dirname(matching_generation_file_paths[0])
            print("Processing: ", generated_midi_folder)

            original_metrics = evaluate_single_midi(original_midi_file_path, return_numpy=False)
            generated_metrics = evaluate_single_midi(matching_generation_file_paths[0], return_numpy=False)
            metric_pairs = compare_single_midi_metrics(original_metrics, generated_metrics)
            
            if len(matching_amt_generation_file_paths) > 0:
                amt_generated_metrics = evaluate_single_midi(matching_amt_generation_file_paths[0], return_numpy=False)
                amt_metric_pairs = compare_single_midi_metrics(original_metrics, amt_generated_metrics)

                # Save the amt metrics as a JSON file
                with open(os.path.join(generated_midi_folder, "amt_metrics.json"), "w") as f:
                    json.dump(amt_metric_pairs, f)

            # Save the metrics as a JSON file
            with open(os.path.join(generated_midi_folder, "metrics.json"), "w") as f:
                json.dump(metric_pairs, f)

            print("Metrics saved in: ", generated_midi_folder)

        print("Evaluation completed")