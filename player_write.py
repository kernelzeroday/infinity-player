"Shuffle your favorite songs into infinite remixes."

import argparse
import gzip
import logging
import os
import pickle
import shutil
from random import random

import librosa
import numpy
import soundfile as sf
import torch
from PIL import Image
from tqdm import tqdm

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'timbre.pickle'), 'rb') as fh:
    TIMBRE_PATTERNS = pickle.load(fh)


def print_progress(i, n):
    cols, lines = shutil.get_terminal_size()
    pos = i * (cols - 5) // n
    s = ''
    for x in range(cols - 5):
        if x == pos:
            s += '|'
        elif x < pos:
            s += '='
        else:
            s += '-'
    s += f' {i:>4}'
    print(s, end='\r')


def enhance_diagonals(jumps, weight=0.2, steps=1):
    for i in range(steps):
        # combine each cell with its diagonal neighbors
        jumps1 = numpy.roll(jumps, (1, 1), (0, 1))
        jumps2 = numpy.roll(jumps, (-1, -1), (0, 1))
        jumps = (weight * (jumps1 + jumps2) + (1 - weight) * jumps) / 2
    return jumps


def iter_beat_slices(y, beat_frames):
    beat_samples = librosa.frames_to_samples(beat_frames)
    beat_samples = [0, *beat_samples, len(y) - 1]
    for start, end in zip(beat_samples[0:-1], beat_samples[1:]):
        yield start, end


def timbre(y):
    spectrum = numpy.abs(librosa.stft(y))
    resized = numpy.array(Image.fromarray(spectrum).resize((70, 50)))

    k = len(TIMBRE_PATTERNS)
    s = numpy.zeros((k, 1))

    for i, pattern in enumerate(TIMBRE_PATTERNS):
        # Assuming pattern is a single value or compatible shape
        pattern_matrix = numpy.full(resized.shape, pattern)
        s[i][0] = numpy.sum(pattern_matrix * resized)
    return s


def analyze(y, sample_rate, beat_frames):
    tim = numpy.array([
        timbre(y[start:end]) for start, end in iter_beat_slices(y, beat_frames)
    ]).T
    # Adjust width based on data size
    max_width = (tim.shape[1] - 1) // 2  # Calculate maximum width based on the shape of tim
    width = min(4, max_width)  # Ensure width is within valid range
    return librosa.segment.recurrence_matrix(tim, width=width, mode='affinity')


def load(filename, force=False):
    try:
        y, sample_rate = librosa.load(filename, mono=False)
    except UserWarning:
        y, sample_rate = librosa.load(filename, sr=None, mono=False)

    fn_inf = filename + '.inf'
    if not force and os.path.exists(fn_inf):
        with gzip.open(fn_inf, 'rb') as fh:
            beat_frames, jumps = pickle.load(fh)
            logging.info(f'Loaded analysis data from {fn_inf}')
    else:
        logging.info('Analyzing…')
        y1, sample_rate1 = librosa.load(filename)
        tempo, beat_frames = librosa.beat.beat_track(y=y1, sr=sample_rate1)
        jumps = analyze(y1, sample_rate1, beat_frames)
        logging.info(f'Analysis complete. Tempo: {tempo}, Beat frames: {len(beat_frames)}')

        with gzip.open(fn_inf, 'wb') as fh:
            pickle.dump((beat_frames, jumps), fh)
            logging.info(f'Saved analysis data to {fn_inf}')

    return y, sample_rate, beat_frames, jumps


def compute_buffers(y, beat_frames):
    int_max = numpy.iinfo(numpy.int16).max
    raw = (y * int_max).astype(numpy.int16).T.copy(order='C')

    buffers = []
    for start, end in tqdm(iter_beat_slices(raw, beat_frames), desc="Computing buffers"):
        buffers.append(y.T[start:end])
        logging.debug(f'Computed buffer from {start} to {end}')

    return buffers


def normalize(jumps, threshold, prefer_start, prefer_middle, avoid_end, buffer_language_tokens=None, prefer_words=False, no_language=False):
    n = len(jumps)

    logging.debug('Enhancing diagonals...')
    jumps = enhance_diagonals(jumps, 0.8, 4)

    # scale
    x_max = jumps.max()
    x_min = x_max * threshold
    y_max = x_max ** 0.5
    jumps = (jumps - x_min) / (x_max - x_min) * y_max
    jumps *= jumps > 0
    logging.debug('Scaling complete.')

    # privilege jumps back in order to prolong playing
    jumps *= numpy.ones((n, n)) - numpy.tri(n, k=-1).T * 0.5
    logging.debug('Privileged jumps back.')

    # privilege wide jumps
    M = numpy.zeros((n, n))
    for i in range(1, n):
        M += numpy.tri(n, k=-i)
        M += numpy.tri(n, k=-i).T
    jumps *= (M / (n - 1)) ** 0.4
    logging.debug('Privileged wide jumps.')

    # prefer the beginning and middle of songs, avoid the end
    for i in range(n):
        if i < n * prefer_start:
            jumps[i] *= 1.5
        elif i > n * (1 - avoid_end):
            jumps[i] *= 0.5
        elif i < n * (1 - avoid_end) and i > n * prefer_middle:
            jumps[i] *= 1.2
    logging.debug('Adjusted jumps based on song position preferences.')

    # prefer or avoid chunks with language tokens
    if buffer_language_tokens is not None:
        for i in range(n):
            if buffer_language_tokens[i]:
                jumps[i] *= 1.5 if prefer_words else 0.5
            if no_language and buffer_language_tokens[i]:
                jumps[i] = 0
        logging.debug('Adjusted jumps based on language tokens.')

    return jumps


def get_next_position(i, jumps):
    for j, p in sorted(enumerate(jumps[i]), key=lambda jp: -jp[1]):
        if p > random():
            logging.debug(f'Jumping from position {i} to {j + 1}')
            return j + 1
    logging.debug(f'No jump from position {i}, moving to {i + 1}')
    return i + 1


def save_to_files(buffers, sample_rate, jumps, output_dir, num_songs, min_length, max_length, buffer_language_tokens=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f'Created output directory {output_dir}')

    for song_num in tqdm(range(num_songs), desc="Saving songs"):
        i = 0
        n = len(buffers)
        song_length = 0
        song_buffers = []

        while True:
            song_buffers.append(buffers[i])
            print_progress(i, n)

            song_length += len(buffers[i]) / sample_rate
            i = get_next_position(i, jumps)
            if i >= n:
                i = 0

            if max_length and song_length >= max_length:
                break

        if min_length and song_length < min_length:
            logging.warning(f'Song {song_num} is shorter than the minimum length of {min_length} seconds.')
        else:
            # Ensure the final window of the song cannot be in the middle of the remix
            if song_buffers[-1] is not buffers[-1]:
                song_buffers.append(buffers[-1])
            song_filename = os.path.join(output_dir, f'song_{song_num}.wav')
            sf.write(song_filename, numpy.concatenate(song_buffers), sample_rate)
            logging.info(f'Saved song {song_num} to {song_filename}')


def parse_args():
    parser = argparse.ArgumentParser(description="Shuffle your favorite songs into infinite remixes.")
    parser.add_argument('--filename', required=True, help='Path to the input audio file.')
    parser.add_argument('--output_dir', required=True, help='Directory to save the output files.')
    parser.add_argument('--num_songs', type=int, required=True, help='Number of songs to produce.')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.8, help='Between 0 and 1. '
        'A higher value will result in fewer but better jumps. (Default: 0.8)')
    parser.add_argument(
        '-f', '--force', action='store_true',
        help='Ignore previously saved analysis data.')
    parser.add_argument(
        '--min_length', type=float, default=None, help='Minimum desired length of each song in seconds.')
    parser.add_argument(
        '--max_length', type=float, default=None, help='Maximum desired length of each song in seconds.')
    parser.add_argument(
        '--prefer_start', type=float, default=0.1, help='Proportion of the song to prefer at the start. (Default: 0.1)')
    parser.add_argument(
        '--prefer_middle', type=float, default=0.5, help='Proportion of the song to prefer in the middle. (Default: 0.5)')
    parser.add_argument(
        '--avoid_end', type=float, default=0.1, help='Proportion of the song to avoid at the end. (Default: 0.1)')
    parser.add_argument(
        '--use_whisper', action='store_true', help='Use Whisper to detect language and prefer chunks without language tokens.')
    parser.add_argument(
        '--whisper_lang', type=str, default='en', help='Language to use for Whisper transcription. Default is "en".')
    parser.add_argument(
        '--prefer_words', action='store_true', help='Prefer sections with detected words.')
    parser.add_argument(
        '--no_language', action='store_true', help='Do not use any segments with detected language.')
    parser.add_argument(
        '--device', type=str, default=None, help='Torch device to use for Whisper (e.g., "cpu", "cuda").')
    parser.add_argument(
        '--list_language_sections', action='store_true', help='List detected language sections without generating files.')
    parser.add_argument(
        '--dry_run', action='store_true', help='Run the process without saving any files.')
    parser.add_argument(
        '--no_post_language_detection_processing', action='store_true', help='Skip post language detection processing.')
    return parser.parse_args()


def detect_language_tokens(filename, buffers, sample_rate, device=None):
    logging.info("Using internal Whisper library for language detection.")
    model = whisper.load_model("large", device=device)
    result = model.transcribe(
        filename,
    )
    logging.info("Audio transcription complete.")
    
    # create language tokens array
    buffer_language_tokens = [False] * len(buffers)
    for segment in tqdm(result['segments'], desc="Processing segments"):
        start = int(segment['start'] * sample_rate)
        end = int(segment['end'] * sample_rate)
        for i, (buffer_start, buffer_end) in enumerate(iter_beat_slices(numpy.zeros(len(buffers)), range(len(buffers)))):
            if buffer_start <= start < buffer_end or buffer_start < end <= buffer_end:
                buffer_language_tokens[i] = True
                logging.debug(f'Language detected in buffer {i} from {buffer_start} to {buffer_end}')
    return buffer_language_tokens

def list_language_sections(buffer_language_tokens):
    for i, has_language in enumerate(buffer_language_tokens):
        if has_language:
            print(f'Buffer {i} contains language.')

def main():
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()

    logging.info('Loading %s', args.filename)
    y, sample_rate, beat_frames, jumps = load(args.filename, args.force)

    buffers = compute_buffers(y, beat_frames)
    buffer_language_tokens = None
    if args.use_whisper and WHISPER_AVAILABLE:
        logging.info('Detecting language tokens in the input song using Whisper...')
        try:
            buffer_language_tokens = detect_language_tokens(args.filename, buffers, sample_rate, args.device)
        except KeyboardInterrupt:
            logging.error('Language detection interrupted by user.')
            return

    if args.list_language_sections and buffer_language_tokens is not None:
        list_language_sections(buffer_language_tokens)
        return

    if not args.no_post_language_detection_processing:
        jumps = normalize(jumps, args.threshold, args.prefer_start, args.prefer_middle, args.avoid_end, buffer_language_tokens, args.prefer_words, args.no_language)
        jump_count = sum(sum(jumps > 0))

        logging.info('Detected %d jump opportunities on %d beats', jump_count, len(buffers))

    if not args.dry_run:
        logging.info('Saving to files…')
        save_to_files(buffers, sample_rate, jumps, args.output_dir, args.num_songs, args.min_length, args.max_length, buffer_language_tokens)


if __name__ == '__main__':
    main()

