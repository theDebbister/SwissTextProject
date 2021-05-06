import numpy as np
import shlex
import subprocess
import sys
import wave
import json
import os
from os.path import join
from deepspeech import Model, version
from timeit import default_timer as timer

# ARGUMENTS
model_file_path = 'models/output_graph.pb'
scorer_file_path = 'data/german-text-corpus/kenlm.scorer'
lm_file_path = 'data/german-text-corpus/lm.binary'
trie_file_path = 'trie'
beam_width = 500
transcription_candidates = 1
sampleAudioFile = '../data/data_swisstext/clips/audio/00020feb-9179-4892-a925-ba8e5e9aea25.wav'
audio_file_path = '../data/data_swisstext/clips/audio/'
transcription_file = '../data/transcription.out'

def append_result(filename, transcription):
    with open(transcription_file, "a") as file_object:
        file_object.write(filename + ","+transcription+"\n")

def metadata_to_string(metadata):
    return ''.join(token.text for token in metadata.tokens)

def words_from_candidate_transcript(metadata):
    word = ""
    word_list = []
    word_start_time = 0
    # Loop through each character
    for i, token in enumerate(metadata.tokens):
        # Append character to word if it's not a space
        if token.text != " ":
            if len(word) == 0:
                # Log the start time of the new word
                word_start_time = token.start_time

            word = word + token.text
        # Word boundary is either a space or the last character in the array
        if token.text == " " or i == len(metadata.tokens) - 1:
            word_duration = token.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = dict()
            each_word["word"] = word
            each_word["start_time"] = round(word_start_time, 4)
            each_word["duration"] = round(word_duration, 4)

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0

    return word_list

def metadata_json_output(metadata):
    json_result = dict()
    json_result["transcripts"] = [{
        "confidence": transcript.confidence,
        "words": words_from_candidate_transcript(transcript),
    } for transcript in metadata.transcripts]
    return json.dumps(json_result, indent=2)

def transcribe_audio_file(audio_file_path, ds):
    fin = wave.open(audio_file_path, 'rb')
    fs_orig = fin.getframerate()
    desired_sample_rate = ds.sampleRate()
    if fs_orig != desired_sample_rate:
        print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(fs_orig, desired_sample_rate), file=sys.stderr)
        fs_new, audio = convert_samplerate(sampleAudioFile, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1/fs_orig)
    fin.close()
    text = ''
    if transcription_candidates > 1:
        text = metadata_json_output(ds.sttWithMetadata(audio, transcription_candidates))
    else:
        text = ds.stt(audio)
    return text

def transcribe_folder(audio_folder_path,ds):
    ext = ('.wav')
    #onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for files in os.listdir(audio_folder_path):
        if files.endswith(ext):
            append_result(files, transcribe_audio_file(join(audio_folder_path,files), ds))

def main():
    print('Loading model from file {}'.format(model_file_path), file=sys.stderr)
    model_load_start = timer()
    # sphinx-doc: python_ref_model_start
    ds = Model(model_file_path)
    # sphinx-doc: python_ref_model_stop
    model_load_end = timer() - model_load_start
    print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

    #if args.beam_width:
    #    ds.setBeamWidth(args.beam_width)

    desired_sample_rate = ds.sampleRate()

    print('Loading scorer from files {}'.format(scorer_file_path), file=sys.stderr)
    scorer_load_start = timer()
    ds.enableExternalScorer(scorer_file_path)
    scorer_load_end = timer() - scorer_load_start
    print('Loaded scorer in {:.3}s.'.format(scorer_load_end), file=sys.stderr)

    #if args.lm_alpha and args.lm_beta:
    #    ds.setScorerAlphaBeta(args.lm_alpha, args.lm_beta)

#    if args.hot_words:
#        print('Adding hot-words', file=sys.stderr)
#        for word_boost in args.hot_words.split(','):
#            word,boost = word_boost.split(':')
#            ds.addHotWord(word,float(boost))
#
    print('Running inference.', file=sys.stderr)
    inference_start = timer()
    #if args.extended:
    #    print(metadata_to_string(ds.sttWithMetadata(audio, 1).transcripts[0]))
    #elif args.json:
    #print(metadata_json_output(ds.sttWithMetadata(audio, 3)))
    #else:
    #print(ds.stt(audio))

    transcribe_folder(audio_file_path, ds)
    inference_end = timer() - inference_start
    print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)

if __name__ == '__main__':
    main()
