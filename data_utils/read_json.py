import json
import numpy as np
import re
from pydub import AudioSegment
from tqdm import tqdm
import argparse

def process_chapter(chap, chapter_filename, audio_filename, prompt_file, audio_dir):
    with open(chapter_filename, 'r') as cf:
        try:
            data = json.load(cf)
        except:
            print('failed to read json %s' % chap)
            return 0
    words = data['words']
    transcript = data['transcript']
    sound = AudioSegment.from_mp3(audio_filename)

    pattern = re.compile(r'\.|; |--|!|\?')
    trans_iterator = pattern.finditer(transcript)

    # this whole thing is going to end up being linear in the length of transcript
    cur_words_index = 0

    count = 0

    audio_len = 0

    for step, t in enumerate(trans_iterator):
        first_word = words[cur_words_index]
        while True:
            # check if word index
            word = words[cur_words_index]
            if word['startOffset'] > t.span()[0]:
                last_word = words[cur_words_index - 1]
                break
            cur_words_index += 1
            if cur_words_index >= len(words): break

        start, end = first_word['startOffset'], last_word['endOffset']
        prompt = transcript[start:end+1].replace('\n', ' ').replace('_', '').replace('-', '').replace(';', '.').replace('\r', '')
        if '"' in prompt or '”' in prompt or '“' in prompt or '\'' in prompt or 'Mr' in prompt or 'Mrs' in prompt: continue

        if first_word['case'] != 'success' or \
           last_word['case']  != 'success': continue

        audio_start, audio_end = first_word['start'] * 1000, last_word['end']* 1000
        audio = sound[audio_start:audio_end]

        if len(prompt) < 80 and len(prompt) > 15:
            count += 1
            audio_len += len(audio)
            print('chap_%s_seg_%s' % (chap, count) + '||' + prompt, file=prompt_file)
            audio.export(audio_dir + '/chap_%s_seg_%s.wav' % (chap, count), format='wav')

    return audio_len / 1000

def process_all(main_dir, audio_ff):
    audio_len = 0
    with open(main_dir + 'prompt.data', 'w') as pf:
        for chap in tqdm(range(1, 39)):
            chap_str = str(chap) if chap > 9 else '0' + str(chap)
            audio_filename = main_dir + 'mp3/' + audio_ff % chap_str
            chapter_filename = main_dir + 'json/chp_%s.json' % chap
            audio_dir = main_dir + 'wavn'
            audio_len += process_chapter(chap, chapter_filename, audio_filename, pf, audio_dir)
    print('saved %.2f hours of audio' % (audio_len/3600))
        

parser = argparse.ArgumentParser()
parser.add_argument('main_dir')
parser.add_argument('audio_ff')
args = parser.parse_args()
process_all(args.main_dir, args.audio_ff)

