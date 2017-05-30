import re
import argparse
import codecs

def segment_chapters(filename, out_dir):
    with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

        chapter_indices = [m.start() for m in re.finditer('CHAPTER', text)]
        chapters = []
        for i in range(len(chapter_indices)-1):
            chapters.append(text[chapter_indices[i]:chapter_indices[i+1]])
        chapters.append(text[chapter_indices[-1]:])

        # write chapters for segmentation
        for i, chap in enumerate(chapters):
            with open(out_dir + '/chp_%s.txt' % (i+1), 'w') as f:
                f.write(chap)

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('out_dir')
args = parser.parse_args()

segment_chapters(args.filename, args.out_dir)



