# Tacotron

Implementation of [Tacotron](https://arxiv.org/abs/1703.10135), an end-to-end neural network for speech synthesis.

## Samples

The following playlist contain samples produced on unseen inputs by Tacotron trained for 180K steps on the Nancy Corpus with r=2 and scheduled sampling 0.5. 

[Samples r=2](https://soundcloud.com/alex-barron-440014733/sets/tacotron-samples-r2)

You can try the synthesizer for yourself by running 'download_weights.sh' and then 'test.py' as described below.

When compared to the [old samples](https://soundcloud.com/alex-barron-440014733/sets/tacotron-samples-1), the alignment learned with r=2 is considerably better but the audio quality is noticeably rougher.
I assume this partially a result of too little training (the original paper trained for at least 20 times longer) but I think it is also related to the scheduled sampling that was necessary to learn the alignment. I also updated the padding which fixed the repetition and corruption at the end of the samples.

## Requirements

[Tensorflow 1.2](https://www.tensorflow.org/versions/r1.2/install/)

[Librosa](https://github.com/librosa/librosa)

[tqdm](https://github.com/noamraph/tqdm)

[matplotlib](https://matplotlib.org/)

## Data

For best results, use the [Nancy corpus](http://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/) from the 2011 Blizzard Challenge. The data is freely availiable for research use on the signing of a license. After obtaining a username and password, add them to the 'download_data.sh' script to fetch it automatically. 

We also download the considerably smaller [CMU ARCTIC](http://festvox.org/cmu_arctic/) dataset for testing which can be obtained without a license, but don't expect to get good results with it.

You can add new datasets in 'preprocess.py' by writing a 'prepare' function which produces a list of prompts and corresponding list of wav filenames. This should be clear from the examples in 'preprocess.py'.

## Usage

To synthesize audio:

First fetch the weights using the script provided

	bash download_weights.sh

Then pass prompts (separated by end lines) to 'test.py' through stdin

	python3 test.py < prompts.txt
	
	echo "This is a test prompt for the system to say." | python3 test.py

To train the model:

First run the data fetching script (preferably after obtaining a username and password for the Nancy corpus)

	bash download_data.sh

Then preprocess the data

	python3 preprocess.py arctic
	python3 preprocess.py nancy 

 Now we're ready to start training

	python3 train.py --train-set nancy (--restore optional)

To see the audio outputs created by Tacotron, open up Tensorboard.

Monitoring the attention alignments produced under the images tab in Tensorboard is by far the best way to debug your model while its training. You'll likely only see generalization to new examples if/when the attention becomes monotonic. The gif below shows the model learning an alignment using the default parameters on the Nancy dataset.

![Attention Alignments](https://github.com/barronalex/Tacotron/raw/master/images/attention.gif)

## Updates

I've begun to implement the multi-speaker tacotron architecture suggested by the [Deep Voice 2](https://arxiv.org/pdf/1705.08947.pdf) paper, but its currently untested. 'preprocess.py' has the VCTK corpus implemented but you need to download the data. Given the scale of this dataset (40 hours), I assume we'll get better results if we can get it to work.

The memory usage has been significantly reduced. An 8 cpu instance on a cloud service should run the code with standard RAM. My macbook with 16GB of RAM also runs it fine (if incredibly slowly).
The reason it's so high is because empirically I found that there was around a 2X speed increase when reading the data from memory instead of disk.

With a K80 and r=2, we process 1 batch every ~2.5 seconds.
With a GTX1080 and r=2, we process 1 batch every ~1.5 seconds. 

Particularly if using a smaller dataset and no scheduled sampling, you're likely to see very different audio quality at training and test time, even on training examples.
This is a result of how this seq2seq model is trained -- in training, the ground truth is provided at each time step in the decoder but in testing we must use the previous time step as input. This will be noisey and thus result in poorer quality future outputs. Scheduled sampling addresses this.

