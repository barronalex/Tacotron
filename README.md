# Tacotron

Implementation of [Tacotron](https://arxiv.org/abs/1703.10135), an end-to-end neural network for speech synthesis.


## Samples

The following playlist contains samples produced on unseen inputs by Tacotron trained for 250K steps on the Nancy Corpus with r=5. 

[Samples](https://soundcloud.com/alex-barron-440014733/sets/tacotron-samples-1)

As you can hear, output is pretty rough around the edges but you can make out words on new inputs and it should get better with more training, tuning and data.

Unsurprisingly, the biggest bottleneck to increased performance seems to be the quality and scale of the dataset. Google use an internal 25 hour dataset which is about 4 times the size of the Nancy corpus we use here so their results are more convincing. I have trained a model on ~20 hours of audiobook data from the 2013 Blizzard challenge, but inconsistent prosody and accents actually caused the outputs to sound much worse than those above. Any suggestions for a better dataset to try are welcomed.

## Requirements

[Tensorflow 1.2](https://www.tensorflow.org/versions/r1.2/install/)

[Librosa](https://github.com/librosa/librosa)

[tqdm](https://github.com/noamraph/tqdm)

## Data

For best results, use the [Nancy corpus](http://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/) from the 2011 Blizzard Challenge. The data is freely availiable for research use on the signing of a license. After obtaining a username and password, add them to the 'download_data.sh' script to fetch it automatically. 

We also download the considerably smaller [CMU ARCTIC](http://festvox.org/cmu_arctic/) dataset for testing which can be obtained without a license, but don't expect to get good results with it.

## Usage

First run the data fetching script (preferably after obtaining a username and password for the Nancy corpus)

	bash download_data.sh

Then preprocess the data

	python3 preprocess.py

Now we're ready to start training

	python3 train.py --train-set nancy 

Finally, create a text file containing the prompts you want to synthesize

	python3 test.py prompts.txt

To see the audio outputs created by Tacotron, open up Tensorboard.

On my GTX 1080, it takes about 5 hours to get to the point where synthesized speech on the training set is discernable and around 20 hours to obtain audible generalization at test time. Despite fairly agressive gradient clipping, the loss is prone to explosion. In that case try restarting from the most recent checkpoint (using the restore flag) with a slightly lowered learning rate. I'm working on improving this problem.


