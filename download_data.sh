#!/bin/bash
#wget http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz
#tar -xvzf VCTK-Corpus.tar.gz
#rm VCTK-Corpus.tar.gz

wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.zip
unzip cmu_us_slt_arctic-0.95-release.zip
rm cmu_us_slt_arctic-0.95-release.zip
mv cmu_us_slt_arctic data

#wget http://www.openslr.org/resources/12/dev-clean.tar.gz
#wget http://www.openslr.org/resources/12/test-clean.tar.gz
#wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
#tar -xvzf dev-clean.tar.gz
#tar -xvzf test-clean.tar.gz
#tar -xvzf train-clean-100.tar.gz
#rm dev-clean.tar.gz
#rm test-clean.tar.gz
#rm train-clean-100.tar.gz
#mkdir data
#mv LibriSpeech data
