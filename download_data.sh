#!/bin/bash
mkdir data
mkdir weights

# arctic
wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.zip
unzip cmu_us_slt_arctic-0.95-release.zip
rm cmu_us_slt_arctic-0.95-release.zip
mv cmu_us_slt_arctic data/arctic

# nancy
mkdir data/nancy

wget --user admb@stanford.edu --password " 92fhpqweuihas" http://data.cstr.ed.ac.uk/blizzard2011/lessac/wavn.tgz
wget --user admb@stanford.edu --password " 92fhpqweuihas" http://data.cstr.ed.ac.uk/blizzard2011/lessac/prompts.data

tar -xvzf wavn.tgz
rm wavn.tgz
mv wavn data/nancy
mv prompts.data data/nancy
