#!/bin/bash
NANCY_USERNAME=""
NANCY_PASSWORD=""

mkdir data
mkdir weights

# arctic
wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.zip
unzip cmu_us_slt_arctic-0.95-release.zip
rm cmu_us_slt_arctic-0.95-release.zip
mv cmu_us_slt_arctic data/arctic

# nancy
if [ -n "$NANCY_PASSWORD" ]
then
	mkdir data/nancy
	wget --user "$NANCY_USERNAME" --password "$NANCY_PASSWORD" http://data.cstr.ed.ac.uk/blizzard2011/lessac/wavn.tgz
	wget --user "$NANCY_USERNAME" --password "$NANCY_PASSWORD" http://data.cstr.ed.ac.uk/blizzard2011/lessac/prompts.data

	tar -xvzf wavn.tgz
	rm wavn.tgz
	mv wavn data/nancy
	mv prompts.data data/nancy
else
	echo "For best results obtain a username and password for the Nancy corpus from http://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/"
fi
