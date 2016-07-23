projroot=~/Code/Research/disease-time-series
cwd=$(pwd)

cd $projroot
rm src/*.pyc
rm src/tools/*.pyc
rm hmm/*.pyc
rm hmm/eval/*.pyc
rm test/*.pyc

cd $cwd
