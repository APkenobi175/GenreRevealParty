#!/bin/bash
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=kmeans_all
#SBATCH --output=results.log


module load cuda
module load python
module load openmp
module load openmpi


echo 'RUNNING SERIAL IMPLEMENTATION'

cd ~/GenreRevealParty/serial
mkdir -p build && cd build
cmake ..
make
cd ..
./build/GenreRevealPartySerial
cd ..

echo 'RUNNING OPEN MP IMPLEMENTATION'
cd ~/GenreRevealParty/openMP
mkdir -p build && cd build
cmake ..
make
cd ..
./build/GenreRevealPartyOpenMP
cd ..

echo 'RUNNING MPI IMPLEMENTATION'
cd ~/GenreRevealParty/OpenMPI
mkdir -p build && cd build
cmake ..
make
cd ..
mpirun -np 4 ./build/GenreRevealPartyMPI
cd ..

echo 'RUNNING CUDA IMPLEMENTATION'
cd ~/GenreRevealParty/parallel_cuda
mkdir -p build && cd build
cmake ..
make
cd ..
./build/GenreRevealPartyCUDA
cd ..

echo 'RUNNING CUDA WITH OPENMP IMPLEMENTATION'
cd ~/GenreRevealParty/cuda_MPI
mkdir -p build && cd build
cmake ..
make
cd ..
./build/GenreRevealPartyCudaMPI
cd ..


echo 'ALL IMPLEMENTATIONS COMPLETE'
echo 'VALIDATING RESULTS'

python validation.py

echo 'Creating Visualizations'

cd ~/GenreRevealParty/visualizer

python visualize.py

