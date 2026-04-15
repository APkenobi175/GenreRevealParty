#!/bin/bash
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=runall
#SBATCH --output=results.log
#SBATCH --ntasks=1

module load cuda
module load python
module load openmpi

echo "GPU INFORMATION"

nvidia-smi

echo ""
echo ""

echo 'RUNNING SERIAL IMPLEMENTATION'
echo ""

cd ~/GenreRevealParty/serial
mkdir -p build && cd build
cmake ..
make
cd ..
./build/GenreRevealPartySerial
mv output.csv serial_output.csv
cd ..

# echo 'RUNNING OPEN MP IMPLEMENTATION'
# cd ~/GenreRevealParty/OpenMP
# mkdir -p build && cd build
# cmake ..
# make
# cd ..
# ./build/GenreRevealPartyOpenMP
# mv output.csv openmp_output.csv
# cd ..
# echo ""
# echo 'RUNNING MPI IMPLEMENTATION with 4 THREADS'
# echo ""
# cd ~/GenreRevealParty/OpenMPI
# mkdir -p build && cd build
# cmake ..
# make
# cd ..
# mpirun -np 4 ./build/GenreRevealPartyMPI
# mv output.csv mpi_output.csv
# cd ..
# echo ""
# echo 'RUNNING CUDA IMPLEMENTATION'
# echo ""
# cd ~/GenreRevealParty/parallel_cuda
# mkdir -p build && cd build
# cmake ..
# make
# cd ..
# ./build/GenreRevealPartyCUDA
# mv output.csv cuda_output.csv

# cd ..
# echo ""
# echo 'RUNNING CUDA AND OPENMPI IMPLEMENTATION with 4 THREADS'
# echo ""
# cd ~/GenreRevealParty/cuda_MPI
# mkdir -p build && cd build
# cmake ..
# make
# cd ..
# mpirun -np 4 ./build/mpi-cuda
# mv output.csv mpi_cuda_output.csv
# cd ..

# echo 'ALL IMPLEMENTATIONS COMPLETE'
