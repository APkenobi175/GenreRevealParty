# Serial

```bash
[u6074058@notch001:serial]$ ./build/GenreRevealPartySerial 
File Read time (ms): 11322
Kmeans clustering time (ms): 1678
File write time (ms): 12045
[u6074058@notch001:serial]$ 
```



# Parallel CUDA

```bash
[u6074058@notch001:parallel_cuda]$ ./build/GenreRevealPartyCUDA 
Reading CSV file...
File Read time (ms): 18163
Starting KMeans Clustering on GPU...
Allocated GPU memory for points: 96322000 bytes
Allocated GPU memory for centroids: 400 bytes
Allocated GPU memory for centroid temps: 400 bytes
Allocated GPU memory for points per cluster: 20 bytes
Initialized centroids and copied to GPU
KMeans Clustering time (ms): 758
File write time (ms): 12383
[u6074058@notch001:parallel_cuda]$ 
```


### ALL OUTPUTS W 4 THREADS and 2 GPUS

```bash
RUNNING SERIAL IMPLEMENTATION
-- Configuring done (0.0s)
-- Generating done (0.1s)
-- Build files have been written to: /uufs/chpc.utah.edu/common/home/u6074058/GenreRevealParty/serial/build
[100%] Built target GenreRevealPartySerial
File Read time (ms): 11441
Kmeans clustering time (ms): 1667
File write time (ms): 12114
RUNNING OPEN MP IMPLEMENTATION
-- Configuring done (0.0s)
-- Generating done (0.1s)
-- Build files have been written to: /uufs/chpc.utah.edu/common/home/u6074058/GenreRevealParty/OpenMP/build
[100%] Built target GenreRevealPartyOpenMP
File Read time (ms): 8695
Kmeans clustering time (ms): 574
File write time (ms): 11784
RUNNING MPI IMPLEMENTATION with 4 THREADS
-- Configuring done (2.0s)
-- Generating done (0.1s)
-- Build files have been written to: /uufs/chpc.utah.edu/common/home/u6074058/GenreRevealParty/OpenMPI/build
[100%] Built target GenreRevealPartyMPI
File Read time (ms): 8681
Kmeans clustering time (ms): 439
File write time (ms): 11721
RUNNING CUDA IMPLEMENTATION
-- Configuring done (0.0s)
-- Generating done (0.1s)
-- Build files have been written to: /uufs/chpc.utah.edu/common/home/u6074058/GenreRevealParty/parallel_cuda/build
[100%] Built target GenreRevealPartyCUDA
Reading CSV file...
File Read time (ms): 17860
Starting KMeans Clustering on GPU...
Allocated GPU memory for points: 96322000 bytes
Allocated GPU memory for centroids: 400 bytes
Allocated GPU memory for centroid temps: 400 bytes
Allocated GPU memory for points per cluster: 20 bytes
Initialized centroids and copied to GPU
KMeans Clustering time (ms): 470
File write time (ms): 12196
RUNNING CUDA AND OPENMPI IMPLEMENTATION with 4 THREADS
-- Configuring done (1.4s)
-- Generating done (0.2s)
-- Build files have been written to: /uufs/chpc.utah.edu/common/home/u6074058/GenreRevealParty/cuda_MPI/build
[ 40%] Built target kernels
[100%] Built target mpi-cuda
File Read time (ms): 17820
Starting KMeans Clustering on GPU...
Starting KMeans Clustering on GPU...
Starting KMeans Clustering on GPU...
Starting KMeans Clustering on GPU...
Allocated GPU memory for points: 24080480 bytes
Allocated GPU memory for centroids: 400 bytes
Allocated GPU memory for centroid temps: 400 bytes
Allocated GPU memory for points per cluster: 20 bytes
Allocated GPU memory for points: 24080480 bytes
Allocated GPU memory for centroids: 400 bytes
Allocated GPU memory for centroid temps: 400 bytes
Allocated GPU memory for points per cluster: 20 bytes
Allocated GPU memory for points: 24080560 bytes
Allocated GPU memory for centroids: 400 bytes
Allocated GPU memory for centroid temps: 400 bytes
Allocated GPU memory for points per cluster: 20 bytes
Allocated GPU memory for points: 24080480 bytes
Allocated GPU memory for centroids: 400 bytes
Allocated GPU memory for centroid temps: 400 bytes
Allocated GPU memory for points per cluster: 20 bytes
Initialized centroids and copied to GPU
Initialized centroids and copied to GPU
Initialized centroids and copied to GPU
Initialized centroids and copied to GPU
Kmeans clustering time (ms): 950
File write time (ms): 12236
ALL IMPLEMENTATIONS COMPLETE
```


### ALL OUTPUTS W 4 THREADS and 4 GPUS

