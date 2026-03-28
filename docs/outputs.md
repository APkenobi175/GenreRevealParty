# Serial

```bash
[u6074058@kp297:serial]$ ./build/GenreRevealPartySerial 
File Read time (ms): 4399
Kmeans clustering time (ms): 1459
File write time (ms): 6084
```

# Parallel CUDA

```bash
[u6074058@kp297:parallel_cuda]$ ./build/GenreRevealPartyCUDA 
Reading CSV file...
/uufs/chpc.utah.edu/common/home/u6074058/GenreRevealParty/parallel_cuda
Working Directory: File Read time (ms): 8980
Starting KMeans Clustering on GPU...
Allocated GPU memory for points: 96322000 bytes
Allocated GPU memory for centroids: 400 bytes
Initialized centroids and copied to GPU
KMeans Clustering time (ms): 466
File write time (ms): 6206
```