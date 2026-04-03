# Serial

- Total Time: 12 seconds
- KMeans Clustering Time: 1500ms

```bash
[u6074058@kp297:serial]$ ./build/GenreRevealPartySerial 
File Read time (ms): 4428
Kmeans clustering time (ms): 1456
File write time (ms): 6131
```

# Parallel CUDA

- Total Time: 15.5 seconds
- KMeans Clustering Time: 298 ms

```bash
[u6074058@kp297:parallel_cuda]$ ./build/GenreRevealPartyCUDA 
Reading CSV file...
/uufs/chpc.utah.edu/common/home/u6074058/GenreRevealParty/parallel_cuda
Working Directory: File Read time (ms): 9031
Starting KMeans Clustering on GPU...
Allocated GPU memory for points: 96322000 bytes
Allocated GPU memory for centroids: 400 bytes
Allocated GPU memory for centroid temps: 400 bytes
Allocated GPU memory for points per cluster: 20 bytes
Initialized centroids and copied to GPU
KMeans Clustering time (ms): 298
File write time (ms): 6224
```





