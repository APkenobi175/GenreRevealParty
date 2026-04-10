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


