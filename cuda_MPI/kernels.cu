#include "../include/kernels.cuh"
#include "../include/kmeans.hpp"
#include "../include/points.hpp"
#include <cstdio>
#include <vector>

void kMeansClustering(Point *points, int point_count, int rank,
                      int cluster_count, int iterations) {

  cudaSetDevice(rank);
  printf("Starting KMeans Clustering on GPU...\n");

  // 1. Allocate GPU memory for points and centroids
  Point *d_points, *d_centroids,
      *d_centroid_temps;     // NEW, added d_centroid_temps for accumulating
                             // centroid sums on GPU
  int *d_points_per_cluster; // NEW, added d_points_per_cluster for counting
                             // points
  cudaMalloc(&d_points, point_count * sizeof(Point));
  printf("Allocated GPU memory for points: %d bytes\n",
         point_count * sizeof(Point));
  cudaMalloc(&d_centroids, cluster_count * sizeof(Point));
  printf("Allocated GPU memory for centroids: %d bytes\n",
         cluster_count * sizeof(Point));
  cudaMalloc(&d_centroid_temps, cluster_count * sizeof(Point));
  printf("Allocated GPU memory for centroid temps: %d bytes\n",
         cluster_count * sizeof(Point));
  cudaMalloc(&d_points_per_cluster, cluster_count * sizeof(int));
  printf("Allocated GPU memory for points per cluster: %d bytes\n",
         cluster_count * sizeof(int));

  // 2. Copy points from host to device
  cudaMemcpy(d_points, points, point_count * sizeof(Point),
             cudaMemcpyHostToDevice);

  // 3. Initialize centroids on CPU then copy to GPU (MUST BE DONE ON CPU DUE TO
  // RANDOM NUMBER GENERATION)
  Point centroids[cluster_count];
  std::vector<Point> points_vec(
      points, points + point_count); // Create a vector from the points array
                                     // for the shared code function
  initializeCentroids(points_vec, centroids, cluster_count);
  cudaMemcpy(d_centroids, centroids, cluster_count * sizeof(Point),
             cudaMemcpyHostToDevice);
  printf("Initialized centroids and copied to GPU\n");

  // 4. For KMEANS_ITERATIONS iterations, assign clusters and update centroids

  int threads_per_block = 1024; // Change this maybe? or actually we can use it
                                // as a parameter in main
  int blocks = (point_count + threads_per_block - 1) /
               threads_per_block; // Calculate number of blocks needed to cover
                                  // all points

  for (int i = 0; i < iterations; i++) {

    // NEW - initialize temp array and points per cluster to 0 on GPU before
    // accumulation
    cudaMemset(d_centroid_temps, 0, cluster_count * sizeof(Point));
    cudaMemset(d_points_per_cluster, 0, cluster_count * sizeof(int));

    // 1. Launch kernal to assign clusters
    assignClustersGPU<<<blocks, threads_per_block>>>(
        d_points, point_count, d_centroids, cluster_count);
    // 2. Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // 3. NEW - Launch kernel to accumulate centroids

    accumulateCentroidsGPU<<<blocks, threads_per_block>>>(
        d_points, point_count, d_centroid_temps, d_points_per_cluster);
    cudaDeviceSynchronize();

    // 4. NEW - Compute new centroids on GPU

    computeCentroidsGPU<<<1, cluster_count>>>(
        d_centroids, d_centroid_temps, d_points_per_cluster, cluster_count);
    cudaDeviceSynchronize();
  }

  // 5. Copy points back to CPU to update centroids
  cudaMemcpy(points, d_points, point_count * sizeof(Point),
             cudaMemcpyDeviceToHost);

  // 6. Free GPU memory
  cudaFree(d_points);
  cudaFree(d_centroids);
  cudaFree(d_centroid_temps);
  cudaFree(d_points_per_cluster);
}
