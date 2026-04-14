#include "../include/kernels.cuh"
#include "../include/points.hpp"
#include <cstdio>

void accumulateCentroids(Point *centroids, Point *centroid_temps,
                         int cluster_count, int *points_per_cluster,
                         int total_points, Point *points);

void accumulatingStep(Point *points, int point_count, int rank,
                      Point *centroids, Point *centroid_temps,
                      int cluster_count, int *points_per_cluster,
                      bool shouldCopyPoints) {

  cudaSetDevice(rank);
  printf("Starting KMeans Clustering on GPU...\n");

  // 1. Allocate GPU memory for points and centroids
  Point *d_points, *d_centroids,
      *d_centroid_temps;     // NEW, added d_centroid_temps for accumulating
                             // centroid sums on GPU
  int *d_points_per_cluster; // NEW, added d_points_per_cluster for counting
                             // points
  cudaMalloc(&d_points, point_count * sizeof(Point));
  cudaMalloc(&d_centroids, cluster_count * sizeof(Point));
  cudaMalloc(&d_centroid_temps, cluster_count * sizeof(Point));
  cudaMalloc(&d_points_per_cluster, cluster_count * sizeof(int));

  // 2. Copy points from host to device
  cudaMemcpy(d_points, points, point_count * sizeof(Point),
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_centroids, centroids, cluster_count * sizeof(Point),
             cudaMemcpyHostToDevice);
  printf("Initialized centroids and copied to GPU\n");

  accumulateCentroids(d_centroids, d_centroid_temps, cluster_count,
                      d_points_per_cluster, point_count, d_points);

  cudaMemcpy(centroid_temps, d_centroid_temps, cluster_count * sizeof(Point),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(points_per_cluster, d_points_per_cluster,
             cluster_count * sizeof(int), cudaMemcpyDeviceToHost);

  if (shouldCopyPoints)
    cudaMemcpy(points, d_points, point_count * sizeof(Point),
               cudaMemcpyDeviceToHost);

  // 6. Free GPU memory
  cudaFree(d_points);
  cudaFree(d_centroids);
  cudaFree(d_centroid_temps);
  cudaFree(d_points_per_cluster);
}

void accumulateCentroids(Point *d_centroids, Point *d_centroid_temps,
                         int cluster_count, int *d_points_per_cluster,
                         int total_points, Point *d_points) {
  int threads_per_block = 1024; // Change this maybe? or actually we can use it
                                // as a parameter in main
  int blocks = (total_points + threads_per_block - 1) /
               threads_per_block; // Calculate number of blocks needed to cover
                                  // all points

  // NEW - initialize temp array and points per cluster to 0 on GPU before
  // accumulation
  cudaMemset(d_centroid_temps, 0, cluster_count * sizeof(Point));
  cudaMemset(d_points_per_cluster, 0, cluster_count * sizeof(int));

  // 1. Launch kernal to assign clusters
  assignClustersGPU<<<blocks, threads_per_block>>>(d_points, total_points,
                                                   d_centroids, cluster_count);
  cudaDeviceSynchronize();

  // 2. NEW - Launch kernel to accumulate centroids
  accumulateCentroidsGPU<<<blocks, threads_per_block>>>(
      d_points, total_points, d_centroid_temps, d_points_per_cluster);
  cudaDeviceSynchronize();
}
