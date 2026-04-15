#include "../include/kernels.cuh"
#include "../include/points.hpp"
#include <cassert>
#include <cstdio>

__global__ void initPoints(Point *arr, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    arr[i] = initialize_point(0, 0, 0, 0, 0, 0, 0);
  }
}

__host__ __device__ cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

void accumulateCentroids(Point *d_centroids, Point *d_centroid_temps,
                         int cluster_count, int *d_points_per_cluster,
                         int total_points, Point *d_points) {
  int threads_per_block = 1024; // Change this maybe? or actually we can use it
                                // as a parameter in main
  int blocks = (total_points + threads_per_block - 1) /
               threads_per_block; // Calculate number of blocks needed to cover
                                  // all points

  // 1. Launch kernal to assign clusters
  assignClustersGPU<<<blocks, threads_per_block>>>(d_points, total_points,
                                                   d_centroids, cluster_count);
  checkCuda(cudaDeviceSynchronize());
  int shared_size = cluster_count * sizeof(Point) + cluster_count * sizeof(int);

#if __CUDA_ARCH__ >= 600
  accumulateCentroidsGPU<<<blocks, threads_per_block>>>(
      d_points, total_points, d_centroid_temps, d_points_per_cluster);

#else
  Point *d_block_centroids;
  int *d_block_counts;

  cudaMalloc(&d_block_centroids, blocks * cluster_count * sizeof(Point));
  cudaMalloc(&d_block_counts, blocks * cluster_count * sizeof(int));

  accumulateBlockCentroids<<<blocks, threads_per_block, shared_size>>>(
      d_points, total_points, d_block_centroids, d_block_counts, cluster_count);

  checkCuda(cudaDeviceSynchronize());
  // WARNING: Hardcoded Grid Dimension
  reduceCentroids<<<1, cluster_count>>>(d_block_centroids, d_block_counts,
                                        d_centroid_temps, d_points_per_cluster,
                                        blocks, cluster_count);
#endif

  checkCuda(cudaDeviceSynchronize());
}

void setDevice(int rank) {
  int device_count;
  cudaGetDeviceCount(&device_count);
  checkCuda(cudaSetDevice(rank % device_count));
}

void accumulatingStep(Point *points, int point_count, int rank,
                      Point *centroids, Point *centroid_temps,
                      int cluster_count, int *points_per_cluster,
                      bool shouldCopyPoints) {

  Point *d_points, *d_centroids,
      *d_centroid_temps;     // NEW, added d_centroid_temps for accumulating
                             // centroid sums on GPU
  int *d_points_per_cluster; // NEW, added d_points_per_cluster for counting
                             // points
  checkCuda(cudaMalloc(&d_points, point_count * sizeof(Point)));
  checkCuda(cudaMalloc(&d_centroids, cluster_count * sizeof(Point)));
  checkCuda(cudaMalloc(&d_centroid_temps, cluster_count * sizeof(Point)));
  checkCuda(cudaMalloc(&d_points_per_cluster, cluster_count * sizeof(int)));

  // 2. Copy points from host to device
  checkCuda(cudaMemcpy(d_points, points, point_count * sizeof(Point),
                       cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_centroids, centroids, cluster_count * sizeof(Point),
                       cudaMemcpyHostToDevice));
  checkCuda(cudaMemset(d_centroid_temps, 0, cluster_count * sizeof(Point)));
  checkCuda(cudaMemset(d_points_per_cluster, 0, cluster_count * sizeof(int)));

  accumulateCentroids(d_centroids, d_centroid_temps, cluster_count,
                      d_points_per_cluster, point_count, d_points);

  checkCuda(cudaMemcpy(centroid_temps, d_centroid_temps,
                       cluster_count * sizeof(Point), cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(points_per_cluster, d_points_per_cluster,
                       cluster_count * sizeof(int), cudaMemcpyDeviceToHost));

#ifdef DEBUG_INFO
  if (rank == 0) {
    for (int i = 0; i < cluster_count; i++) {
      Point centroid = centroid_temps[i];
      printf("Centroid: %f %f %f %f %f %f %d\n", centroid.acousticness,
             centroid.danceability, centroid.energy, centroid.instrumentalness,
             centroid.liveliness, centroid.valence, centroid.cluster);
    }
    printf("Cluster counts\n");
    for (int i = 0; i < cluster_count; i++)
      printf("Points: %d\n", points_per_cluster[i]);
  }
#endif

  checkCuda(cudaMemcpy(points, d_points, point_count * sizeof(Point),
                       cudaMemcpyDeviceToHost));

  // 6. Free GPU memory
  checkCuda(cudaFree(d_points));
  checkCuda(cudaFree(d_centroids));
  checkCuda(cudaFree(d_centroid_temps));
  checkCuda(cudaFree(d_points_per_cluster));
}
