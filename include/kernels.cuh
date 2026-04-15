#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "points.hpp"

__device__ inline double atomicAddDouble(double *address, double val) {
#if __CUDA_ARCH__ >= 600
  // ✔ Pascal (6.0+) and newer: hardware atomic
  return atomicAdd(address, val);

#else
  // ✔ Maxwell (5.2) fallback using CAS
  unsigned long long int *address_as_ull = (unsigned long long int *)address;

  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

  } while (assumed != old);

  return __longlong_as_double(old);

#endif
}

__global__ void assignClustersGPU(Point *points, int point_count,
                                  Point *centroids, int cluster_count) {
  // This kernal will assign each point to its nearest centroid, and store the
  // assigned cluster in the point's cluster field. Each thread will assign one
  // point

  // 1. Calculate the thread ID using formula from class
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  // 2. If the thread ID is greater than the number of points, return, protect
  // against out of bounds threads
  if (thread_id >= point_count) {
    return;
  }

  // 3. Calculate the distance from the point to each centroid, and assign the
  // point to the nearest centroid's cluster. Store the assigned cluster in the
  // point's cluster field.
  // __DBL_MAX__ is the maximum double value, use this to initialize the minimum
  // distance before comparing to actual distances.

  Point *p = &points[thread_id];
  double minDistance = __DBL_MAX__;
  int bestCluster = 0;

  // 4. Find nearest centroid

  for (int c = 0; c < cluster_count; c++) {
    // this uses distance_point from points.hpp
    double distance = p->distance_point(&centroids[c]);
    if (distance < minDistance) {
      minDistance = distance;
      bestCluster = c;
    }
  }

  // 5. Assign the point to the nearest centroid's cluster
  p->cluster = bestCluster;
}

__global__ void accumulateCentroidsGPU(Point *points, int point_count,
                                       Point *centroid_temps,
                                       int *pointersPerCluster) {
  // Each thread accumulates one point into its cluster's centroid sum

  // 1. Calculate the thread ID using formula from class
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= point_count) {
    return;
  }

  // 2. Accumulate the point's features into the appropriate centroid sum

  int c = points[idx].cluster;
  atomicAdd(&pointersPerCluster[c], 1);
  atomicAddDouble(&centroid_temps[c].danceability, points[idx].danceability);
  atomicAddDouble(&centroid_temps[c].energy, points[idx].energy);
  atomicAddDouble(&centroid_temps[c].speechiness, points[idx].speechiness);
  atomicAddDouble(&centroid_temps[c].acousticness, points[idx].acousticness);
  atomicAddDouble(&centroid_temps[c].instrumentalness,
                  points[idx].instrumentalness);
  atomicAddDouble(&centroid_temps[c].liveliness, points[idx].liveliness);
  atomicAddDouble(&centroid_temps[c].valence, points[idx].valence);
}

__global__ void accumulateBlockCentroids(Point *points, int point_count,
                                         Point *block_centroids,
                                         int *block_counts, int K) {
  extern __shared__ unsigned char smem[];

  Point *s_centroids = (Point *)smem;
  int *s_counts = (int *)&s_centroids[K];

  // 1. Initialize shared memory
  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    s_counts[i] = 0;

    s_centroids[i].danceability = 0.0;
    s_centroids[i].energy = 0.0;
    s_centroids[i].speechiness = 0.0;
    s_centroids[i].acousticness = 0.0;
    s_centroids[i].instrumentalness = 0.0;
    s_centroids[i].liveliness = 0.0;
    s_centroids[i].valence = 0.0;
  }
  __syncthreads();

  // 2. Process points
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < point_count) {
    int c = points[idx].cluster;

    // Shared memory atomics (fast, low contention)
    atomicAdd(&s_counts[c], 1);
    atomicAddDouble(&s_centroids[c].danceability, points[idx].danceability);
    atomicAddDouble(&s_centroids[c].energy, points[idx].energy);
    atomicAddDouble(&s_centroids[c].speechiness, points[idx].speechiness);
    atomicAddDouble(&s_centroids[c].acousticness, points[idx].acousticness);
    atomicAddDouble(&s_centroids[c].instrumentalness,
                    points[idx].instrumentalness);
    atomicAddDouble(&s_centroids[c].liveliness, points[idx].liveliness);
    atomicAddDouble(&s_centroids[c].valence, points[idx].valence);
  }

  __syncthreads();

  // 3. Write ONE result per block (no atomics!)
  int base = blockIdx.x * K;

  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    block_counts[base + i] = s_counts[i];
    block_centroids[base + i] = s_centroids[i];
  }
}

__global__ void reduceCentroids(Point *block_centroids, int *block_counts,
                                Point *centroid_temps, int *pointersPerCluster,
                                int num_blocks, int K) {

  int c = threadIdx.x + blockIdx.x * blockDim.x;
  if (c >= K)
    return;

  Point sum = {};
  sum.cluster = c;
  int count = 0;

  for (int b = 0; b < num_blocks; b++) {
    int idx = b * K + c;

    count += block_counts[idx];
    sum.danceability += block_centroids[idx].danceability;
    sum.energy += block_centroids[idx].energy;
    sum.speechiness += block_centroids[idx].speechiness;
    sum.acousticness += block_centroids[idx].acousticness;
    sum.instrumentalness += block_centroids[idx].instrumentalness;
    sum.liveliness += block_centroids[idx].liveliness;
    sum.valence += block_centroids[idx].valence;
  }

  pointersPerCluster[c] = count;
  centroid_temps[c] = sum;
}

__global__ void computeCentroidsGPU(Point *centroids, Point *centroid_temps,
                                    int *pointsPerCluster, int cluster_count) {

  // 1 Calculate the cluster ID using formula from class
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cluster_count) {
    return;
  }
  // 2. If the number of points in the cluster is greater than 0, divide the
  // accumulated centroid sum by the number of points to get the new centroid
  // position. Store the new centroid position in the centroids array.
  if (pointsPerCluster[c] > 0) {
    // this uses div_point from points.hpp
    centroid_temps[c].div_point(pointsPerCluster[c]);
  }

  // 3. Otherwise, if a cluster has no points assigned, leave it uncahgned

  centroids[c] = centroid_temps[c];
  centroids[c].cluster = c; // Set the cluster ID for the centroid
}

#endif
