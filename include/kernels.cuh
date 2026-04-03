#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "points.hpp"



__global__ void assignClustersGPU(Point *points, int point_count, Point *centroids, int cluster_count){
    // This kernal will assign each point to its nearest centroid, and store the assigned cluster in the point's cluster field.
    // Each thread will assign one point

    // 1. Calculate the thread ID using formula from class
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. If the thread ID is greater than the number of points, return, protect against out of bounds threads
    if (thread_id >= point_count) {
        return;
    }

    // 3. Calculate the distance from the point to each centroid, and assign the point to the nearest centroid's cluster. Store the assigned cluster in the point's cluster field.
    // __DBL_MAX__ is the maximum double value, use this to initialize the minimum distance before comparing to actual distances.

    Point *p = &points[thread_id];
    double minDistance = __DBL_MAX__;
    int bestCluster = 0;

    // 4. Find nearest centroid

    for(int c = 0; c < cluster_count; c++){
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

__global__ void accumulateCentroidsGPU(Point *points, int point_count, Point *centroid_temps, int *pointersPerCluster){
    // Each thread accumulates one point into its cluster's centroid sum

    // 1. Calculate the thread ID using formula from class
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count) {
        return;
    }

    // 2. Accumulate the point's features into the appropriate centroid sum

    int c = points[idx].cluster;
    atomicAdd(&pointersPerCluster[c], 1);
    atomicAdd(&centroid_temps[c].danceability, points[idx].danceability);
    atomicAdd(&centroid_temps[c].energy, points[idx].energy);
    atomicAdd(&centroid_temps[c].loudness, points[idx].loudness);
    atomicAdd(&centroid_temps[c].speechiness, points[idx].speechiness);
    atomicAdd(&centroid_temps[c].acousticness, points[idx].acousticness);
    atomicAdd(&centroid_temps[c].instrumentalness, points[idx].instrumentalness);
    atomicAdd(&centroid_temps[c].liveliness, points[idx].liveliness);
    atomicAdd(&centroid_temps[c].valence, points[idx].valence);

}

__global__ void computeCentroidsGPU(Point *centroids, Point *centroid_temps, int *pointsPerCluster, int cluster_count){

    // 1 Calculate the cluster ID using formula from class
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cluster_count) {
        return;
    }
    // 2. If the number of points in the cluster is greater than 0, divide the accumulated centroid sum by the number of points to get the new centroid position.
    // Store the new centroid position in the centroids array.
    if (pointsPerCluster[c] > 0){
        // this uses div_point from points.hpp
        centroid_temps[c].div_point(pointsPerCluster[c]);
    }

    // 3. Otherwise, if a cluster has no points assigned, leave it uncahgned

    centroids[c] = centroid_temps[c];
    centroids[c].cluster = c; // Set the cluster ID for the centroid
}


#endif

