// 1. Include repeated code

#include "../include/fileops.hpp"
#include "../include/points.hpp"
#include "../include/kmeans.hpp"

#include <iostream>
#include <ctime>
#include <vector>
#include <cmath>
#include <array>
#include <cuda_runtime.h>
#include <chrono>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __global__
#endif


constexpr int CLUSTER_COUNT = 5;
constexpr int KMEANS_ITERATIONS = 5;

// TODO: CUDA KERNALS GO HERE

__global__ void assignClusters(Point *points, int point_count, Point *centroids, int cluster_count){
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
        double distance = p->distance_point(&centroids[c]);
        if (distance < minDistance) {
            minDistance = distance;
            bestCluster = c;
        }
    }

    // 5. Assign the point to the nearest centroid's cluster
    p->cluster = bestCluster;



}


void kMeansClustering(Point *points, int point_count) {
    printf("Starting KMeans Clustering on GPU...\n");

    // 1. Allocate GPU memory for points and centroids
    Point *d_points, *d_centroids;
    cudaMalloc(&d_points, point_count * sizeof(Point));
    printf("Allocated GPU memory for points: %d bytes\n", point_count * sizeof(Point));
    cudaMalloc(&d_centroids, CLUSTER_COUNT * sizeof(Point));
    printf("Allocated GPU memory for centroids: %d bytes\n", CLUSTER_COUNT * sizeof(Point));

    // 2. Copy points from host to device
    cudaMemcpy(d_points, points, point_count * sizeof(Point), cudaMemcpyHostToDevice);

    // 3. Initialize centroids then copy to GPU
    Point centroids[CLUSTER_COUNT];
    std::vector<Point> points_vec(points, points + point_count); // Create a vector from the points array for the shared code function
    initializeCentroids(points_vec, centroids, CLUSTER_COUNT);
    cudaMemcpy(d_centroids, centroids, CLUSTER_COUNT * sizeof(Point), cudaMemcpyHostToDevice);
    printf("Initialized centroids and copied to GPU\n");

    // 4. For KMEANS_ITERATIONS iterations, assign clusters and update centroids

    int threads_per_block = 1024; // Change this maybe? or actually we can use it as a parameter in main
    int blocks = (point_count + threads_per_block - 1) / threads_per_block; // Calculate number of blocks needed to cover all points

    for (int i = 0; i< KMEANS_ITERATIONS; i++) {
        // 1. Launch kernal to assign clusters
        assignClusters<<<blocks, threads_per_block>>>(d_points, point_count, d_centroids, CLUSTER_COUNT);
        // 2. Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        // 3. Copy points back to CPU to update centroids
        cudaMemcpy(points, d_points, point_count * sizeof(Point), cudaMemcpyDeviceToHost);
        Point centroid_temps[CLUSTER_COUNT] = {};
        int points_per_cluster[CLUSTER_COUNT] = {};
        // 4. Iteratte over points on CPU to calculate new centroids
        for (int p = 0; p< point_count; p++){
            int c = points[p].cluster;
            if (c < 0 || c >= CLUSTER_COUNT) {
                std::cerr << "Error: Point assigned to invalid cluster " << c << " " << p << std::endl;
                return;
            }
            centroid_temps[c].add_point(&points[p]);
            points_per_cluster[c]++;
        }
        // 5. iterate over centroids to divide by number of points in each cluster to get mean
        for (int c = 0; c < CLUSTER_COUNT; c++){
            if (points_per_cluster[c] > 0) {
                centroid_temps[c].div_point(points_per_cluster[c]);
            }
                // If a cluster has no points assigned, we can choose to reinitialize it randomly or leave it unchanged. Here we will leave it unchanged.
            centroids[c] = centroid_temps[c];
            centroids[c].cluster = c; 
        }

        // 6. Copy new centroids to GPU for next iteration
        cudaMemcpy(d_centroids, centroids, CLUSTER_COUNT * sizeof(Point), cudaMemcpyHostToDevice);
    }

    // 6. Free GPU memory
    cudaFree(d_points);
    cudaFree(d_centroids);

}



int main(){
    // 1. Read CSV file and convert to plain C array
    auto start_file_time = std::chrono::high_resolution_clock::now();
    printf("Reading CSV file...\n");
    std::vector<Point> temp = readcsv();
    printf("Working Directory: ");
    system("pwd");
    Point *points = temp.data();
    int point_count = temp.size();
    auto end_file_time = std::chrono::high_resolution_clock::now();
    auto file_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_file_time - start_file_time);
    printf("File Read time (ms): %ld\n", file_duration.count());

    // 2. Run K means clustering

    auto start_kmeans_time = std::chrono::high_resolution_clock::now();
    kMeansClustering(points, point_count);
    auto end_kmeans_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_kmeans_time - start_kmeans_time);

    printf("KMeans Clustering time (ms): %ld\n", duration.count());
    // 3. Write output to csv file

    auto start_write_time = std::chrono::high_resolution_clock::now();
    writecsv(points, point_count);
    auto end_write_time = std::chrono::high_resolution_clock::now();
    auto write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_write_time - start_write_time);

    printf("File write time (ms): %ld\n", write_duration.count());
}