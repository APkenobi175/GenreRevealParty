// 1. Include repeated code

#include "../include/fileops.hpp"
#include "../include/points.hpp"
#include "../include/kmeans.hpp"
#include "../include/kernels.cuh" // NEW - include the kernels

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






void kMeansClustering(Point *points, int point_count) {
    printf("Starting KMeans Clustering on GPU...\n");

    // 1. Allocate GPU memory for points and centroids
    Point *d_points, *d_centroids, *d_centroid_temps; // NEW, added d_centroid_temps for accumulating centroid sums on GPU
    int *d_points_per_cluster; // NEW, added d_points_per_cluster for counting points
    cudaMalloc(&d_points, point_count * sizeof(Point));
    printf("Allocated GPU memory for points: %d bytes\n", point_count * sizeof(Point));
    cudaMalloc(&d_centroids, CLUSTER_COUNT * sizeof(Point));
    printf("Allocated GPU memory for centroids: %d bytes\n", CLUSTER_COUNT * sizeof(Point));
    cudaMalloc(&d_centroid_temps, CLUSTER_COUNT * sizeof(Point));
    printf("Allocated GPU memory for centroid temps: %d bytes\n", CLUSTER_COUNT * sizeof(Point));
    cudaMalloc(&d_points_per_cluster, CLUSTER_COUNT * sizeof(int));
    printf("Allocated GPU memory for points per cluster: %d bytes\n", CLUSTER_COUNT * sizeof(int));

    // 2. Copy points from host to device
    cudaMemcpy(d_points, points, point_count * sizeof(Point), cudaMemcpyHostToDevice);

    // 3. Initialize centroids on CPU then copy to GPU (MUST BE DONE ON CPU DUE TO RANDOM NUMBER GENERATION)
    Point centroids[CLUSTER_COUNT];
    std::vector<Point> points_vec(points, points + point_count); // Create a vector from the points array for the shared code function
    initializeCentroids(points_vec, centroids, CLUSTER_COUNT);
    cudaMemcpy(d_centroids, centroids, CLUSTER_COUNT * sizeof(Point), cudaMemcpyHostToDevice);
    printf("Initialized centroids and copied to GPU\n");

    // 4. For KMEANS_ITERATIONS iterations, assign clusters and update centroids

    int threads_per_block = 1024; // Change this maybe? or actually we can use it as a parameter in main
    int blocks = (point_count + threads_per_block - 1) / threads_per_block; // Calculate number of blocks needed to cover all points

    for (int i = 0; i< KMEANS_ITERATIONS; i++) {

        // NEW - initialize temp array and points per cluster to 0 on GPU before accumulation
        cudaMemset(d_centroid_temps, 0, CLUSTER_COUNT * sizeof(Point));
        cudaMemset(d_points_per_cluster, 0, CLUSTER_COUNT * sizeof(int));


        // 1. Launch kernal to assign clusters
        assignClusters<<<blocks, threads_per_block>>>(d_points, point_count, d_centroids, CLUSTER_COUNT);
        // 2. Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        // 3. NEW - Launch kernel to accumulate centroids

        accumulateCentroids<<<blocks, threads_per_block>>>(d_points, point_count, d_centroid_temps, d_points_per_cluster);
        cudaDeviceSynchronize();

        // 4. NEW - Compute new centroids on GPU

        computeCentroidsGPU<<<1, CLUSTER_COUNT>>>(d_centroids, d_centroid_temps, d_points_per_cluster, CLUSTER_COUNT);
        cudaDeviceSynchronize();  

    }


    // 5. Copy points back to CPU to update centroids
    cudaMemcpy(points, d_points, point_count * sizeof(Point), cudaMemcpyDeviceToHost);

    // 6. Free GPU memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_centroid_temps);
    cudaFree(d_points_per_cluster);

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