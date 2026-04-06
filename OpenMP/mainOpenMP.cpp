#include "fileops.hpp"
#include "kmeans.hpp"
#include "points.hpp"

#include <array>
#include <ctime>
#include <iostream>
#include <ostream>
#include <vector>
#include <chrono>
#include <omp.h>
constexpr int CLUSTER_COUNT = 5;
constexpr int KMEANS_ITERATIONS = 5;
inline void partitionOpenMPClusters(Point centroids[], int c_count, Point points[],
                              int p_count, Point *c_temp, int *cluster_amt,
                              bool resetClusters) {  
  // We assume that c_sum and c_amt are initialized to zeros!
  int thread_count = omp_get_max_threads();
  std::vector<std::vector<Point> > local_c_temp(thread_count, std::vector<Point>(c_count, initialize_point(0, 0, 0, 0, 0, 0, 0, 0)));
  std::vector<std::vector<int> > local_cluster_amt(thread_count, std::vector<int>(c_count, 0));
  for (int c = 0; c < c_count; c++) {
    
    //parallelize this loop using OpenMP
    #pragma omp parallel for
    for (int p_i = 0; p_i < p_count; p_i++) {
      int thread_id = omp_get_thread_num();
      Point *current_point = &points[p_i];
      double dist = current_point->distance_point(&centroids[c]);

      if (dist < current_point->minDist) {
        current_point->minDist = dist;
        
        local_c_temp[thread_id][c].add_point(current_point);
        local_cluster_amt[thread_id][c]++;
        // If point already belongs to a cluster
        // we need to erase it from the other cluster
        if (current_point->cluster != -1) {
          local_c_temp[thread_id][current_point->cluster].sub_point(current_point);
          local_cluster_amt[thread_id][current_point->cluster]--;
        }
        current_point->cluster = c;
      }
    }
    
  }
  for(int d = 0; d < c_count; d++){
    for(int t = 0; t < thread_count; t++){
      c_temp[d].add_point(&local_c_temp[t][d]);
      cluster_amt[d] += local_cluster_amt[t][d];
    }
  }
  if (resetClusters) {

    // Reset clusters back to zero
    //parallelize this loop using OpenMP
   #pragma omp parallel for
    for (int p_i = 0; p_i < p_count; p_i++) {
      points[p_i].cluster = -1;
      points[p_i].minDist = __DBL_MAX__;
    }
  }
}
void kMeansClustering(std::vector<Point> &points) {
  std::array<Point, CLUSTER_COUNT> centroids;
  initializeCentroids(points, centroids.data(), CLUSTER_COUNT);

  std::array<Point, CLUSTER_COUNT> centroid_temps;
  std::array<int, CLUSTER_COUNT> points_per_cluster;
  centroid_temps.fill(initialize_point(0, 0, 0, 0, 0, 0, 0, 0));
  points_per_cluster.fill(0);
  //parallelize this or section using OpenMP
  int i;

  for (i = 0; i < KMEANS_ITERATIONS; i++) {
    partitionOpenMPClusters(centroids.data(), CLUSTER_COUNT, &points[0],
                      points.size(), centroid_temps.data(),
                      points_per_cluster.data(), i != (KMEANS_ITERATIONS - 1));
    computeCentroids(centroids.data(), centroid_temps.data(),
                     points_per_cluster.data(), CLUSTER_COUNT);
    // Reset Temporary variables
    centroid_temps.fill(initialize_point(0, 0, 0, 0, 0, 0, 0, 0));
    points_per_cluster.fill(0);
  }
}
//}

int main() {
  auto start_file_time = std::chrono::high_resolution_clock::now();
  std::vector<Point> points = readcsv();
  // 2. Print how long it took to read the file
  auto end_file_time = std::chrono::high_resolution_clock::now();
  auto file_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_file_time - start_file_time);
  std::cout << "File Read time (ms): " << file_duration.count() << std::endl;
  // 3. Perform Kmeans Clustering and time how long it takes, and print the time to the console.
  auto start_kmeans_time = std::chrono::high_resolution_clock::now();
  kMeansClustering(points);
  auto end_kmeans_time = std::chrono::high_resolution_clock::now();
  auto kmeans_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_kmeans_time - start_kmeans_time);
  std::cout << "Kmeans clustering time (ms): " << kmeans_duration.count() << std::endl;
  auto start_write_time = std::chrono::high_resolution_clock::now();
  writecsv(&points[0], points.size());
  auto end_write_time = std::chrono::high_resolution_clock::now();
  auto write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_write_time - start_write_time);
  std::cout << "File write time (ms): " << write_duration.count() << std::endl;
          
}
