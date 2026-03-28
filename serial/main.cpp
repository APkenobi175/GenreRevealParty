#include "fileops.hpp"
#include "kmeans.hpp"
#include "points.hpp"

#include <array>
#include <ctime>
#include <iostream>
#include <ostream>
#include <vector>


// 1. Initialize 5 Clusters and 5 KMeans Iterations
constexpr int CLUSTER_COUNT = 5;
constexpr int KMEANS_ITERATIONS = 5;

void kMeansClustering(std::vector<Point> &points) {
  // 2. Intialize Centroids Array and Randomly Assign Centroids
  std::array<Point, CLUSTER_COUNT> centroids;
  initializeCentroids(points, centroids.data(), CLUSTER_COUNT);
  // 3. Allocate Temporary Variables for KMeans Clustering 
  std::array<Point, CLUSTER_COUNT> centroid_temps;
  std::array<int, CLUSTER_COUNT> points_per_cluster;
  centroid_temps.fill(initialize_point(0, 0, 0, 0, 0, 0, 0, 0));
  points_per_cluster.fill(0);
  // 4. For each iteration: Assign each point to its nearest centroid, then recompute centroids as a mean of their assigned points.
  for (int i = 0; i < KMEANS_ITERATIONS; i++) {
    partitionClusters(centroids.data(), CLUSTER_COUNT, &points[0],
                      points.size(), centroid_temps.data(),
                      points_per_cluster.data(), i != (KMEANS_ITERATIONS - 1));
    computeCentroids(centroids.data(), centroid_temps.data(),
                     points_per_cluster.data(), CLUSTER_COUNT);
    // Reset Temporary variables
    centroid_temps.fill(initialize_point(0, 0, 0, 0, 0, 0, 0, 0));
    points_per_cluster.fill(0);
  }
  // 5. Write the output to a csv file
  writecsv(&points[0], points.size());
}

int main() {
  // A single point is a song with 8 fields (each field is a double)
  // The point struct can be found in include/points.hpp and the csv reading function can be found in include/fileops.hpp
  /*
     The 8 fields are as follows:

    1. Danceability
    2. Energy
    3. Loudness
    4. Speechiness
    5. Acousticness
    6. Instrumentalness
    7. Liveliness
    8. Valence
    */

  // 1. Reads CSV file and store points in the vector
  auto start_file_time = std::time(nullptr);
  std::vector<Point> points = readcsv();
  // 2. Print how long it took to read the file
  std::cout << "File Read time (s): " << std::time(nullptr) - start_file_time
            << std::endl;
  // 3. Perform Kmeans Clustering and time how long it takes, and print the time to the console.
  auto start_kmeans_time = std::time(nullptr);
  kMeansClustering(points);
  std::cout << "Kmeans clustering & writing time (s): "
            << std::time(nullptr) - start_kmeans_time << std::endl;
}
