#include "fileops.hpp"
#include "kmeans.hpp"
#include "points.hpp"

#include <array>
#include <vector>

constexpr int CLUSTER_COUNT = 5;
constexpr int KMEANS_ITERATIONS = 5;

void kMeansClustering(std::vector<Point> &points) {
  std::array<Point, CLUSTER_COUNT> centroids;
  initializeCentroids(points, centroids.data(), CLUSTER_COUNT);

  std::array<Point, CLUSTER_COUNT> centroid_temps;
  std::array<int, CLUSTER_COUNT> points_per_cluster;
  centroid_temps.fill(initialize_point(0, 0));
  points_per_cluster.fill(0);
  for (int i = 0; i < KMEANS_ITERATIONS; i++) {
    partitionClusters(centroids.data(), CLUSTER_COUNT, &points[0],
                      points.size(), centroid_temps.data(),
                      points_per_cluster.data(), i != (KMEANS_ITERATIONS - 1));
    computeCentroids(centroids.data(), centroid_temps.data(),
                     points_per_cluster.data(), CLUSTER_COUNT);
    // Reset Temporary variables
    centroid_temps.fill(initialize_point(0, 0));
    points_per_cluster.fill(0);
  }
  writecsv(&points[0], points.size());
}

int main() {
  std::vector<Point> points = readcsv();
  kMeansClustering(points);
}
