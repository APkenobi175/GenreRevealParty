#ifndef KMEANS_L_HPP
#define KMEANS_L_HPP

#include "points.hpp"
#include <cstdlib>
#include <stdexcept>
#include <vector>

// Serial Call, Initializes an array of centroids of size cluster_count
// based on points vector
inline void initializeCentroids(std::vector<Point> points, Point *centroids,
                                int cluster_count) {
  srand(100);

  for (int i = 0; i < cluster_count; i++) {
    centroids[i] = points.at(rand() % points.size());
    centroids[i].cluster = i;
  }
}

// Thread-safe, MPI compliant partitioning of an array of points into
// centroid groups, with size of each group stored in c_amt, and sum
// of each cluster_group stored for each centroid in c_sum
inline void partitionClusters(Point centroids[], int c_count, Point points[],
                              int p_count, Point *c_temp, int *cluster_amt) {

  // We assume that c_sum and c_amt are initialized to zeros!
  for (int c = 0; c < c_count; c++) {
    for (int p_i = 0; p_i < p_count; p_i++) {

      Point *current_point = &points[p_i];
      double dist = current_point->distance_point(&centroids[c]);

      if (dist < current_point->minDist) {
        current_point->minDist = dist;
        c_temp[c].add_point(current_point);
        cluster_amt[c]++;
        // If point already belongs to a cluster
        // we need to erase it from the other cluster
        if (current_point->cluster != -1) {
          c_temp[current_point->cluster].sub_point(current_point);
          cluster_amt[current_point->cluster]--;
        }
        current_point->cluster = c;
      }
      if (c_temp[c].x > 100000000 || c_temp[c].y > 100000000) {
        throw new std::runtime_error("Uh oh");
      }
    }
  }

  // Reset clusters back to zero
  for (int p_i = 0; p_i < p_count; p_i++) {
    points[p_i].cluster = -1;
    points[p_i].minDist = __DBL_MAX__;
  }
}

// This function requires fully summed centroids and total
// cluster amounts, so centroid sums must have been calculated
// already, and cluster sizes must have already been summed
inline void computeCentroids(Point *centroids, Point *centroid_temps,
                             int *cluster_amt, int centroids_size) {
  for (int c = 0; c < centroids_size; c++) {
    if (cluster_amt[c] > 0) {
      centroid_temps[c].div_point(cluster_amt[c]);
    }
    centroids[c] = centroid_temps[c];
    centroids[c].cluster = c;
  }
}

#endif // !KMEANS_L
