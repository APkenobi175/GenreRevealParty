#ifndef POINTS_L_HPP
#define POINTS_L_HPP

// CUDA compatibility
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

// WARNING: To Ensure Compatibility with MPI, we need this struct to be plain
// data So please only use known sized types (or pointers) in this struct
// see
// https://stackoverflow.com/questions/422830/structure-of-a-c-object-in-memory-vs-a-struct
#include <cmath>
struct Point {
  double danceability, energy, speechiness, acousticness, instrumentalness,
      liveliness, valence;
  int cluster;    // no default cluster
  double minDist; // default infinite dist to nearest cluster

  __host__ __device__ double distance_point(const Point *q) {
    double dance_diff = this->danceability - q->danceability;
    double energy_diff = this->energy - q->energy;
    double speechiness_diff = this->speechiness - q->speechiness;
    double acousticness_diff = this->acousticness - q->acousticness;
    double instrumentalness_diff = this->instrumentalness - q->instrumentalness;
    double liveliness_diff = this->liveliness - q->liveliness;
    double valence_diff = this->valence - q->valence;

    double dance_square = dance_diff * dance_diff;
    double energy_square = energy_diff * energy_diff;
    double speechiness_square = speechiness_diff * speechiness_diff;
    double acousticness_square = acousticness_diff * acousticness_diff;
    double instrumentalness_square =
        instrumentalness_diff * instrumentalness_diff;
    double liveliness_square = liveliness_diff * liveliness_diff;
    double valence_square = valence_diff * valence_diff;

    return std::sqrt(dance_square + energy_square + speechiness_square +
                     acousticness_square + instrumentalness_square +
                     liveliness_square + valence_square);
  }

  __host__ __device__ void add_point(const Point *point) {
    danceability += point->danceability;
    energy += point->energy;
    speechiness += point->speechiness;
    acousticness += point->acousticness;
    instrumentalness += point->instrumentalness;
    liveliness += point->liveliness;
    valence += point->valence;
  }

  __host__ __device__ void sub_point(const Point *point) {
    danceability -= point->danceability;
    energy -= point->energy;
    speechiness -= point->speechiness;
    acousticness -= point->acousticness;
    instrumentalness -= point->instrumentalness;
    liveliness -= point->liveliness;
    valence -= point->valence;
  }

  __host__ __device__ void div_point(int divisor) {
    danceability /= divisor;
    energy /= divisor;
    speechiness /= divisor;
    acousticness /= divisor;
    instrumentalness /= divisor;
    liveliness /= divisor;
    valence /= divisor;
  }
};

inline Point initialize_point(double danceability, double energy,
                              double speechiness, double acousticness,
                              double instrumentalness, double liveliness,
                              double valence) {
  return {danceability, energy,  speechiness, acousticness, instrumentalness,
          liveliness,   valence, -1,          __DBL_MAX__};
}
#endif // !POINTS_L_HPP
