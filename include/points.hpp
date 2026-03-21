#ifndef POINTS_L_HPP
#define POINTS_L_HPP

// WARNING: To Ensure Compatibility with MPI, we need this struct to be plain
// data So please only use known sized types (or pointers) in this struct
// see
// https://stackoverflow.com/questions/422830/structure-of-a-c-object-in-memory-vs-a-struct
#include <cmath>
struct Point {
  double danceability, energy, loudness, speechiness, acousticness,
      instrumentalness, liveliness, valence;
  int cluster;    // no default cluster
  double minDist; // default infinite dist to nearest cluster

  double distance_point(const Point *q) {
    double dance_diff = this->danceability - q->danceability;
    double energy_diff = this->energy - q->energy;
    double loudness_diff = this->loudness - q->loudness;
    double speechiness_diff = this->speechiness - q->speechiness;
    double acousticness_diff = this->acousticness - q->acousticness;
    double instrumentalness_diff = this->instrumentalness - q->instrumentalness;
    double liveliness_diff = this->liveliness - q->liveliness;
    double valence_diff = this->valence - q->valence;

    double dance_square = dance_diff * dance_diff;
    double energy_square = energy_diff * energy_diff;
    double loudness_square = loudness_diff * loudness_diff;
    double speechiness_square = speechiness_diff * speechiness_diff;
    double acousticness_square = acousticness_diff * acousticness_diff;
    double instrumentalness_square =
        instrumentalness_diff * instrumentalness_diff;
    double liveliness_square = liveliness_diff * liveliness_diff;
    double valence_square = valence_diff * valence_diff;

    return std::sqrt(dance_square + energy_square + loudness_square +
                     speechiness_square + acousticness_square +
                     instrumentalness_square + liveliness_diff +
                     valence_square);
  }

  void add_point(const Point *point) {
    danceability += point->danceability;
    energy += point->energy;
    loudness += point->loudness;
    speechiness += point->speechiness;
    acousticness += point->acousticness;
    instrumentalness += point->instrumentalness;
    liveliness += point->liveliness;
    valence += point->valence;
  }

  void sub_point(const Point *point) {
    danceability -= point->danceability;
    energy -= point->energy;
    loudness -= point->loudness;
    speechiness -= point->speechiness;
    acousticness -= point->acousticness;
    instrumentalness -= point->instrumentalness;
    liveliness -= point->liveliness;
    valence -= point->valence;
  }

  void div_point(int divisor) {
    danceability /= divisor;
    energy /= divisor;
    loudness /= divisor;
    speechiness /= divisor;
    acousticness /= divisor;
    instrumentalness /= divisor;
    liveliness /= divisor;
    valence /= divisor;
  }
};

inline Point initialize_point(double danceability, double energy,
                              double loudness, double speechiness,
                              double acousticness, double instrumentalness,
                              double liveliness, double valence) {
  return {danceability,     energy,     loudness, speechiness, acousticness,
          instrumentalness, liveliness, valence,  -1,          __DBL_MAX__};
}
#endif // !POINTS_L_HPP
