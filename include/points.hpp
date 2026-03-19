#ifndef POINTS_L_HPP
#define POINTS_L_HPP

// WARNING: To Ensure Compatibility with MPI, we need this struct to be plain
// data So please only use known sized types (or pointers) in this struct
// see
// https://stackoverflow.com/questions/422830/structure-of-a-c-object-in-memory-vs-a-struct
struct Point {
  double x, y;    // coordinates
  int cluster;    // no default cluster
  double minDist; // default infinite dist to nearest cluster

  inline double distance_point(const Point *q) {
    int x_diff = this->x - q->x;
    int y_diff = this->y - q->y;
    return x_diff * x_diff + y_diff * y_diff;
  }

  void add_point(const Point *point) {
    x += point->x;
    y += point->y;
  }

  void sub_point(const Point *point) {
    x -= point->x;
    y -= point->y;
  }

  void div_point(int divisor) {
    x /= divisor;
    y /= divisor;
  }
};

inline Point initialize_point(double x, double y) {
  return {x, y, -1, __DBL_MAX__};
}
#endif // !POINTS_L_HPP
