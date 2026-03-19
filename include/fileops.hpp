#ifndef FILEOPS_L_HPP
#define FILEOPS_L_HPP

#include "points.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
inline std::vector<Point> readcsv() {
  std::vector<Point> points;
  std::string line;
  std::ifstream file("dummy.csv");

  while (getline(file, line)) {
    std::stringstream lineStream(line);
    std::string bit;
    double x, y;
    getline(lineStream, bit, ',');
    x = stof(bit);
    getline(lineStream, bit, '\n');
    y = stof(bit);

    points.push_back(initialize_point(x, y));
  }
  return points;
}

inline void writecsv(Point *points, int point_size) {
  std::ofstream myfile;
  myfile.open("output.csv");
  // Header Write
  myfile << "x,y,c" << std::endl;

  for (int i = 0; i < point_size; i++) {
    Point *it = &points[i];
    myfile << it->x << "," << it->y << "," << it->cluster << std::endl;
  }
  myfile.close();
}
#endif // !FILEOPS_L_HPP
