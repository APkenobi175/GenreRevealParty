#ifndef FILEOPS_L_HPP
#define FILEOPS_L_HPP

#include "points.hpp"
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static inline int count_end_chars(std::string &string, char c) {

  int end_chars = 0;
  for (int offset = string.length() - 1; offset >= 0; offset--) {
    if (string[offset] == '\"') {
      end_chars++;
    } else {
      break;
    }
  }
  return end_chars;
}

static inline void safe_get_csv_val(std::stringstream &stream,
                                    std::string &substring) {
  std::string temp;
  getline(stream, substring, ',');
  // Implement safe comma escaping
  if (substring.find("\"") != substring.npos) {

    int end_quote_count = count_end_chars(substring, '\"');

    while (end_quote_count % 2 == 0) {
      getline(stream, temp, ',');
      substring.append(", " + temp);

      // To properly handle escaped strings, we must ensure each quote is
      // escaped before returning the csv value
      end_quote_count = count_end_chars(substring, '\"');
    }
  }
}
inline std::vector<Point> readcsv() {
  std::vector<Point> points;
  std::string line;
  std::ifstream file("../data/tracks_features.csv");

  // Skip Header
  std::getline(file, line);
  while (getline(file, line)) {
    std::stringstream lineStream(line);
    std::string bit;
    double danceability, energy, loudness, speechiness, acousticness,
        instrumentalness, liveliness, valence;
    // Skip first nine elements in tracks_features

    for (int i = 0; i < 9; i++)
      safe_get_csv_val(lineStream, bit);

    safe_get_csv_val(lineStream, bit);
    danceability = stof(bit);
    safe_get_csv_val(lineStream, bit);
    energy = stof(bit);

    // Skip key
    safe_get_csv_val(lineStream, bit);

    safe_get_csv_val(lineStream, bit);
    loudness = stof(bit);

    // Skip mode
    safe_get_csv_val(lineStream, bit);

    safe_get_csv_val(lineStream, bit);
    speechiness = stof(bit);
    safe_get_csv_val(lineStream, bit);
    acousticness = stof(bit);
    safe_get_csv_val(lineStream, bit);
    instrumentalness = stof(bit);
    safe_get_csv_val(lineStream, bit);
    liveliness = stof(bit);
    safe_get_csv_val(lineStream, bit);
    valence = stof(bit);
    getline(lineStream, bit, '\n');
    points.push_back(initialize_point(danceability, energy, loudness,
                                      speechiness, acousticness,
                                      instrumentalness, liveliness, valence));
  }
  return points;
}

inline void writecsv(Point *points, int point_size) {
  std::ofstream myfile;
  myfile.open("output.csv");
  // Header Write
  myfile << "danceability,energy,loudness,speechiness,acousticness,"
            "instrumentalness,liveliness,valence,c"
         << std::endl;

  for (int i = 0; i < point_size; i++) {
    Point *it = &points[i];
    myfile << it->danceability << "," << it->energy << "," << it->loudness
           << "," << it->speechiness << "," << it->acousticness << ","
           << it->instrumentalness << "," << it->liveliness << ","
           << it->valence << "," << it->cluster << std::endl;
  }
  myfile.close();
}
#endif // !FILEOPS_L_HPP
