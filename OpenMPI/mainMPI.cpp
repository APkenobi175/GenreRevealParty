#include "fileops.hpp"
#include "kmeans.hpp"
#include "points.hpp"

#include <array>
#include <ctime>
#include <iostream>
#include <ostream>
#include <vector>
#include <chrono>
#include <mpi.h>
constexpr int CLUSTER_COUNT = 5;
constexpr int KMEANS_ITERATIONS = 5;

void point_sum_wrapper(void *point1, void *point2, int *len, MPI_Datatype *datatype) {
    Point *p1 = static_cast<Point*>(point1);
    Point *p2 = static_cast<Point*>(point2);
    
    for (int i = 0; i < *len; i++) {
        p2[i].add_point(&p1[i]); 
    }
}
void kMeansClustering(std::vector<Point> &points, int my_rank, int comm_sz, int total_points) {
  std::array<Point, CLUSTER_COUNT> centroids;
  if(my_rank == 0){
  initializeCentroids(points, centroids.data(), CLUSTER_COUNT);
  }
  std::array<Point, CLUSTER_COUNT> global_centroid_temps;
    std::array<Point, CLUSTER_COUNT> local_centroid_temps;
  std::array<int, CLUSTER_COUNT> local_points_per_cluster;
std::array<int, CLUSTER_COUNT> global_points_per_cluster;
  local_centroid_temps.fill(initialize_point(0, 0, 0, 0, 0, 0, 0, 0));
  local_points_per_cluster.fill(0);
  int i;

  Point my_point;
  MPI_Aint base_address;
  MPI_Datatype typesig[3] = {MPI_DOUBLE, MPI_INT, MPI_DOUBLE};
  int block_lengths[3] = {8, 1, 1};
MPI_Aint displacements[3];
MPI_Get_address(&my_point, &base_address);
MPI_Get_address(&my_point.danceability, &displacements[0]); 
MPI_Get_address(&my_point.cluster, &displacements[1]);      
MPI_Get_address(&my_point.minDist, &displacements[2]);  
displacements[0] -= base_address;
displacements[1] -= base_address;
displacements[2] -= base_address;    
MPI_Datatype MPI_POINT_TYPE;
MPI_Type_create_struct(3, block_lengths, displacements, typesig, &MPI_POINT_TYPE);
MPI_Type_commit(&MPI_POINT_TYPE);
MPI_Op MPI_POINT_SUM;
MPI_Op_create(&point_sum_wrapper, 1, &MPI_POINT_SUM);
  
  int threadAlloc[comm_sz];
    for(int j = 0; j < comm_sz; j++){
        threadAlloc[j] = 0;
    }
    int displs[comm_sz];
    
    // Determine the work share for each process (handles uneven division)
    for(int j = 0; j < total_points; j++){
        threadAlloc[j%comm_sz]++;
    }
    
    // Calculate the memory offsets (displacements) for the Scatterv call
    displs[0] = 0;
    for(int j = 1; j < comm_sz; j++){
        displs[j] = displs[j-1] + threadAlloc[j-1];
    }

    // Prepare local buffers for the work chunk
    int recvcount = threadAlloc[my_rank];
    std::vector<Point> local_data(recvcount);
  MPI_Bcast(&centroids[0], CLUSTER_COUNT, MPI_POINT_TYPE, 0, MPI_COMM_WORLD);
  MPI_Scatterv(points.data(), threadAlloc, displs, MPI_POINT_TYPE, local_data.data(), recvcount, MPI_POINT_TYPE, 0, MPI_COMM_WORLD);
  for (i = 0; i < KMEANS_ITERATIONS; i++) {
    partitionClusters(centroids.data(), CLUSTER_COUNT, &local_data[0],
                      local_data.size(), local_centroid_temps.data(),
                      local_points_per_cluster.data(), i != (KMEANS_ITERATIONS - 1));
    MPI_Allreduce(local_centroid_temps.data(), global_centroid_temps.data(), CLUSTER_COUNT, MPI_POINT_TYPE, MPI_POINT_SUM,
              MPI_COMM_WORLD);
    MPI_Allreduce(local_points_per_cluster.data(), global_points_per_cluster.data(), CLUSTER_COUNT, MPI_INT, MPI_SUM,
              MPI_COMM_WORLD);
    computeCentroids(centroids.data(), global_centroid_temps.data(),
                     global_points_per_cluster.data(), CLUSTER_COUNT);
    // Reset Temporary variables
    local_centroid_temps.fill(initialize_point(0, 0, 0, 0, 0, 0, 0, 0));
    local_points_per_cluster.fill(0);
  }
  
    MPI_Gatherv(local_data.data(), local_data.size(), MPI_POINT_TYPE,
    points.data(), threadAlloc, displs, MPI_POINT_TYPE,
    0, MPI_COMM_WORLD);
    MPI_Type_free(&MPI_POINT_TYPE);
MPI_Op_free(&MPI_POINT_SUM);
  
}
//}

int main() {
  int my_rank, comm_sz;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 
  std::vector<Point> points;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_file_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_kmeans_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_kmeans_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_write_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_write_time;
  int total_points;
  if(my_rank == 0){
    start_file_time = std::chrono::high_resolution_clock::now();
    points = readcsv();
    total_points = points.size();
  // 2. Print how long it took to read the file
  auto end_file_time = std::chrono::high_resolution_clock::now();
  auto file_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_file_time - start_file_time);
  std::cout << "File Read time (ms): " << file_duration.count() << std::endl;
  // 3. Perform Kmeans Clustering and time how long it takes, and print the time to the console.
  start_kmeans_time = std::chrono::high_resolution_clock::now();
  }
   MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
 
  kMeansClustering(points, my_rank, comm_sz, total_points);
  if(my_rank == 0){
  end_kmeans_time = std::chrono::high_resolution_clock::now();
  auto kmeans_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_kmeans_time - start_kmeans_time);
  std::cout << "Kmeans clustering time (ms): " << kmeans_duration.count() << std::endl;
  start_write_time = std::chrono::high_resolution_clock::now();
  writecsv(&points[0], points.size());
  end_write_time = std::chrono::high_resolution_clock::now();
  auto write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_write_time - start_write_time);
  std::cout << "File write time (ms): " << write_duration.count() << std::endl;
  }
  MPI_Finalize();        
}
