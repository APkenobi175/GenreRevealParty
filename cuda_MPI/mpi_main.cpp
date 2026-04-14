#include "../include/fileops.hpp"
#include "../include/kmeans.hpp"
#include "../include/mpi_shared.h"
#include "../include/points.hpp"

#include <array>
#include <chrono>
#include <ctime>
#include <mpi.h>
constexpr int CLUSTER_COUNT = 5;
constexpr int KMEANS_ITERATIONS = 5;
void accumulatingStep(Point *points, int point_count, int rank,
                      Point *centroids, Point *centroid_temps,
                      int cluster_count, int *points_per_cluster,
                      bool shouldCopyPoints);

void startGPUs(std::vector<Point> &points, int my_rank, int comm_sz,
               int total_points) {
  std::array<Point, CLUSTER_COUNT> centroids;
  if (my_rank == 0) {
    initializeCentroids(points, centroids.data(), CLUSTER_COUNT);
  }
  std::array<Point, CLUSTER_COUNT> global_centroid_temps;
  std::array<Point, CLUSTER_COUNT> local_centroid_temps;
  std::array<int, CLUSTER_COUNT> local_points_per_cluster;
  std::array<int, CLUSTER_COUNT> global_points_per_cluster;
  local_centroid_temps.fill(initialize_point(0, 0, 0, 0, 0, 0, 0));
  local_points_per_cluster.fill(0);
  int i;

  MPI_Datatype MPI_POINT_TYPE;
  create_point_mpi_type(&MPI_POINT_TYPE);

  MPI_Op MPI_POINT_SUM;
  generate_point_sum_operation(&MPI_POINT_SUM);

  int threadAlloc[comm_sz];
  int displs[comm_sz];

  // AP 4/11/26 - Change points.size() to total_points as points.size() is only
  // valid for rank 0 This is why it works for only one task, but as soon as we
  // have more than one task it doesn't work.
  calculate_displacements(comm_sz, total_points, displs, threadAlloc);

  // Prepare local buffers for the work chunk
  int recvcount = threadAlloc[my_rank];
  std::vector<Point> local_data(recvcount);

  MPI_Bcast(&centroids[0], CLUSTER_COUNT, MPI_POINT_TYPE, 0, MPI_COMM_WORLD);
  MPI_Scatterv(points.data(), threadAlloc, displs, MPI_POINT_TYPE,
               local_data.data(), recvcount, MPI_POINT_TYPE, 0, MPI_COMM_WORLD);

  for (i = 0; i < KMEANS_ITERATIONS; i++) {
    // Partition and accumulate centroids
    accumulatingStep(local_data.data(), local_data.size(), my_rank,
                     centroids.data(), local_centroid_temps.data(),
                     CLUSTER_COUNT, local_points_per_cluster.data(),
                     i == (KMEANS_ITERATIONS - 1));

    // Sum centroids and points per cluster
    MPI_Allreduce(local_centroid_temps.data(), global_centroid_temps.data(),
                  CLUSTER_COUNT, MPI_POINT_TYPE, MPI_POINT_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_points_per_cluster.data(),
                  global_points_per_cluster.data(), CLUSTER_COUNT, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);
    computeCentroids(centroids.data(), global_centroid_temps.data(),
                     global_points_per_cluster.data(), CLUSTER_COUNT);
    // Reset Temporary variables
    local_centroid_temps.fill(initialize_point(0, 0, 0, 0, 0, 0, 0));
    local_points_per_cluster.fill(0);
  }

  MPI_Gatherv(local_data.data(), local_data.size(), MPI_POINT_TYPE,
              points.data(), threadAlloc, displs, MPI_POINT_TYPE, 0,
              MPI_COMM_WORLD);
  MPI_Type_free(&MPI_POINT_TYPE);
  MPI_Op_free(&MPI_POINT_SUM);
}

int main() {
  using namespace std::chrono;
  int my_rank, comm_sz;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  std::vector<Point> points;
  time_point<high_resolution_clock> start_file_time;
  time_point<high_resolution_clock> start_kmeans_time;
  time_point<high_resolution_clock> end_kmeans_time;
  time_point<high_resolution_clock> start_write_time;
  time_point<high_resolution_clock> end_write_time;
  int total_points;
  if (my_rank == 0) {
    start_file_time = high_resolution_clock::now();
    points = readcsv();
    total_points = points.size();
    // 2. Print how long it took to read the file
    auto end_file_time = high_resolution_clock::now();
    auto file_duration =
        duration_cast<milliseconds>(end_file_time - start_file_time);
    std::cout << "File Read time (ms): " << file_duration.count() << std::endl;
    // 3. Perform Kmeans Clustering and time how long it takes, and print the
    // time to the console.
  }

  // Time duration of algorithm from the point where the GPUs are initialized
  MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
  start_kmeans_time = high_resolution_clock::now();
  startGPUs(points, my_rank, comm_sz, total_points);

  end_kmeans_time = high_resolution_clock::now();
  auto kmeans_duration =
      duration_cast<milliseconds>(end_kmeans_time - start_kmeans_time);
  long local_ms_taken = kmeans_duration.count();
  long ms_taken;
  MPI_Reduce(&local_ms_taken, &ms_taken, 1, MPI_LONG, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (my_rank == 0) {
    std::cout << "Kmeans clustering time (ms): " << ms_taken << std::endl;
    start_write_time = high_resolution_clock::now();
    writecsv(&points[0], points.size());
    end_write_time = high_resolution_clock::now();
    auto write_duration = duration_cast<std::chrono::milliseconds>(
        end_write_time - start_write_time);
    std::cout << "File write time (ms): " << write_duration.count()
              << std::endl;
  }
  MPI_Finalize();
}
