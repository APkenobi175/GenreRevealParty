#include "points.hpp"
#include <mpi.h>

inline void create_point_mpi_type(MPI_Datatype *out) {
  Point my_point;
  MPI_Aint base_address;
  MPI_Datatype typesig[3] = {MPI_DOUBLE, MPI_INT, MPI_DOUBLE};
  int block_lengths[3] = {7, 1, 1};
  MPI_Aint displacements[3];
  MPI_Get_address(&my_point, &base_address);
  MPI_Get_address(&my_point.danceability, &displacements[0]);
  MPI_Get_address(&my_point.cluster, &displacements[1]);
  MPI_Get_address(&my_point.minDist, &displacements[2]);
  displacements[0] -= base_address;
  displacements[1] -= base_address;
  displacements[2] -= base_address;

  MPI_Type_create_struct(3, block_lengths, displacements, typesig, out);
  MPI_Type_commit(out);
}

inline void calculate_displacements(int proc_count, int point_count,
                                    int *displs, int *thread_allocations) {

  int base_allocation = point_count / proc_count;
  int left_over = point_count % proc_count;

  // Determine the work share for each process (handles uneven division)
  for (int j = 0; j < proc_count; j++) {
    thread_allocations[j] = base_allocation + (left_over > j ? 1 : 0);
  }

  // Calculate the memory offsets (displacements) for the Scatterv call
  displs[0] = 0;
  for (int j = 1; j < proc_count; j++) {
    displs[j] = displs[j - 1] + thread_allocations[j - 1];
  }
}
void point_sum_wrapper(void *point1, void *point2, int *len,
                       MPI_Datatype *datatype);

inline void generate_point_sum_operation(MPI_Op *out) {

  MPI_Op_create(&point_sum_wrapper, 1, out);
}

void point_sum_wrapper(void *point1, void *point2, int *len,
                       MPI_Datatype *datatype) {
  Point *p1 = static_cast<Point *>(point1);
  Point *p2 = static_cast<Point *>(point2);

  for (int i = 0; i < *len; i++) {
    p2[i].add_point(&p1[i]);
  }
}
