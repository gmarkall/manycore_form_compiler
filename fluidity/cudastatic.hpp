// cudastatic.hpp

#ifndef _CUDASTATIC_HPP
#define _CUDASTATIC_HPP

#define THREAD_ID threadIdx.x+blockIdx.x*blockDim.x
#define THREAD_COUNT blockDim.x*gridDim.x

// Hide the __global__ keywords from the ROSE compiler.

__global__ void transform_to_physical(double *node_coords, double *dn, double *quad_weights, double *dshape, double *detwei, int n_ele, int n_dim, int n_quad, int nodes_per_ele);

__global__ void matrix_addto(int* glob_mat_findrm, int* glob_mat_colm, double* glob_mat_val, int *local_to_global, double *local_matrices, int n_ele, int nodes_per_ele);

__global__ void vector_addto(double* glob_vec, int *local_to_global, double *local_vectors, int n_ele, int nodes_per_ele);

void cg_solve(int* findrm, int size_findrm, int* colm, int size_colm, double* k_val, double* b, int rhs_size, double *x);
void cg_solve_lma(int* k_findrm, int size_findrm, int* k_colm, int size_colm, double* k_val, double* k_b, int rhs_val_size, double *x_p, int n_ele, double *matrix, int *node_nums);

// Change data from the one-value-per node to one-value-per-node-per element layout and back
__global__ void expand_data(double* dest, double* src, int* node_nums, int n_ele, int n_vals_per_node, int nodes_per_ele);

__global__ void contract_data(double* dest, double* src, int* node_nums, int n_ele, int n_vals_per_node, int nodes_per_ele);

int t2p_shmemsize(int block_x_dim, int n_dim, int nodes_per_ele);

#endif
