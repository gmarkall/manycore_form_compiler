// cudastatic.cu
//
// Used by generated code - contains functions that will be the same in 
// every implementation.

// Utility functions for calculating locations when using 
// expanded data layout.

#include "cudastatic.hpp"

#include <unistd.h> // For debugging only
#include <cstdio> // likewise

__device__ int locToGlobIdx(int ele, int node, int nodes_per_ele)
{
  return ele*nodes_per_ele + node;
}

__device__ int eleIdx(int ele, int node, int n_ele) 
{
  return node*n_ele + ele;
}

__device__ int eleIdx(int ele, int node, int val, int n_ele, int n_vals_per_node)
{
  return node*n_ele*n_vals_per_node + val*n_ele + ele;
}

__device__ int lMatIdx(int x, int y, int cur_ele, int n_ele) 
{
  return x*3*n_ele + y*n_ele + cur_ele;
}

// For the addto in global matrices.

__device__ void atomicDoubleAdd(double *address, double val)
{
  unsigned long long int new_val, old;
  unsigned long long int old2 = __double_as_longlong(*address);

  do 
  {
    old = old2;
    new_val = __double_as_longlong(__longlong_as_double(old) + val);
    old2 = atomicCAS((unsigned long long int *)address, old, new_val);
  } while(old2!=old);

}

// For searching CSR sparsity patterns for the index into val for a
// given x,y.

__device__ int pos(int i, int j, int* findrm, int* colm)
{
  int csr_pos, this_pos, this_j, row, base;
  int upper_pos, upper_j, lower_pos, lower_j;

  row = findrm[i-1]-1;
  base = row;
  upper_pos = findrm[i]-findrm[i-1]-1;
  upper_j = colm[row+upper_pos];
  lower_pos=0;
  lower_j = colm[row];

  if (upper_j==j) 
  {
    csr_pos=upper_pos+base;
    return csr_pos;
  }
  else if (lower_j==j) 
  {
    csr_pos=lower_pos+base;
    return csr_pos;
  }

  while(upper_pos-lower_pos>1) 
  {
    this_pos=(upper_pos+lower_pos)/2;
    this_j = colm[row+this_pos];

    if(this_j==j) 
    {
      csr_pos=this_pos+base;
      return csr_pos;
    }
    else if(this_j>j) 
    {
      upper_j=this_j;
      upper_pos=this_pos;
    }
    else if(this_j<j) 
    {
      lower_j=this_j;
      lower_pos=this_pos;
    }
  }

  return INT_MAX; // Hope to generate ULF by going so far out of the array that we pass the end of the memory.
}

// Addto Kernels

// These are only going to work for scalar fields at the moment,
// as the block_csr_matrix type from fluidity is not implemented
// at present (I think that implementing it would be the best
// (easiest?)

__global__ void matrix_addto(int*findrm, int *colm, double *global_matrix_val, int *local_to_global, double *local_matrices, int n_ele, int nodes_per_ele)
{
  for (int i=THREAD_ID; i<n_ele; i+=THREAD_COUNT) 
  {
    for (int x=0; x<nodes_per_ele; x++) 
    {
      for (int y=0; y<nodes_per_ele; y++) 
      {
        int mat_x = local_to_global[locToGlobIdx(i,x,nodes_per_ele)];
        int mat_y = local_to_global[locToGlobIdx(i,y,nodes_per_ele)];
        int mpos = pos(mat_x, mat_y, findrm, colm);
        int localMatrixIdx = lMatIdx(x,y,i,n_ele);
        atomicDoubleAdd(&global_matrix_val[mpos], local_matrices[localMatrixIdx]);
      }
    }
  }
}

__global__ void vector_addto(double *global_vector, int *local_to_global, double *local_vectors, int n_ele, int nodes_per_ele)
{
  for (int ele=THREAD_ID; ele<n_ele; ele+=THREAD_COUNT) 
  {
    for (int i=0; i<nodes_per_ele; i++)
    {
      int globalIdx = local_to_global[locToGlobIdx(ele, i, nodes_per_ele)]-1;
      int localIdx = eleIdx(ele,i,n_ele);
      atomicDoubleAdd(&global_vector[globalIdx], local_vectors[localIdx]);
    }
  }
}

__global__ void vector_addto_spmv(int n, int n_ele, int *findrm, int *colm, double *vec, double *vec_elemental) {
  for(int row=THREAD_ID; row<n; row+=THREAD_COUNT) {
    int a=findrm[row];
    int b=findrm[row+1];
    for(int k=a;k<b;k++) {
      int i=colm[k-1]-1;
      vec[row] += vec_elemental[eleIdx(i/3,i%3,n_ele)];
    }
  }
}


// Data Expansion and contraction kernels

__global__ void expand_data(double *dest, double *src, int *local_to_global, int n_ele, int n_vals_per_node, int nodes_per_ele)
{
  for (int ele=THREAD_ID; ele<n_ele; ele+=THREAD_COUNT) 
  {
    for (int node=0; node<nodes_per_ele; node++)
    {
      for (int val=0; val<n_vals_per_node; val++)
      {
        int destIdx = eleIdx(ele,node,val,n_ele,n_vals_per_node);
        int srcIdx = (n_vals_per_node*(local_to_global[ele*nodes_per_ele+node]-1))+val;
        dest[destIdx] = src[srcIdx];
      }
    }
  }
}

__global__ void contract_data(double *dest, double *src, int *local_to_global, int n_ele, int n_vals_per_node, int nodes_per_ele)
{
  for (int ele=THREAD_ID; ele<n_ele; ele+=THREAD_COUNT) 
  {
    for (int node=0; node<nodes_per_ele; node++)
    {
      for (int val=0; val<n_vals_per_node; val++)
      {
        int srcIdx =  eleIdx(ele,node,val,n_ele,n_vals_per_node);
        int destIdx = (n_vals_per_node*(local_to_global[ele*nodes_per_ele+node]-1))+val;
        dest[destIdx] = src[srcIdx];
      }
    }
  }
}
