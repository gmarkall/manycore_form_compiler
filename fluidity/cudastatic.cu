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

// Transform_to_physical - assuming linear space and element for now.

// Compute Shared mem size for transform to physical

int t2p_shmemsize(int block_x_dim, int n_dim, int nodes_per_ele)
{
  int jac_size = block_x_dim*n_dim*n_dim;
  int invJ_size = jac_size;
  int local_dn_size = block_x_dim*n_dim*nodes_per_ele;
  int local_coords_size = local_dn_size;
  int shape_dn_size = block_x_dim*n_dim;
  int local_dshape_size = block_x_dim*n_dim;

  int shmemsize = (jac_size+local_dn_size+invJ_size+local_coords_size+shape_dn_size+local_dshape_size) 
                  * sizeof(double);

  return shmemsize;
}

// Little utility for small matrix multiplies.

__device__ void matmult(double *a, int ax, int ay, double *b, int bx, int by, double *result)
{
  int i, j, k;
  for(i=0; i<ax; i++) {
    for(j=0; j<by; j++) {
      result[j*ax+i]=0.0;
      for(k=0; k<ay; k++)
        result[j*ax+i] += a[k*ax+i] * b[j*bx+k];
    }
  }
}

// The new version, closer to what's done in fluidity than I was using previously.

__global__ void transform_to_physical(double *node_coords, double *dn, double *quad_weights, double *dshape, double *detwei, int n_ele, int n_dim, int n_quad, int nodes_per_ele)
{
  extern __shared__ double shmem[];
  
  for (int ele=THREAD_ID; ele<n_ele; ele+=THREAD_COUNT)
  {
    // Size of local data stores
    const int jac_size = n_dim*n_dim;
    const int jacs_in_a_block_size = blockDim.x*jac_size;
    const int local_dn_size = n_dim*nodes_per_ele;
    const int local_dn_in_a_block_size = blockDim.x*local_dn_size;
    const int shape_dn_size = n_dim;
    const int shape_dn_in_a_block_size = blockDim.x*shape_dn_size;
    const int local_dshape_size = n_dim;
    // Unused for now - uncomment it when you want it later
    // (might as well leave it in rather than risk messing up
    // when you rewrite it
    //const int local_dshape_in_a_block_size = blockDim.x*local_dshape_size;

    // Pointers to shared mem for the local data
    double *J_local_T = shmem + (threadIdx.x*jac_size);
    double *local_dn = shmem + jacs_in_a_block_size + threadIdx.x*local_dn_size;
    double *invJ_local = shmem + jacs_in_a_block_size + local_dn_in_a_block_size + threadIdx.x*jac_size;
    double *local_coords = invJ_local + jacs_in_a_block_size +threadIdx.x*local_dn_size;
    double *shape_dn = local_coords +local_dn_in_a_block_size + threadIdx.x*shape_dn_size;
    double *dshape_local = shape_dn + shape_dn_in_a_block_size + threadIdx.x*local_dshape_size;

    // Other local variables
    double detJ_local;
    
    for(int q=0; q<n_quad; q++)
    {
      // Gather data for matrix multiplication
      for(int i=0; i<nodes_per_ele; ++i)
      {
        for(int j=0; j<n_dim; ++j)
	{
	  const int localIdx = j*nodes_per_ele+i;
	  const int localCoordsIdx = i*n_dim+j;
	  const int nodeCoordsIdx = eleIdx(ele,i,j,n_ele,n_dim);
	  local_dn[localIdx] = dn[j*(n_quad*nodes_per_ele) + q*(n_dim) + i];
	  local_coords[localCoordsIdx] = node_coords[nodeCoordsIdx];
	}
      }

      matmult(local_coords, n_dim, nodes_per_ele, local_dn, nodes_per_ele, n_dim, J_local_T);

      // Compute inverse of Jacobian
      switch (n_dim)
      {
	case 1:
	  invJ_local[0] = 1.0;
	  break;

	case 2:
	  invJ_local[0] =  J_local_T[3];
	  invJ_local[1] = -J_local_T[2];
	  invJ_local[2] = -J_local_T[1];
	  invJ_local[3] =  J_local_T[0];
	  break;

	case 3:
	  // Implement later.
	  break;
      }

      //Compute determinant of Jacobian
      switch (n_dim)
      {
	case 1:
	  detJ_local = J_local_T[0];
	  break;
	
	case 2:
	  detJ_local = (J_local_T[0]*J_local_T[3]) - (J_local_T[1]*J_local_T[2]);
	  break;

	case 3:
	  // Implement later.
	  break;
      }

      // Scale inverse by determinant
      for (int i=0; i<jac_size; ++i)
      {
	invJ_local[i] = invJ_local[i]/detJ_local;
      }

      // Evaluate derivatives in physical space.
      for (int i=0; i<nodes_per_ele; ++i)
      {
        // First, gather data for matrix multiplication
        for (int j=0; j<n_dim; ++j)
	{
	  const int localIdx = j;
	  shape_dn[localIdx] = dn[j*(n_quad*nodes_per_ele) + q*(n_dim) + i];
	}
        
	// Transform to physical space
	matmult(invJ_local, n_dim, n_dim, shape_dn, n_dim, 1, dshape_local);
	
	// Scatter result into dshape
	for (int j=0; j<n_dim; ++j)
	{
	  const int dshapeIdx = ele + j*n_ele + q*n_dim*n_ele + i*n_quad*n_dim*n_ele;
	  dshape[dshapeIdx] = dshape_local[j];
	}
      }

      const int detweiIdx = eleIdx(ele,q,n_ele);
      detwei[detweiIdx] = fabs(detJ_local)*quad_weights[q];

    }

  }

  /* For debugging - delete evenutally.
  for (int ele=0; ele<n_ele; ele++)
  {
    for (int dim=0; dim<n_dim; dim++)
    {
      for (int q=0; q<n_quad; q++)
      {
        for (int loc=0; loc<nodes_per_ele; loc++)
	{
	  const int dshapeIdx = ele + dim*n_ele + q*n_dim*n_ele + loc*n_quad*n_dim*n_ele;
          printf("dshape[%d,%d,%d,%d]=%f\n",ele,loc,q,dim,dshape[dshapeIdx]);
	}
      }
    }
  }*/
}

