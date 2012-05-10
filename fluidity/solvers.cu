#include "cudastatic.hpp"
#include <sys/time.h>

// Texture references for CSR matrix 
texture<int,1> tex_findrm, tex_colm;
texture<int2,1> tex_val;

// Scratchpad used by vector dot product for reduction
double* scratchpad;

// Kernel block and grid parameters - threads in a block and blocks in a grid
#define NUM_THREADS 128
#define NUM_BLOCKS 128

// Solver parameters - relative tolerance and maximum iterations
// FIXME this should really be read from the flml!
#define epsilon2 1e-14
#define IMAX 300

// For timing solver
double utime () {
  struct timeval tv;

  gettimeofday (&tv, NULL);

  return (tv.tv_sec + double (tv.tv_usec) * 1e-6);
}

__device__ int eleId(int ele, int node, int n) {
  return node*n+ele;
}

__device__ int lmatIdx(int a, int b, int c, int n) {
  return a*3*n+b*n+c;
}

// Creates a diagonal matrix stored in a vector pcmat, from the CSR matrix findrm, colm, val.
// n is the matrix size.
static __device__ double fetch_double(texture<int2,1> val, int elem)
{
  int2 v = tex1Dfetch(val, elem);
  return __hiloint2double(v.y, v.x);
}

__global__ void create_jac_sym(int n, int* findrm, int* colm, double* val, double* pcmat)
{
  int k, elem;

  for(elem=THREAD_ID; elem<n; elem+=THREAD_COUNT)
    for(k=findrm[elem]-1; k<findrm[elem+1]-1; k++)
      if(colm[k]==elem+1)
        pcmat[elem] = 1.0/val[k];
}

// Multiplies diagonal matrix mat stored as a vector by the vector src, storing result in dest.
// n is the vector length.
__global__ void diag_spmv(int n, double *mat, double *src, double *dest)
{
  int elem;

  for (elem=THREAD_ID; elem<n; elem+=THREAD_COUNT)
    dest[elem] = mat[elem]*src[elem];
}

// Sets the length-n vector vec to the zero vector.
__global__ void veczero(int n, double* vec)
{
  int elem;

  for (elem=THREAD_ID; elem<n; elem+=THREAD_COUNT)
    vec[elem] = 0;
}

// Allows fetching double values from texture memory, which only supports integers

// Multiplies the CSR matrix in texture memory tex_findrm, tex_colm, tex_val by src and stores the
// result in dest. n is the matrix size/vector length.
__global__ void csr_spmv(int n, double* src, double* dest, int *findrm)
{
  int elem;

  for (elem=THREAD_ID; elem<n; elem+=THREAD_COUNT) {
    dest[elem] = 0;
    int a=findrm[elem];
    int b=findrm[elem+1];
    for (int k=a;k<b;k++)
      dest[elem] += fetch_double(tex_val,k-1)*src[tex1Dfetch(tex_colm,k-1)-1];

  }
}

// Computes the dot product of length-n vectors vec1 and vec2. This is reduced in tmp into a
// single value per thread block. The reduced value is stored in the array partial.
__global__ void vecdot_partial(int n, double* vec1, double* vec2, double* partial)
{
  int elem;
  __shared__ double tmp[512];
  tmp[threadIdx.x] = 0;

  for (elem=THREAD_ID; elem<n; elem+=THREAD_COUNT)
    tmp[threadIdx.x] += vec1[elem]*vec2[elem];

  for (int i=blockDim.x/2;i>=1;i = i/2) {
    __syncthreads();
    if (threadIdx.x < i)
      tmp[threadIdx.x] += tmp[i + threadIdx.x];
  }

  if (threadIdx.x == 0)
    partial[blockIdx.x] = tmp[0];
}

// Reduces the output of the vecdot_partial kernel to a single value. The result is stored in result.
__global__ void vecdot_reduce(double* partial, double* result)
{
  __shared__ double tmp[NUM_BLOCKS];

  if (threadIdx.x < NUM_BLOCKS)
    tmp[threadIdx.x] = partial[threadIdx.x];
  else
    tmp[threadIdx.x] = 0;

  for (int i=blockDim.x/2;i>=1;i = i/2) {
    __syncthreads();
    if (threadIdx.x < i)
      tmp[threadIdx.x] += tmp[i + threadIdx.x];
  }

  if (threadIdx.x == 0)
    *result = tmp[0];
}

// Divides num by den and stores the result in result. This is very wasteful of the GPU.
__global__ void scalardiv(double* num, double* den, double* result)
{
  if(threadIdx.x==0 && blockIdx.x==0)
    *result = (*num)/(*den);
}

// Computes r= a*x+y for n-length vectors x and y, and scalar a.
__global__ void axpy(int n, double* a, double* x, double* y, double* r)
{
  int elem;

  for (elem=THREAD_ID; elem<n; elem+=THREAD_COUNT)
    r[elem] = y[elem] + (*a)*x[elem];
}

// Computes y= y-a*x for n-length vectors x and y, and scalar a.
__global__ void ymax(int n, double* a, double* x, double* y)
{
  int elem;

  for (elem=THREAD_ID; elem<n; elem+=THREAD_COUNT)
    y[elem] = y[elem] - (*a)*x[elem];
}

// Convenient function for performing a vector dot product and reduce all in one go.
void vecdot(int n, double* vec1, double* vec2, double* result)
{
  dim3 BlockDim(NUM_THREADS);
  dim3 GridDim(NUM_BLOCKS);

  vecdot_partial<<<GridDim,BlockDim>>>(n, vec1, vec2, scratchpad);
  vecdot_reduce<<<1,NUM_BLOCKS>>>(scratchpad, result);
}

// Sets dest=src for scalars on the GPU.
void scalarassign(double* dest, double* src)
{
  cudaMemcpy(dest, src, sizeof(double), cudaMemcpyDeviceToDevice);
}

// Sets dest=src for n-length vectors on the GPU.
void vecassign(double *dest, double *src, int n)
{
  cudaMemcpy(dest, src, sizeof(double)*n, cudaMemcpyDeviceToDevice);
}


//void cg_solve(int* findrm, int size_findrm, int* colm, int size_colm, double* k_val, double* b, int rhs_size, double *x)

void cg_solve(int* k_findrm, int size_findrm, int* k_colm, int size_colm, double* k_val, double* k_b, int rhs_val_size, double *x_p)
{
  // Vectors on the GPU
  double
    //*k_x, *k_r,
    *k_d, *k_q, *k_s;
  // Diagonal matrix on the GPU (stored as a vector)
  double* k_jac;
  // Scalars on the GPU
  double  *k_alpha, *k_snew, *k_beta, *k_sold, *k_s0;

  // Scalars on the host
  double s0, snew;
  int iterations = 0;

  // Allocate space on the GPU for the CSR matrix and RHS vector, and copy from host to GPU
  cudaBindTexture(NULL, tex_colm, k_colm, sizeof(int)*(size_colm));

  // Allocate space for vectors on the GPU
  //cudaMalloc((void**)&k_x, sizeof(double)*(*rhs_val_size));
  //cudaMalloc((void**)&k_r, sizeof(double)*(*rhs_val_size));
  cudaMalloc((void**)&k_s, sizeof(double)*(rhs_val_size));
  cudaMalloc((void**)&k_d, sizeof(double)*(rhs_val_size));
  cudaMalloc((void**)&k_q, sizeof(double)*(rhs_val_size));
  cudaMalloc((void**)&k_jac, sizeof(double)*(rhs_val_size));
  cudaMalloc((void**)&k_alpha, sizeof(double));
  cudaMalloc((void**)&scratchpad, sizeof(double)*NUM_BLOCKS);
  cudaMalloc((void**)&k_snew, sizeof(double));
  cudaMalloc((void**)&k_sold, sizeof(double));
  cudaMalloc((void**)&k_beta, sizeof(double));
  cudaMalloc((void**)&k_s0, sizeof(double));

  // Dimensions of blocks and grid on the GPU
  dim3 BlockDim(NUM_THREADS);
  dim3 GridDim(NUM_BLOCKS);

  // Create diagonal preconditioning matrix (J = 1/diag(M)) 
  create_jac_sym<<<GridDim,BlockDim>>>(rhs_val_size, k_findrm, k_colm, k_val, k_jac);
  //  printd("jac", k_jac, 1000);
  // Bind the matrix to the texture cache - this was not done earlier as we modified the matrix
  cudaBindTexture(NULL, tex_val, k_val, sizeof(double)*(size_colm));

  // Initialise result vector (x=0)
  veczero<<<GridDim,BlockDim>>>(rhs_val_size, x_p);

  // r=b-Ax (r=b since x=0), and d=M^(-1)r
  //cudaMemcpy(k_r, k_b, sizeof(double)*(*rhs_val_size), cudaMemcpyDeviceToDevice);
  //cudaMemcpy(k_d, k_r, sizeof(double)*(*rhs_val_size), cudaMemcpyDeviceToDevice);
  diag_spmv<<<GridDim,BlockDim>>>(rhs_val_size, k_jac, k_b, k_d);

  // s0 = r.d
  vecdot(rhs_val_size, k_b, k_d, k_s0);
  // snew = s0
  scalarassign(k_snew, k_s0);

  // Copy snew and s0 back to host so that host can evaluate stopping condition
  cudaMemcpy(&snew, k_snew, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&s0, k_s0, sizeof(double), cudaMemcpyDeviceToHost);
  // While i < imax and snew > epsilon^2*s0
  while (iterations < IMAX && snew > epsilon2*s0)
  {
    // q = Ad
    csr_spmv<<<GridDim,BlockDim>>>(rhs_val_size, k_d, k_q, k_findrm);
    // alpha = snew/(d.q)
    vecdot(rhs_val_size, k_d, k_q, k_alpha);
    scalardiv<<<1,1>>>(k_snew, k_alpha, k_alpha);
    // x = x + alpha*d
    axpy<<<GridDim,BlockDim>>>(rhs_val_size, k_alpha, k_d, x_p, x_p);
    // r = r - alpha*q
    ymax<<<GridDim,BlockDim>>>(rhs_val_size, k_alpha, k_q, k_b);
    // s = M^(-1)r
    diag_spmv<<<GridDim,BlockDim>>>(rhs_val_size, k_jac, k_b, k_s);
    // sold = snew
    scalarassign(k_sold, k_snew);
    // snew = r.s
    vecdot(rhs_val_size, k_b, k_s, k_snew);
    // beta = snew/sold
    scalardiv<<<1,1>>>(k_snew, k_sold, k_beta);
    // d = s + beta*d
    axpy<<<GridDim,BlockDim>>>(rhs_val_size, k_beta, k_d, k_s, k_d);
    // Copy back snew so the host can evaluate the stopping condition
    cudaMemcpy(&snew, k_snew, sizeof(double), cudaMemcpyDeviceToHost);
    // i = i+1
    iterations++;
  }

  cudaUnbindTexture(tex_colm);
  cudaUnbindTexture(tex_val);

  cudaFree(k_s);
  cudaFree(k_d);
  cudaFree(k_q);
  cudaFree(k_jac);
  cudaFree(k_alpha);
  cudaFree(k_snew);
  cudaFree(k_sold);
  cudaFree(k_beta);
  cudaFree(k_s0);
  cudaFree(scratchpad);

}

// LMA Addtions

__device__ int eleId(int ele, int node, int n) {
  return node*n+ele;
}

__device__ int lmatIdx(int a, int b, int c, int n) {
  return a*3*n+b*n+c;
}

// This definitely gives me the diagonal of the BD local matrix
__global__ void extract_diagonal(int n_ele, double *matrix, double *jac_tmp)
{
  for(int i=THREAD_ID; i<n_ele; i+=THREAD_COUNT) {
    jac_tmp[i*3  ] = matrix[lmatIdx(0,0,i,n_ele)];
    jac_tmp[i*3+1] = matrix[lmatIdx(1,1,i,n_ele)];
    jac_tmp[i*3+2] = matrix[lmatIdx(2,2,i,n_ele)];
  }
}

__global__ void create_jac_pc(int n, int *findrm, int *colm, double *jac, double *jac_tmp) {
  for(int row=THREAD_ID; row<n; row+=THREAD_COUNT) {
    jac[row] = 0;
    int a=findrm[row];
    int b=findrm[row+1];
    for(int k=a;k<b;k++) {
      jac[row] += jac_tmp[colm[k-1]-1];
    }
    jac[row] = 1.0/jac[row];
  }
}

__global__ void spmv_stage1_2(int n_ele, double *matrix, double *src, double *temp2, int *node_nums)
{

  //Compute M*s*b
  for(int ele=THREAD_ID; ele<n_ele; ele+=THREAD_COUNT) {

    double tmpa = src[node_nums[eleId(ele,0,n_ele)]-1];
    double tmpb = src[node_nums[eleId(ele,1,n_ele)]-1];
    double tmpc = src[node_nums[eleId(ele,2,n_ele)]-1];
    temp2[ele*3]   = matrix[lmatIdx(0,0,ele,n_ele)]*tmpa
                   + matrix[lmatIdx(0,1,ele,n_ele)]*tmpb
                   + matrix[lmatIdx(0,2,ele,n_ele)]*tmpc;
    temp2[ele*3+1] = matrix[lmatIdx(1,0,ele,n_ele)]*tmpa
                   + matrix[lmatIdx(1,1,ele,n_ele)]*tmpb
                   + matrix[lmatIdx(1,2,ele,n_ele)]*tmpc;
    temp2[ele*3+2] = matrix[lmatIdx(2,0,ele,n_ele)]*tmpa
                   + matrix[lmatIdx(2,1,ele,n_ele)]*tmpb
                   + matrix[lmatIdx(2,2,ele,n_ele)]*tmpc;
  }
}

__global__ void spmv_stage3(int nodes, double *temp2, double *dest, int *findrm, int *colm)
{
  // Compute s^T*M*s*b
  for(int row=THREAD_ID; row<nodes; row+=THREAD_COUNT) {
    dest[row] = 0;
    int a=findrm[row];
    int b=findrm[row+1];
    for(int k=a;k<b;k++)
      dest[row] += temp2[colm[k-1]-1];
  }
}

void cg_solve_lma(int* k_findrm, int size_findrm, int* k_colm, int size_colm, double* k_val, double* k_b, int rhs_val_size, double *x_p)
{
  // Vectors on the GPU
  double *k_d, *k_q, *k_s;
  double *temp1, *temp2, *k_jac_tmp;

  // Diagonal matrix on the GPU (stored as a vector)
  double* k_jac;

  // Scalars on the GPU
  double  *k_alpha, *k_snew, *k_beta, *k_sold, *k_s0;

  // Scalars on the host
  double s0, snew;
  int iterations = 0;

  // Allocate space for vectors on the GPU
  cudaMalloc((void**)&k_s, sizeof(double)*(rhs_val_size));
  cudaMalloc((void**)&k_d, sizeof(double)*(rhs_val_size));
  cudaMalloc((void**)&k_q, sizeof(double)*(rhs_val_size));
  cudaMalloc((void**)&k_jac, sizeof(double)*(rhs_val_size));
  cudaMalloc((void**)&k_alpha, sizeof(double));
  cudaMalloc((void**)&scratchpad, sizeof(double)*NUM_BLOCKS);
  cudaMalloc((void**)&k_snew, sizeof(double));
  cudaMalloc((void**)&k_sold, sizeof(double));
  cudaMalloc((void**)&k_beta, sizeof(double));
  cudaMalloc((void**)&k_s0, sizeof(double));
  cudaMalloc((void**)&temp1, sizeof(double)*n_ele*3);
  cudaMalloc((void**)&temp2, sizeof(double)*n_ele*3);
  cudaMalloc((void**)&k_jac_tmp, sizeof(double)*n_ele*3);


  // Dimensions of blocks and grid on the GPU
  dim3 BlockDim(NUM_THREADS);
  dim3 GridDim(NUM_BLOCKS);

  // Create diagonal preconditioning matrix (J = 1/diag(M)) 
  extract_diagonal<<<GridDim,BlockDim>>>(n_ele, matrix, k_jac_tmp);
  create_jac_pc<<<GridDim,BlockDim>>>(rhs_val_size, k_findrm, k_colm, k_jac, k_jac_tmp);

  // Initialise result vector (x=0)
  veczero<<<GridDim,BlockDim>>>(rhs_val_size, x_p);

  // r=b-Ax (r=b since x=0), and d=M^(-1)r
  diag_spmv<<<GridDim,BlockDim>>>(rhs_val_size, k_jac, k_b, k_d);

  // s0 = r.d
  vecdot(rhs_val_size, k_b, k_d, k_s0);
  // snew = s0
  scalarassign(k_snew, k_s0);

  // Copy snew and s0 back to host so that host can evaluate stopping condition
  cudaMemcpy(&snew, k_snew, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&s0, k_s0, sizeof(double), cudaMemcpyDeviceToHost);
  
  // While i < imax and snew > epsilon^2*s0
  while (iterations < IMAX && snew > epsilon2*s0)
  {
    // q = Ad
    spmv_stage1_2<<<GridDim,BlockDim>>>(n_ele, matrix, k_d, temp2, node_nums);
    spmv_stage3<<<GridDim,BlockDim>>>(rhs_val_size, temp2, k_q, k_findrm, k_colm);
    // alpha = snew/(d.q)
    vecdot(rhs_val_size, k_d, k_q, k_alpha);
    scalardiv<<<1,1>>>(k_snew, k_alpha, k_alpha);
    // x = x + alpha*d
    axpy<<<GridDim,BlockDim>>>(rhs_val_size, k_alpha, k_d, x_p, x_p);
    // r = r - alpha*q
    ymax<<<GridDim,BlockDim>>>(rhs_val_size, k_alpha, k_q, k_b);
    // s = M^(-1)r
    diag_spmv<<<GridDim,BlockDim>>>(rhs_val_size, k_jac, k_b, k_s);
    // sold = snew
    scalarassign(k_sold, k_snew);
    // snew = r.s
    vecdot(rhs_val_size, k_b, k_s, k_snew);
    // beta = snew/sold
    scalardiv<<<1,1>>>(k_snew, k_sold, k_beta);
    // d = s + beta*d
    axpy<<<GridDim,BlockDim>>>(rhs_val_size, k_beta, k_d, k_s, k_d);
    // Copy back snew so the host can evaluate the stopping condition
    cudaMemcpy(&snew, k_snew, sizeof(double), cudaMemcpyDeviceToHost);
    // i = i+1
    iterations++;
  }

  cudaFree(k_s);
  cudaFree(k_d);
  cudaFree(k_q);
  cudaFree(k_jac);
  cudaFree(k_alpha);
  cudaFree(k_snew);
  cudaFree(k_sold);
  cudaFree(k_beta);
  cudaFree(k_s0);
  cudaFree(scratchpad);
  cudaFree(temp1);
  cudaFree(temp2);
  cudaFree(k_jac_tmp);

}

