MOOOOOSE
#include "cudastatic.hpp"
#include "cudastate.hpp"
double* localVector;
double* localMatrix;
double* globalVector;
double* globalMatrix;
double* solutionVector;
int matrix_colm_size;
int matrix_findrm_size;
int* matrix_colm;
int* matrix_findrm;


__global__ void A(double* localTensor, int n_ele, double dt, double* detwei, double* CG1)
{
  for(int i_ele = THREAD_ID; i_ele < n_ele; (i_ele += THREAD_COUNT))
  {
    for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
    {
      for(int i_r_1 = 0; i_r_1 < 3; i_r_1++)
      {
        localTensor[((i_ele + (n_ele * i_r_0)) + (3 * (n_ele * i_r_1)))] = 0.0;
        for(int i_g = 0; i_g < 6; i_g++)
        {
          localTensor[((i_ele + (n_ele * i_r_0)) + (3 * (n_ele * i_r_1)))] += ((CG1[(i_r_0 + (3 * i_g))] * CG1[(i_r_1 + (3 * i_g))]) * detwei[(i_ele + (n_ele * i_g))]);
        };
      };
    };
  };
}

__global__ void RHS(double* localTensor, int n_ele, double dt, double* detwei, double* c0, double* CG1)
{
  for(int i_ele = THREAD_ID; i_ele < n_ele; (i_ele += THREAD_COUNT))
  {
    __shared__ double c_q0[6];
    for(int i_g = 0; i_g < 6; i_g++)
    {
      c_q0[i_g] = 0.0;
      for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
      {
        c_q0[i_g] += (c0[(i_ele + (n_ele * i_r_0))] * CG1[(i_r_0 + (3 * i_g))]);
      };
    };
    for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
    {
      localTensor[(i_ele + (n_ele * i_r_0))] = 0.0;
      for(int i_g = 0; i_g < 6; i_g++)
      {
        localTensor[(i_ele + (n_ele * i_r_0))] += ((CG1[(i_r_0 + (3 * i_g))] * c_q0[i_g]) * detwei[(i_ele + (n_ele * i_g))]);
      };
    };
  };
}

StateHolder* state;
extern "C" void initialise_gpu_()
{
  state = new StateHolder();
  state -> initialise();
  state -> extractField("Tracer", 0);
  state -> allocateAllGPUMemory();
  state -> transferAllFields();
  state -> insertTemporaryField("p", "Tracer");
  int numEle = (state -> getNumEle());
  int numNodes = (state -> getNumNodes());
  CsrSparsity* sparsity = (state -> getSparsity("Tracer"));
  matrix_colm = (sparsity -> getCudaColm());
  matrix_findrm = (sparsity -> getCudaFindrm());
  matrix_colm_size = (sparsity -> getSizeColm());
  matrix_findrm_size = (sparsity -> getSizeFindrm());
  int numValsPerNode = (state -> getValsPerNode("Tracer"));
  int numVectorEntries = (state -> getNodesPerEle("Tracer"));
  numVectorEntries = (numVectorEntries * numValsPerNode);
  int numMatrixEntries = (numVectorEntries * numVectorEntries);
  cudaMalloc((void**)(&localVector), (sizeof(double) * (numEle * numVectorEntries)));
  cudaMalloc((void**)(&localMatrix), (sizeof(double) * (numEle * numMatrixEntries)));
  cudaMalloc((void**)(&globalVector), (sizeof(double) * (numNodes * numValsPerNode)));
  cudaMalloc((void**)(&globalMatrix), (sizeof(double) * matrix_colm_size));
  cudaMalloc((void**)(&solutionVector), (sizeof(double) * (numNodes * numValsPerNode)));
}

extern "C" void finalise_gpu_()
{
  delete state;
}

extern "C" void run_model_(double* dt_pointer)
{
  double dt = *dt_pointer;
  int numEle = (state -> getNumEle());
  int numNodes = (state -> getNumNodes());
  double* detwei = (state -> getDetwei());
  int* eleNodes = (state -> getEleNodes());
  double* coordinates = (state -> getCoordinates());
  double* dn = (state -> getReferenceDn());
  double* quadWeights = (state -> getQuadWeights());
  int nDim = (state -> getDimension("Coordinate"));
  int nQuad = (state -> getNumQuadPoints("Coordinate"));
  int nodesPerEle = (state -> getNodesPerEle("Coordinate"));
  double* shape = (state -> getBasisFunction("Coordinate"));
  double* dShape = (state -> getBasisFunctionDerivative("Coordinate"));
  int blockXDim = 1;
  int gridXDim = 1;
  int shMemSize = t2p_shmemsize(blockXDim, nDim, nodesPerEle);
  transform_to_physical<<<gridXDim,blockXDim,shMemSize>>>(coordinates, dn, quadWeights, dShape, detwei, numEle, nDim, nQuad, nodesPerEle);
  A<<<gridXDim,blockXDim>>>(localMatrix, numEle, dt, detwei, shape);
  double* TracerCoeff = (state -> getElementValue("Tracer"));
  RHS<<<gridXDim,blockXDim>>>(localVector, numEle, dt, detwei, TracerCoeff, shape);
  int numValsPerNode = (state -> getValsPerNode("Tracer"));
  cudaMemset(globalMatrix, 0, (sizeof(double) * matrix_colm_size));
  cudaMemset(globalVector, 0, (sizeof(double) * (numValsPerNode * numNodes)));
  matrix_addto<<<gridXDim,blockXDim>>>(matrix_findrm, matrix_colm, globalMatrix, eleNodes, localMatrix, numEle, nodesPerEle);
  vector_addto<<<gridXDim,blockXDim>>>(globalVector, eleNodes, localVector, numEle, nodesPerEle);
  cg_solve(matrix_findrm, matrix_findrm_size, matrix_colm, matrix_colm_size, globalMatrix, globalVector, numNodes, solutionVector);
  double* pCoeff = (state -> getElementValue("p"));
  expand_data<<<gridXDim,blockXDim>>>(pCoeff, solutionVector, eleNodes, numEle, numValsPerNode, nodesPerEle);
  state -> returnFieldToHost("Tracer", "p");
}



