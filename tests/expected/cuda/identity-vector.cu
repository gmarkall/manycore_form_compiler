#include "cudastatic.hpp"
#include "cudastate.hpp"
double* localVector;
double* localMatrix;
double* globalVector;
double* globalMatrix;
double* solutionVector;
int* Velocity_findrm;
int Velocity_findrm_size;
int* Velocity_colm;
int Velocity_colm_size;


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
          for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
          {
            localTensor[((i_ele + (n_ele * i_r_0)) + (3 * (n_ele * i_r_1)))] += ((CG1[(i_r_0 + (3 * i_g))] * CG1[(i_r_1 + (3 * i_g))]) * detwei[(i_ele + (n_ele * i_g))]);
          };
        };
      };
    };
  };
}

__global__ void RHS(double* localTensor, int n_ele, double dt, double* detwei, double* c0, double* CG1)
{
  for(int i_ele = THREAD_ID; i_ele < n_ele; (i_ele += THREAD_COUNT))
  {
    double c_q0[12];
    for(int i_g = 0; i_g < 6; i_g++)
    {
      for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
      {
        c_q0[(i_g + (6 * i_d_0))] = 0.0;
        for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
        {
          c_q0[(i_g + (6 * i_d_0))] += (c0[((i_ele + (n_ele * i_d_0)) + (2 * (n_ele * i_r_0)))] * CG1[(i_r_0 + (3 * i_g))]);
        };
      };
    };
    for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
    {
      localTensor[(i_ele + (n_ele * i_r_0))] = 0.0;
      for(int i_g = 0; i_g < 6; i_g++)
      {
        for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
        {
          localTensor[(i_ele + (n_ele * i_r_0))] += ((CG1[(i_r_0 + (3 * i_g))] * c_q0[(i_g + (6 * i_d_0))]) * detwei[(i_ele + (n_ele * i_g))]);
        };
      };
    };
  };
}

StateHolder* state;
extern "C" void initialise_gpu_()
{
  state = new StateHolder();
  state -> initialise();
  state -> extractField("Velocity", 1);
  state -> allocateAllGPUMemory();
  state -> transferAllFields();
  int numEle = (state -> getNumEle());
  int numNodes = (state -> getNumNodes());
  CsrSparsity* Velocity_sparsity = (state -> getSparsity("Velocity"));
  Velocity_colm = (Velocity_sparsity -> getCudaColm());
  Velocity_findrm = (Velocity_sparsity -> getCudaFindrm());
  Velocity_colm_size = (Velocity_sparsity -> getSizeColm());
  Velocity_findrm_size = (Velocity_sparsity -> getSizeFindrm());
  int numValsPerNode = (state -> getValsPerNode("Velocity"));
  int numVectorEntries = (state -> getNodesPerEle("Velocity"));
  numVectorEntries = (numVectorEntries * numValsPerNode);
  int numMatrixEntries = (numVectorEntries * numVectorEntries);
  cudaMalloc((void**)(&localVector), (sizeof(double) * (numEle * numVectorEntries)));
  cudaMalloc((void**)(&localMatrix), (sizeof(double) * (numEle * numMatrixEntries)));
  cudaMalloc((void**)(&globalVector), (sizeof(double) * (numNodes * numValsPerNode)));
  cudaMalloc((void**)(&globalMatrix), (sizeof(double) * Velocity_colm_size));
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
  int blockXDim = 64;
  int gridXDim = 128;
  int shMemSize = t2p_shmemsize(blockXDim, nDim, nodesPerEle);
  transform_to_physical<<<gridXDim,blockXDim,shMemSize>>>(coordinates, dn, quadWeights, dShape, detwei, numEle, nDim, nQuad, nodesPerEle);
  A<<<gridXDim,blockXDim>>>(localMatrix, numEle, dt, detwei, shape);
  double* VelocityCoeff = (state -> getElementValue("Velocity"));
  RHS<<<gridXDim,blockXDim>>>(localVector, numEle, dt, detwei, VelocityCoeff, shape);
  cudaMemset(globalMatrix, 0, (sizeof(double) * Velocity_colm_size));
  cudaMemset(globalVector, 0, (sizeof(double) * ((state -> getValsPerNode("Velocity")) * numNodes)));
  matrix_addto<<<gridXDim,blockXDim>>>(Velocity_findrm, Velocity_colm, globalMatrix, eleNodes, localMatrix, numEle, nodesPerEle);
  vector_addto<<<gridXDim,blockXDim>>>(globalVector, eleNodes, localVector, numEle, nodesPerEle);
  cg_solve(Velocity_findrm, Velocity_findrm_size, Velocity_colm, Velocity_colm_size, globalMatrix, globalVector, numNodes, solutionVector);
  expand_data<<<gridXDim,blockXDim>>>(VelocityCoeff, solutionVector, eleNodes, numEle, (state -> getValsPerNode("Velocity")), nodesPerEle);
  state -> returnFieldToHost("Velocity");
}



