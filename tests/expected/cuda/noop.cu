#include "cudastatic.hpp"
#include "cudastate.hpp"
double* localVector;
double* localMatrix;
double* globalVector;
double* globalMatrix;
double* solutionVector;


StateHolder* state;
extern "C" void initialise_gpu_()
{
  state = new StateHolder();
  state->initialise();
  state->extractField("Tracer", 0);
  state->allocateAllGPUMemory();
  state->transferAllFields();
  int numEle = state->getNumEle();
  int numNodes = state->getNumNodes();
}

extern "C" void finalise_gpu_()
{
  delete state;
}

extern "C" void run_model_(double* dt_pointer)
{
  double dt = *dt_pointer;
  int numEle = state->getNumEle();
  int numNodes = state->getNumNodes();
  double* detwei = state->getDetwei();
  int* eleNodes = state->getEleNodes();
  double* coordinates = state->getCoordinates();
  double* dn = state->getReferenceDn();
  double* quadWeights = state->getQuadWeights();
  int nDim = state->getDimension("Coordinate");
  int nQuad = state->getNumQuadPoints("Coordinate");
  int nodesPerEle = state->getNodesPerEle("Coordinate");
  double* shape = state->getBasisFunction("Coordinate");
  double* dShape = state->getBasisFunctionDerivative("Coordinate");
  int blockXDim = 64;
  int gridXDim = 128;
  int shMemSize = t2p_shmemsize(blockXDim, nDim, nodesPerEle);
  transform_to_physical<<<gridXDim,blockXDim,shMemSize>>>(coordinates, dn, quadWeights, dShape, detwei, numEle, nDim, nQuad, nodesPerEle);
}



