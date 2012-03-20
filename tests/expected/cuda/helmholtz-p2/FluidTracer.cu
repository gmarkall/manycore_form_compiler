// Generated by the Manycore Form Compiler.
// https://github.com/gmarkall/manycore_form_compiler


#include "cudastatic.hpp"
#include "cudastate.hpp"
double* localVector;
double* localMatrix;
double* globalVector;
double* globalMatrix;
double* solutionVector;
int* Tracer_findrm;
int Tracer_findrm_size;
int* Tracer_colm;
int Tracer_colm_size;


__global__ void A(int n_ele, double* localTensor, double dt, double* c0)
{
  const double CG2[6][6] = { { -0.0748038077481965,-0.0748038077481965,
                               0.5176323419876725,-0.0482083778155121,
                              -0.0482083778155121,-0.084730493093978  },
                             {  0.0335448115231485, 0.299215230992786 ,
                               0.299215230992786 , 0.7954802262009056,
                               0.1928335112620482, 0.1928335112620482 },
                             { -0.0748038077481965, 0.5176323419876725,
                              -0.0748038077481965,-0.0482083778155121,
                              -0.084730493093978 ,-0.0482083778155121 },
                             {  0.299215230992786 , 0.0335448115231485,
                               0.299215230992786 , 0.1928335112620482,
                               0.7954802262009056, 0.1928335112620482 },
                             {  0.299215230992786 , 0.299215230992786 ,
                               0.0335448115231485, 0.1928335112620482,
                               0.1928335112620482, 0.7954802262009056 },
                             {  0.5176323419876725,-0.0748038077481965,
                              -0.0748038077481965,-0.084730493093978 ,
                              -0.0482083778155121,-0.0482083778155121 } };
  const double d_CG1[3][6][2] = { { {  1., 0. },
                                   {  1., 0. },
                                   {  1., 0. },
                                   {  1., 0. },
                                   {  1., 0. },
                                   {  1., 0. } },

                                  { {  0., 1. },
                                   {  0., 1. },
                                   {  0., 1. },
                                   {  0., 1. },
                                   {  0., 1. },
                                   {  0., 1. } },

                                  { { -1.,-1. },
                                   { -1.,-1. },
                                   { -1.,-1. },
                                   { -1.,-1. },
                                   { -1.,-1. },
                                   { -1.,-1. } } };
  const double d_CG2[6][6][2] = { { { -0.633695145960917 ,-0.                 },
                                   { -0.633695145960917 ,-0.                 },
                                   {  2.2673902919218341, 0.                 },
                                   {  0.7837939636638596,-0.                 },
                                   {  0.7837939636638596,-0.                 },
                                   { -0.5675879273277191,-0.                 } },

                                  { {  0.366304854039083 , 0.366304854039083  },
                                   {  3.2673902919218341, 0.366304854039083  },
                                   {  0.366304854039083 ,
                                     3.2673902919218341 },
                                   {  1.7837939636638596,
                                     1.7837939636638596 },
                                   {  0.4324120726722809,
                                     1.7837939636638596 },
                                   {  1.7837939636638596,
                                     0.4324120726722809 } },

                                  { { -0.                ,-0.633695145960917  },
                                   {  0.                ,
                                     2.2673902919218341 },
                                   { -0.                ,-0.633695145960917  },
                                   { -0.                ,
                                     0.7837939636638596 },
                                   { -0.                ,
                                    -0.5675879273277191 },
                                   { -0.                ,
                                     0.7837939636638596 } },

                                  { {  2.9010854378827511,-0.366304854039083  },
                                   {  0.                ,-0.366304854039083  },
                                   { -2.9010854378827511,
                                    -3.2673902919218341 },
                                   { -1.3513818909915787,
                                    -1.7837939636638596 },
                                   {  0.                ,
                                    -1.7837939636638596 },
                                   {  1.3513818909915787,
                                    -0.4324120726722809 } },

                                  { { -0.366304854039083 ,
                                     2.9010854378827511 },
                                   { -3.2673902919218341,
                                    -2.9010854378827511 },
                                   { -0.366304854039083 , 0.                 },
                                   { -1.7837939636638596,
                                    -1.3513818909915787 },
                                   { -0.4324120726722809,
                                     1.3513818909915787 },
                                   { -1.7837939636638596, 0.                 } },

                                  { { -2.2673902919218341,
                                    -2.2673902919218341 },
                                   {  0.633695145960917 , 0.633695145960917  },
                                   {  0.633695145960917 , 0.633695145960917  },
                                   {  0.5675879273277191,
                                     0.5675879273277191 },
                                   { -0.7837939636638596,
                                    -0.7837939636638596 },
                                   { -0.7837939636638596,
                                    -0.7837939636638596 } } };
  const double w[6] = {  0.0549758718276609, 0.0549758718276609,
                         0.0549758718276609, 0.1116907948390057,
                         0.1116907948390057, 0.1116907948390057 };
  double c_q0[6][2][2];
  for(int i_ele = THREAD_ID; i_ele < n_ele; i_ele += THREAD_COUNT)
  {
    for(int i_g = 0; i_g < 6; i_g++)
    {
      for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
      {
        for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
        {
          c_q0[i_g][i_d_0][i_d_1] = 0.0;
          for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
          {
            c_q0[i_g][i_d_0][i_d_1] += c0[i_ele + n_ele * (i_d_0 + 2 * q_r_0)] * d_CG1[q_r_0][i_g][i_d_1];
          };
        };
      };
    };
    for(int i_r_0 = 0; i_r_0 < 6; i_r_0++)
    {
      for(int i_r_1 = 0; i_r_1 < 6; i_r_1++)
      {
        localTensor[i_ele + n_ele * (i_r_0 + 6 * i_r_1)] = 0.0;
        for(int i_g = 0; i_g < 6; i_g++)
        {
          double ST1 = 0.0;
          double ST0 = 0.0;
          double ST2 = 0.0;
          ST1 += -1 * CG2[i_r_0][i_g] * CG2[i_r_1][i_g];
          double l95[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
          double l35[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
          ST2 += c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0];
          for(int i_d_5 = 0; i_d_5 < 2; i_d_5++)
          {
            for(int i_d_3 = 0; i_d_3 < 2; i_d_3++)
            {
              for(int i_d_9 = 0; i_d_9 < 2; i_d_9++)
              {
                ST0 += (l35[i_d_3][i_d_5] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG2[i_r_0][i_g][i_d_3] * (l95[i_d_9][i_d_5] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG2[i_r_1][i_g][i_d_9];
              };
            };
          };
          localTensor[i_ele + n_ele * (i_r_0 + 6 * i_r_1)] += ST2 * (ST0 + ST1) * w[i_g];
        };
      };
    };
  };
}

__global__ void RHS(int n_ele, double* localTensor, double dt, double* c0, double* c1)
{
  const double CG2[6][6] = { { -0.0748038077481965,-0.0748038077481965,
                               0.5176323419876725,-0.0482083778155121,
                              -0.0482083778155121,-0.084730493093978  },
                             {  0.0335448115231485, 0.299215230992786 ,
                               0.299215230992786 , 0.7954802262009056,
                               0.1928335112620482, 0.1928335112620482 },
                             { -0.0748038077481965, 0.5176323419876725,
                              -0.0748038077481965,-0.0482083778155121,
                              -0.084730493093978 ,-0.0482083778155121 },
                             {  0.299215230992786 , 0.0335448115231485,
                               0.299215230992786 , 0.1928335112620482,
                               0.7954802262009056, 0.1928335112620482 },
                             {  0.299215230992786 , 0.299215230992786 ,
                               0.0335448115231485, 0.1928335112620482,
                               0.1928335112620482, 0.7954802262009056 },
                             {  0.5176323419876725,-0.0748038077481965,
                              -0.0748038077481965,-0.084730493093978 ,
                              -0.0482083778155121,-0.0482083778155121 } };
  const double d_CG1[3][6][2] = { { {  1., 0. },
                                   {  1., 0. },
                                   {  1., 0. },
                                   {  1., 0. },
                                   {  1., 0. },
                                   {  1., 0. } },

                                  { {  0., 1. },
                                   {  0., 1. },
                                   {  0., 1. },
                                   {  0., 1. },
                                   {  0., 1. },
                                   {  0., 1. } },

                                  { { -1.,-1. },
                                   { -1.,-1. },
                                   { -1.,-1. },
                                   { -1.,-1. },
                                   { -1.,-1. },
                                   { -1.,-1. } } };
  const double w[6] = {  0.0549758718276609, 0.0549758718276609,
                         0.0549758718276609, 0.1116907948390057,
                         0.1116907948390057, 0.1116907948390057 };
  double c_q1[6];
  double c_q0[6][2][2];
  for(int i_ele = THREAD_ID; i_ele < n_ele; i_ele += THREAD_COUNT)
  {
    for(int i_g = 0; i_g < 6; i_g++)
    {
      c_q1[i_g] = 0.0;
      for(int q_r_0 = 0; q_r_0 < 6; q_r_0++)
      {
        c_q1[i_g] += c1[i_ele + n_ele * q_r_0] * CG2[q_r_0][i_g];
      };
      for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
      {
        for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
        {
          c_q0[i_g][i_d_0][i_d_1] = 0.0;
          for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
          {
            c_q0[i_g][i_d_0][i_d_1] += c0[i_ele + n_ele * (i_d_0 + 2 * q_r_0)] * d_CG1[q_r_0][i_g][i_d_1];
          };
        };
      };
    };
    for(int i_r_0 = 0; i_r_0 < 6; i_r_0++)
    {
      localTensor[i_ele + n_ele * i_r_0] = 0.0;
      for(int i_g = 0; i_g < 6; i_g++)
      {
        double ST3 = 0.0;
        ST3 += CG2[i_r_0][i_g] * c_q1[i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
        localTensor[i_ele + n_ele * i_r_0] += ST3 * w[i_g];
      };
    };
  };
}

StateHolder* state;
extern "C" void initialise_gpu_()
{
  state = new StateHolder();
  state->initialise();
  state->extractField("Tracer", 0);
  state->extractField("Coordinate", 1);
  state->allocateAllGPUMemory();
  state->transferAllFields();
  int numEle = state->getNumEle();
  int numNodes = state->getNumNodes();
  CsrSparsity* Tracer_sparsity = state->getSparsity("Tracer");
  Tracer_colm = Tracer_sparsity->getCudaColm();
  Tracer_findrm = Tracer_sparsity->getCudaFindrm();
  Tracer_colm_size = Tracer_sparsity->getSizeColm();
  Tracer_findrm_size = Tracer_sparsity->getSizeFindrm();
  int numValsPerNode = state->getValsPerNode("Tracer");
  int numVectorEntries = state->getNodesPerEle("Tracer");
  numVectorEntries = numVectorEntries * numValsPerNode;
  int numMatrixEntries = numVectorEntries * numVectorEntries;
  cudaMalloc((void**)(&localVector), sizeof(double) * numEle * numVectorEntries);
  cudaMalloc((void**)(&localMatrix), sizeof(double) * numEle * numMatrixEntries);
  cudaMalloc((void**)(&globalVector), sizeof(double) * numNodes * numValsPerNode);
  cudaMalloc((void**)(&globalMatrix), sizeof(double) * Tracer_colm_size);
  cudaMalloc((void**)(&solutionVector), sizeof(double) * numNodes * numValsPerNode);
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
  int* eleNodes = state->getEleNodes();
  int blockXDim = 64;
  int gridXDim = 128;
  double* CoordinateCoeff = state->getElementValue("Coordinate");
  A<<<gridXDim,blockXDim>>>(numEle, localMatrix, dt, CoordinateCoeff);
  double* TracerCoeff = state->getElementValue("Tracer");
  RHS<<<gridXDim,blockXDim>>>(numEle, localVector, dt, CoordinateCoeff, TracerCoeff);
  cudaMemset(globalMatrix, 0, sizeof(double) * Tracer_colm_size);
  cudaMemset(globalVector, 0, sizeof(double) * state->getValsPerNode("Tracer") * numNodes);
  matrix_addto<<<gridXDim,blockXDim>>>(Tracer_findrm, Tracer_colm, globalMatrix, eleNodes, localMatrix, numEle, state->getNodesPerEle("Tracer"));
  vector_addto<<<gridXDim,blockXDim>>>(globalVector, eleNodes, localVector, numEle, state->getNodesPerEle("Tracer"));
  cg_solve(Tracer_findrm, Tracer_findrm_size, Tracer_colm, Tracer_colm_size, globalMatrix, globalVector, numNodes, solutionVector);
  expand_data<<<gridXDim,blockXDim>>>(TracerCoeff, solutionVector, eleNodes, numEle, state->getValsPerNode("Tracer"), state->getNodesPerEle("Tracer"));
}

extern "C" void return_fields_()
{
  state->returnFieldToHost("Tracer");
}



