// Generated by the Manycore Form Compiler.
// https://github.com/gmarkall/manycore_form_compiler


#include "cudastatic.hpp"
#include "cudastate.hpp"


__global__ void A(int n_ele, double* localTensor, double dt, double* c0, double* c1)
{
  const double CG1[3][6] = { {  0.0915762135097707, 0.0915762135097707,
                               0.8168475729804585, 0.4459484909159649,
                               0.4459484909159649, 0.1081030181680702 },
                             {  0.0915762135097707, 0.8168475729804585,
                               0.0915762135097707, 0.4459484909159649,
                               0.1081030181680702, 0.4459484909159649 },
                             {  0.8168475729804585, 0.0915762135097707,
                               0.0915762135097707, 0.1081030181680702,
                               0.4459484909159649, 0.4459484909159649 } };
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
  double c_q1[6][2][2];
  double c_q0[6][2][2];
  for(int i_ele = THREAD_ID; i_ele < n_ele; i_ele += THREAD_COUNT)
  {
    for(int i_g = 0; i_g < 6; i_g++)
    {
      for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
      {
        for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
        {
          c_q1[i_g][i_d_0][i_d_1] = 0.0;
          for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
          {
            c_q1[i_g][i_d_0][i_d_1] += c1[i_ele + n_ele * (i_d_0 + 2 * (i_d_1 + 2 * q_r_0))] * CG1[q_r_0][i_g];
          };
        };
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
    for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
    {
      for(int i_r_1 = 0; i_r_1 < 3; i_r_1++)
      {
        localTensor[i_ele + n_ele * (i_r_0 + 3 * i_r_1)] = 0.0;
        for(int i_g = 0; i_g < 6; i_g++)
        {
          double ST2 = 0.0;
          double ST1 = 0.0;
          double ST0 = 0.0;
          ST2 += CG1[i_r_0][i_g] * CG1[i_r_1][i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
          ST1 += c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0];
          double l117[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
          double l50[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
          for(int i_d_7 = 0; i_d_7 < 2; i_d_7++)
          {
            for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
            {
              for(int i_d_5 = 0; i_d_5 < 2; i_d_5++)
              {
                for(int i_d_11 = 0; i_d_11 < 2; i_d_11++)
                {
                  ST0 += c_q1[i_g][i_d_0][i_d_1] * (l50[i_d_5][i_d_0] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_5] * (l117[i_d_11][i_d_7] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_1][i_g][i_d_11];
                };
              };
            };
          };
          localTensor[i_ele + n_ele * (i_r_0 + 3 * i_r_1)] += (-1 * 0.5 * -1 * ST0 * ST1 + ST2) * w[i_g];
        };
      };
    };
  };
}

__global__ void d(int n_ele, double* localTensor, double dt, double* c0, double* c1)
{
  const double CG1[3][6] = { {  0.0915762135097707, 0.0915762135097707,
                               0.8168475729804585, 0.4459484909159649,
                               0.4459484909159649, 0.1081030181680702 },
                             {  0.0915762135097707, 0.8168475729804585,
                               0.0915762135097707, 0.4459484909159649,
                               0.1081030181680702, 0.4459484909159649 },
                             {  0.8168475729804585, 0.0915762135097707,
                               0.0915762135097707, 0.1081030181680702,
                               0.4459484909159649, 0.4459484909159649 } };
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
  double c_q1[6][2][2];
  double c_q0[6][2][2];
  for(int i_ele = THREAD_ID; i_ele < n_ele; i_ele += THREAD_COUNT)
  {
    for(int i_g = 0; i_g < 6; i_g++)
    {
      for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
      {
        for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
        {
          c_q1[i_g][i_d_0][i_d_1] = 0.0;
          for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
          {
            c_q1[i_g][i_d_0][i_d_1] += c1[i_ele + n_ele * (i_d_0 + 2 * (i_d_1 + 2 * q_r_0))] * CG1[q_r_0][i_g];
          };
        };
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
    for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
    {
      for(int i_r_1 = 0; i_r_1 < 3; i_r_1++)
      {
        localTensor[i_ele + n_ele * (i_r_0 + 3 * i_r_1)] = 0.0;
        for(int i_g = 0; i_g < 6; i_g++)
        {
          double ST8 = 0.0;
          double ST7 = 0.0;
          ST8 += c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0];
          double l117[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
          double l50[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
          for(int i_d_7 = 0; i_d_7 < 2; i_d_7++)
          {
            for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
            {
              for(int i_d_5 = 0; i_d_5 < 2; i_d_5++)
              {
                for(int i_d_11 = 0; i_d_11 < 2; i_d_11++)
                {
                  ST7 += c_q1[i_g][i_d_0][i_d_1] * (l50[i_d_5][i_d_0] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_5] * (l117[i_d_11][i_d_7] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_1][i_g][i_d_11];
                };
              };
            };
          };
          localTensor[i_ele + n_ele * (i_r_0 + 3 * i_r_1)] += -1 * ST7 * ST8 * w[i_g];
        };
      };
    };
  };
}

__global__ void M(int n_ele, double* localTensor, double dt, double* c0)
{
  const double CG1[3][6] = { {  0.0915762135097707, 0.0915762135097707,
                               0.8168475729804585, 0.4459484909159649,
                               0.4459484909159649, 0.1081030181680702 },
                             {  0.0915762135097707, 0.8168475729804585,
                               0.0915762135097707, 0.4459484909159649,
                               0.1081030181680702, 0.4459484909159649 },
                             {  0.8168475729804585, 0.0915762135097707,
                               0.0915762135097707, 0.1081030181680702,
                               0.4459484909159649, 0.4459484909159649 } };
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
    for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
    {
      for(int i_r_1 = 0; i_r_1 < 3; i_r_1++)
      {
        localTensor[i_ele + n_ele * (i_r_0 + 3 * i_r_1)] = 0.0;
        for(int i_g = 0; i_g < 6; i_g++)
        {
          double ST6 = 0.0;
          ST6 += CG1[i_r_0][i_g] * CG1[i_r_1][i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
          localTensor[i_ele + n_ele * (i_r_0 + 3 * i_r_1)] += ST6 * w[i_g];
        };
      };
    };
  };
}

__global__ void rhs(int n_ele, double* localTensor, double dt, double* c0, double* c1, double* c2)
{
  const double CG1[3][6] = { {  0.0915762135097707, 0.0915762135097707,
                               0.8168475729804585, 0.4459484909159649,
                               0.4459484909159649, 0.1081030181680702 },
                             {  0.0915762135097707, 0.8168475729804585,
                               0.0915762135097707, 0.4459484909159649,
                               0.1081030181680702, 0.4459484909159649 },
                             {  0.8168475729804585, 0.0915762135097707,
                               0.0915762135097707, 0.1081030181680702,
                               0.4459484909159649, 0.4459484909159649 } };
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
  double c_q2[6][2][2];
  double c_q0[6][2][2];
  double c_q1[6];
  double d_c_q1[6][2];
  for(int i_ele = THREAD_ID; i_ele < n_ele; i_ele += THREAD_COUNT)
  {
    for(int i_g = 0; i_g < 6; i_g++)
    {
      for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
      {
        for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
        {
          c_q2[i_g][i_d_0][i_d_1] = 0.0;
          for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
          {
            c_q2[i_g][i_d_0][i_d_1] += c2[i_ele + n_ele * (i_d_0 + 2 * (i_d_1 + 2 * q_r_0))] * CG1[q_r_0][i_g];
          };
        };
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
      c_q1[i_g] = 0.0;
      for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
      {
        c_q1[i_g] += c1[i_ele + n_ele * q_r_0] * CG1[q_r_0][i_g];
      };
      for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
      {
        d_c_q1[i_g][i_d_0] = 0.0;
        for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
        {
          d_c_q1[i_g][i_d_0] += c1[i_ele + n_ele * q_r_0] * d_CG1[q_r_0][i_g][i_d_0];
        };
      };
    };
    for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
    {
      localTensor[i_ele + n_ele * i_r_0] = 0.0;
      for(int i_g = 0; i_g < 6; i_g++)
      {
        double ST5 = 0.0;
        double ST4 = 0.0;
        double ST3 = 0.0;
        ST5 += CG1[i_r_0][i_g] * c_q1[i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
        ST4 += c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0];
        double l117[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
        double l50[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
        for(int i_d_7 = 0; i_d_7 < 2; i_d_7++)
        {
          for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
          {
            for(int i_d_5 = 0; i_d_5 < 2; i_d_5++)
            {
              for(int i_d_11 = 0; i_d_11 < 2; i_d_11++)
              {
                ST3 += c_q2[i_g][i_d_0][i_d_1] * (l50[i_d_5][i_d_0] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_5] * (l117[i_d_11][i_d_7] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_c_q1[i_g][i_d_11];
              };
            };
          };
        };
        localTensor[i_ele + n_ele * i_r_0] += (0.5 * -1 * ST3 * ST4 + ST5) * w[i_g];
      };
    };
  };
}

StateHolder* state;
extern "C" void initialise_gpu_()
{
  state = new StateHolder();
  state->initialise();
  state->extractField("Coordinate", 1);
  state->extractField("Tracer", 0);
  state->extractField("TracerDiffusivity", 2);
  state->allocateAllGPUMemory();
  state->transferAllFields();
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
  double* localVector;
  double* localMatrix;
  double* globalVector;
  double* globalMatrix;
  double* solutionVector;
  int numValsPerNode;
  int numVectorEntries;
  CsrSparsity* Tracer_sparsity = state->getSparsity("Tracer");
  int* Tracer_colm = Tracer_sparsity->getCudaColm();
  int* Tracer_findrm = Tracer_sparsity->getCudaFindrm();
  int Tracer_colm_size = Tracer_sparsity->getSizeColm();
  int Tracer_findrm_size = Tracer_sparsity->getSizeFindrm();
  numValsPerNode = state->getValsPerNode("Tracer");
  numVectorEntries = state->getNodesPerEle("Tracer");
  numVectorEntries = numVectorEntries * numValsPerNode;
  cudaMalloc((void**)(&localMatrix), sizeof(double) * numEle * numVectorEntries * numVectorEntries);
  double* CoordinateCoeff = state->getElementValue("Coordinate");
  double* TracerDiffusivityCoeff = state->getElementValue("TracerDiffusivity");
  A<<<gridXDim,blockXDim>>>(numEle, localMatrix, dt, CoordinateCoeff, TracerDiffusivityCoeff);
  cudaMalloc((void**)(&globalMatrix), sizeof(double) * Tracer_colm_size);
  cudaMemset(globalMatrix, 0, sizeof(double) * Tracer_colm_size);
  matrix_addto<<<gridXDim,blockXDim>>>(Tracer_findrm, Tracer_colm, globalMatrix, eleNodes, localMatrix, numEle, state->getNodesPerEle("Tracer"));
  cudaFree(localMatrix);
  cudaMalloc((void**)(&localVector), sizeof(double) * numEle * numVectorEntries);
  double* TracerCoeff = state->getElementValue("Tracer");
  rhs<<<gridXDim,blockXDim>>>(numEle, localVector, dt, CoordinateCoeff, TracerCoeff, TracerDiffusivityCoeff);
  cudaMalloc((void**)(&globalVector), sizeof(double) * numNodes * numValsPerNode);
  cudaMemset(globalVector, 0, sizeof(double) * state->getValsPerNode("Tracer") * numNodes);
  vector_addto<<<gridXDim,blockXDim>>>(globalVector, eleNodes, localVector, numEle, state->getNodesPerEle("Tracer"));
  cudaFree(localVector);
  cudaMalloc((void**)(&solutionVector), sizeof(double) * numNodes * numValsPerNode);
  cg_solve(Tracer_findrm, Tracer_findrm_size, Tracer_colm, Tracer_colm_size, globalMatrix, globalVector, numNodes, solutionVector);
  cudaFree(globalMatrix);
  cudaFree(globalVector);
  expand_data<<<gridXDim,blockXDim>>>(TracerCoeff, solutionVector, eleNodes, numEle, state->getValsPerNode("Tracer"), state->getNodesPerEle("Tracer"));
  cudaFree(solutionVector);
}

extern "C" void return_fields_()
{
  state->returnFieldToHost("Tracer");
}



