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

__global__ void A_0(int n_ele, double* localTensor, double dt, double* c0, double* c1)
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
          }
        }
      }
      for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
      {
        for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
        {
          c_q0[i_g][i_d_0][i_d_1] = 0.0;
          for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
          {
            c_q0[i_g][i_d_0][i_d_1] += c0[i_ele + n_ele * (i_d_0 + 2 * q_r_0)] * d_CG1[q_r_0][i_g][i_d_1];
          }
        }
      }
    }
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
          double l164165[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
          double l157159[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
          for(int i_d_165 = 0; i_d_165 < 2; i_d_165++)
          {
            for(int i_d_159 = 0; i_d_159 < 2; i_d_159++)
            {
              for(int i_d_157 = 0; i_d_157 < 2; i_d_157++)
              {
                for(int i_d_164 = 0; i_d_164 < 2; i_d_164++)
                {
                  ST0 += c_q1[i_g][i_d_159][i_d_158] * (l157159[i_d_157][i_d_159] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_157] * (l164165[i_d_164][i_d_165] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_1][i_g][i_d_164];
                }
              }
            }
          }
          localTensor[i_ele + n_ele * (i_r_0 + 3 * i_r_1)] += (-1 * 0.5 * -1 * ST0 * ST1 + ST2) * w[i_g];
        }
      }
    }
  }
}


__global__ void d_0(int n_ele, double* localTensor, double dt, double* c0, double* c1)
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
          }
        }
      }
      for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
      {
        for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
        {
          c_q0[i_g][i_d_0][i_d_1] = 0.0;
          for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
          {
            c_q0[i_g][i_d_0][i_d_1] += c0[i_ele + n_ele * (i_d_0 + 2 * q_r_0)] * d_CG1[q_r_0][i_g][i_d_1];
          }
        }
      }
    }
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
          double l151152[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
          double l144146[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
          for(int i_d_152 = 0; i_d_152 < 2; i_d_152++)
          {
            for(int i_d_146 = 0; i_d_146 < 2; i_d_146++)
            {
              for(int i_d_144 = 0; i_d_144 < 2; i_d_144++)
              {
                for(int i_d_151 = 0; i_d_151 < 2; i_d_151++)
                {
                  ST7 += c_q1[i_g][i_d_146][i_d_145] * (l144146[i_d_144][i_d_146] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_144] * (l151152[i_d_151][i_d_152] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_1][i_g][i_d_151];
                }
              }
            }
          }
          localTensor[i_ele + n_ele * (i_r_0 + 3 * i_r_1)] += -1 * ST7 * ST8 * w[i_g];
        }
      }
    }
  }
}


__global__ void M_0(int n_ele, double* localTensor, double dt, double* c0)
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
          }
        }
      }
    }
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
        }
      }
    }
  }
}


__global__ void rhs_0(int n_ele, double* localTensor, double dt, double* c0, double* c1, double* c2)
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
          }
        }
      }
      for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
      {
        for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
        {
          c_q0[i_g][i_d_0][i_d_1] = 0.0;
          for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
          {
            c_q0[i_g][i_d_0][i_d_1] += c0[i_ele + n_ele * (i_d_0 + 2 * q_r_0)] * d_CG1[q_r_0][i_g][i_d_1];
          }
        }
      }
      c_q1[i_g] = 0.0;
      for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
      {
        c_q1[i_g] += c1[i_ele + n_ele * q_r_0] * CG1[q_r_0][i_g];
      }
      for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
      {
        d_c_q1[i_g][i_d_0] = 0.0;
        for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
        {
          d_c_q1[i_g][i_d_0] += c1[i_ele + n_ele * q_r_0] * d_CG1[q_r_0][i_g][i_d_0];
        }
      }
    }
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
        double l138139[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
        double l131133[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
        for(int i_d_139 = 0; i_d_139 < 2; i_d_139++)
        {
          for(int i_d_133 = 0; i_d_133 < 2; i_d_133++)
          {
            for(int i_d_131 = 0; i_d_131 < 2; i_d_131++)
            {
              for(int i_d_138 = 0; i_d_138 < 2; i_d_138++)
              {
                ST3 += c_q2[i_g][i_d_133][i_d_132] * (l131133[i_d_131][i_d_133] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_131] * (l138139[i_d_138][i_d_139] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_c_q1[i_g][i_d_138];
              }
            }
          }
        }
        localTensor[i_ele + n_ele * i_r_0] += (0.5 * -1 * ST3 * ST4 + ST5) * w[i_g];
      }
    }
  }
}


StateHolder* state;
extern "C" void initialise_gpu_()
{
  state = new StateHolder();
  state->initialise();
  state->extractField("Tracer", 0);
  state->extractField("TracerDiffusivity", 2);
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
  cudaMalloc((void**)(&localVector), 
             sizeof(double) * numEle * numVectorEntries);
  cudaMalloc((void**)(&localMatrix), 
             sizeof(double) * numEle * numMatrixEntries);
  cudaMalloc((void**)(&globalVector), 
             sizeof(double) * numNodes * numValsPerNode);
  cudaMalloc((void**)(&globalMatrix), sizeof(double) * Tracer_colm_size);
  cudaMalloc((void**)(&solutionVector), 
             sizeof(double) * numNodes * numValsPerNode);
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
  int nodesPerEle = state->getNodesPerEle("Coordinate");
  int blockXDim = 64;
  int gridXDim = 128;
  cudaMemset(globalMatrix, 0, sizeof(double) * Tracer_colm_size);
  cudaMemset(globalVector, 0, 
             sizeof(double) * state->getValsPerNode("Tracer") * numNodes);
  double* CoordinateCoeff = state->getElementValue("Coordinate");
  double* TracerDiffusivityCoeff = state->getElementValue("TracerDiffusivity");
  A_0<<<gridXDim,blockXDim>>>(numEle, localMatrix, dt, CoordinateCoeff, 
                              TracerDiffusivityCoeff);
  matrix_addto<<<gridXDim,blockXDim>>>(Tracer_findrm, Tracer_colm, 
                                       globalMatrix, eleNodes, localMatrix, 
                                       numEle, nodesPerEle);
  double* TracerCoeff = state->getElementValue("Tracer");
  rhs_0<<<gridXDim,blockXDim>>>(numEle, localVector, dt, CoordinateCoeff, 
                                TracerCoeff, TracerDiffusivityCoeff);
  vector_addto<<<gridXDim,blockXDim>>>(globalVector, eleNodes, localVector, 
                                       numEle, nodesPerEle);
  cg_solve(Tracer_findrm, Tracer_findrm_size, Tracer_colm, Tracer_colm_size, 
           globalMatrix, globalVector, numNodes, solutionVector);
  expand_data<<<gridXDim,blockXDim>>>(TracerCoeff, solutionVector, eleNodes, 
                                      numEle, state->getValsPerNode("Tracer"), 
                                      nodesPerEle);
}

extern "C" void return_fields_()
{
  state->returnFieldToHost("Tracer");
}


