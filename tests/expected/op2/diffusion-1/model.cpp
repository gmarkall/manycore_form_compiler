// Generated by the Manycore Form Compiler.
// https://github.com/gmarkall/manycore_form_compiler


#include "op_lib_cpp.h"
#include "op_seq_mat.h"
extern "C" void* get_state();

void A_0(double* localTensor, double* dt, double* c0[2], double* c1[2][2], int i_r_0, int i_r_1)
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
  for(int i_g = 0; i_g < 6; i_g++)
  {
    for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
    {
      for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
      {
        c_q1[i_g][i_d_0][i_d_1] = 0.0;
        for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
        {
          c_q1[i_g][i_d_0][i_d_1] += c1[q_r_0][i_d_0][i_d_1] * CG1[q_r_0][i_g];
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
          c_q0[i_g][i_d_0][i_d_1] += c0[q_r_0][i_d_0] * d_CG1[q_r_0][i_g][i_d_1];
        }
      }
    }
  }
  for(int i_g = 0; i_g < 6; i_g++)
  {
    double ST2 = 0.0;
    double ST1 = 0.0;
    double ST0 = 0.0;
    ST2 += CG1[i_r_0][i_g] * CG1[i_r_1][i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
    ST1 += c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0];
    double l101[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
    double l50[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
    for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
    {
      for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
      {
        for(int i_d_5 = 0; i_d_5 < 2; i_d_5++)
        {
          for(int i_d_10 = 0; i_d_10 < 2; i_d_10++)
          {
            ST0 += c_q1[i_g][i_d_0][i_d_1] * -1 * (l50[i_d_5][i_d_0] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_5] * (l101[i_d_10][i_d_1] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_1][i_g][i_d_10];
          }
        }
      }
    }
    localTensor[0] += (-1 * 0.5 * ST0 * ST1 + ST2) * w[i_g];
  }
}


void d_0(double* localTensor, double* dt, double* c0[2], double* c1[2][2], int i_r_0, int i_r_1)
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
  for(int i_g = 0; i_g < 6; i_g++)
  {
    for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
    {
      for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
      {
        c_q1[i_g][i_d_0][i_d_1] = 0.0;
        for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
        {
          c_q1[i_g][i_d_0][i_d_1] += c1[q_r_0][i_d_0][i_d_1] * CG1[q_r_0][i_g];
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
          c_q0[i_g][i_d_0][i_d_1] += c0[q_r_0][i_d_0] * d_CG1[q_r_0][i_g][i_d_1];
        }
      }
    }
  }
  for(int i_g = 0; i_g < 6; i_g++)
  {
    double ST8 = 0.0;
    double ST7 = 0.0;
    ST8 += c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0];
    double l101[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
    double l50[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
    for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
    {
      for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
      {
        for(int i_d_5 = 0; i_d_5 < 2; i_d_5++)
        {
          for(int i_d_10 = 0; i_d_10 < 2; i_d_10++)
          {
            ST7 += c_q1[i_g][i_d_0][i_d_1] * -1 * (l50[i_d_5][i_d_0] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_5] * (l101[i_d_10][i_d_1] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_1][i_g][i_d_10];
          }
        }
      }
    }
    localTensor[0] += ST7 * ST8 * w[i_g];
  }
}


void M_0(double* localTensor, double* dt, double* c0[2], int i_r_0, int i_r_1)
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
  for(int i_g = 0; i_g < 6; i_g++)
  {
    for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
    {
      for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
      {
        c_q0[i_g][i_d_0][i_d_1] = 0.0;
        for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
        {
          c_q0[i_g][i_d_0][i_d_1] += c0[q_r_0][i_d_0] * d_CG1[q_r_0][i_g][i_d_1];
        }
      }
    }
  }
  for(int i_g = 0; i_g < 6; i_g++)
  {
    double ST6 = 0.0;
    ST6 += CG1[i_r_0][i_g] * CG1[i_r_1][i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
    localTensor[0] += ST6 * w[i_g];
  }
}


void rhs_0(double** localTensor, double* dt, double* c0[2], double* c1[1], double* c2[2][2])
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
  for(int i_g = 0; i_g < 6; i_g++)
  {
    for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
    {
      for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
      {
        c_q2[i_g][i_d_0][i_d_1] = 0.0;
        for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
        {
          c_q2[i_g][i_d_0][i_d_1] += c2[q_r_0][i_d_0][i_d_1] * CG1[q_r_0][i_g];
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
          c_q0[i_g][i_d_0][i_d_1] += c0[q_r_0][i_d_0] * d_CG1[q_r_0][i_g][i_d_1];
        }
      }
    }
    c_q1[i_g] = 0.0;
    for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
    {
      c_q1[i_g] += c1[q_r_0][0] * CG1[q_r_0][i_g];
    }
    for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
    {
      d_c_q1[i_g][i_d_0] = 0.0;
      for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
      {
        d_c_q1[i_g][i_d_0] += c1[q_r_0][0] * d_CG1[q_r_0][i_g][i_d_0];
      }
    }
  }
  for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
  {
    for(int i_g = 0; i_g < 6; i_g++)
    {
      double ST5 = 0.0;
      double ST4 = 0.0;
      double ST3 = 0.0;
      ST5 += CG1[i_r_0][i_g] * c_q1[i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
      ST4 += c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0];
      double l101[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
      double l50[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
      for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
      {
        for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
        {
          for(int i_d_5 = 0; i_d_5 < 2; i_d_5++)
          {
            for(int i_d_10 = 0; i_d_10 < 2; i_d_10++)
            {
              ST3 += c_q2[i_g][i_d_0][i_d_1] * -1 * (l50[i_d_5][i_d_0] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_5] * (l101[i_d_10][i_d_1] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_c_q1[i_g][i_d_10];
            }
          }
        }
      }
      localTensor[i_r_0][0] += (0.5 * ST3 * ST4 + ST5) * w[i_g];
    }
  }
}


extern "C" void initialise_gpu_()
{
  op_init(0, 0, 2);
}

extern "C" void finalise_gpu_()
{
  op_exit();
}

extern "C" void run_model_(double* dt_pointer)
{
  void* state = get_state();
  op_field_struct TracerDiffusivity = extract_op_tensor_field(state, "TracerDiffusivity", 0);
  op_field_struct Coordinate = extract_op_vector_field(state, "Coordinate", 0);
  op_field_struct Tracer = extract_op_scalar_field(state, "Tracer", 0);
  op_sparsity A_sparsity = op_decl_sparsity(Tracer.map, Tracer.map, "A_sparsity");
  op_mat A_mat = op_decl_mat(A_sparsity, Tracer.dat.dim, "double", 8, "A_mat");
  op_par_loop(A, "A", Tracer.map.from, 
              op_arg_mat(A_mat, OP_ALL, Tracer.map, OP_ALL, Tracer.map, 
                         OP_INC), 
              op_arg_dat(Coordinate.dat, OP_ALL, Coordinate.map, OP_READ), 
              op_arg_dat(TracerDiffusivity.dat, OP_ALL, TracerDiffusivity.map, 
                         OP_READ));
  op_dat rhs_vec = op_decl_vec(Tracer.dat, "rhs_vec");
  op_par_loop(rhs, "rhs", Tracer.map.from, 
              op_arg_dat(rhs_vec, OP_ALL, Tracer.map, OP_INC), 
              op_arg_dat(Coordinate.dat, OP_ALL, Coordinate.map, OP_READ), 
              op_arg_dat(Tracer.dat, OP_ALL, Tracer.map, OP_READ), 
              op_arg_dat(TracerDiffusivity.dat, OP_ALL, TracerDiffusivity.map, 
                         OP_READ));
  op_solve(A_mat, rhs_vec, Tracer.dat);
  op_free_vec(rhs_vec);
  op_free_mat(A_mat);
}

extern "C" void return_fields_()
{
  void* state = get_state();
  op_field_struct Tracer = extract_op_scalar_field(state, "Tracer", 0);
  op_fetch_data(Tracer.dat);
}


