// Generated by the Manycore Form Compiler.
// https://github.com/gmarkall/manycore_form_compiler


#include "op_lib_cpp.h"
#include "op_seq_mat.h"
#include "ufl_utilities.h"

void A_0(double* localTensor, double* dt, double* c0[2], int i_r_0, int i_r_1)
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
    double ST3 = 0.0;
    double ST2 = 0.0;
    double ST1 = 0.0;
    double ST0 = 0.0;
    ST3 += CG1[i_r_0][i_g] * CG1[i_r_1][i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
    ST2 += c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0];
    ST1 += 0.1 * -1 * *dt;
    double l95[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
    double l35[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
    for(int i_d_5 = 0; i_d_5 < 2; i_d_5++)
    {
      for(int i_d_3 = 0; i_d_3 < 2; i_d_3++)
      {
        for(int i_d_9 = 0; i_d_9 < 2; i_d_9++)
        {
          ST0 += (l35[i_d_3][i_d_5] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_3] * (l95[i_d_9][i_d_5] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_1][i_g][i_d_9];
        }
      }
    }
    localTensor[0] += (-1 * 0.5 * ST0 * ST1 * ST2 + ST3) * w[i_g];
  }
}


void d_0(double* localTensor, double* dt, double* c0[2], int i_r_0, int i_r_1)
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
    double ST14 = 0.0;
    double ST13 = 0.0;
    double ST12 = 0.0;
    ST14 += c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0];
    ST13 += 0.1 * -1 * *dt;
    double l95[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
    double l35[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
    for(int i_d_5 = 0; i_d_5 < 2; i_d_5++)
    {
      for(int i_d_3 = 0; i_d_3 < 2; i_d_3++)
      {
        for(int i_d_9 = 0; i_d_9 < 2; i_d_9++)
        {
          ST12 += (l35[i_d_3][i_d_5] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_3] * (l95[i_d_9][i_d_5] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_1][i_g][i_d_9];
        }
      }
    }
    localTensor[0] += ST12 * ST13 * ST14 * w[i_g];
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
    double ST11 = 0.0;
    ST11 += CG1[i_r_0][i_g] * CG1[i_r_1][i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
    localTensor[0] += ST11 * w[i_g];
  }
}


void diff_rhs_0(double** localTensor, double* dt, double* c0[2], double* c1[1])
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
  double c_q1[6];
  double d_c_q1[6][2];
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
      double ST7 = 0.0;
      double ST6 = 0.0;
      double ST5 = 0.0;
      double ST4 = 0.0;
      ST7 += CG1[i_r_0][i_g] * c_q1[i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
      ST6 += c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0];
      ST5 += 0.1 * -1 * *dt;
      double l95[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
      double l35[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
      for(int i_d_5 = 0; i_d_5 < 2; i_d_5++)
      {
        for(int i_d_3 = 0; i_d_3 < 2; i_d_3++)
        {
          for(int i_d_9 = 0; i_d_9 < 2; i_d_9++)
          {
            ST4 += (l35[i_d_3][i_d_5] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_3] * (l95[i_d_9][i_d_5] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_c_q1[i_g][i_d_9];
          }
        }
      }
      localTensor[i_r_0][0] += (0.5 * ST4 * ST5 * ST6 + ST7) * w[i_g];
    }
  }
}


void adv_rhs_0(double** localTensor, double* dt, double* c0[2], double* c1[2], double* c2[1])
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
  double c_q2[6];
  double c_q1[6][2];
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
    c_q2[i_g] = 0.0;
    for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
    {
      c_q2[i_g] += c2[q_r_0][0] * CG1[q_r_0][i_g];
    }
    for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
    {
      c_q1[i_g][i_d_0] = 0.0;
      for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
      {
        c_q1[i_g][i_d_0] += c1[q_r_0][i_d_0] * CG1[q_r_0][i_g];
      }
    }
  }
  for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
  {
    for(int i_g = 0; i_g < 6; i_g++)
    {
      double ST8 = 0.0;
      double ST9 = 0.0;
      double ST10 = 0.0;
      ST8 += CG1[i_r_0][i_g] * c_q2[i_g];
      double l40[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
      ST10 += c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0];
      for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
      {
        for(int i_d_4 = 0; i_d_4 < 2; i_d_4++)
        {
          ST9 += c_q1[i_g][i_d_0] * (l40[i_d_4][i_d_0] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_4];
        }
      }
      localTensor[i_r_0][0] += ST10 * (c_q2[i_g] * *dt * ST9 + ST8) * w[i_g];
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
  op_field_struct Coordinate = extract_op_vector_field(state, "Coordinate", 0);
  op_field_struct Velocity = extract_op_vector_field(state, "Velocity", 0);
  op_field_struct Tracer = extract_op_scalar_field(state, "Tracer", 0);
  op_dat t_adv = op_decl_vec(Tracer.dat, "t_adv");
  op_sparsity M_sparsity = op_decl_sparsity(Tracer.map, Tracer.map, "M_sparsity");
  op_mat M_mat = op_decl_mat(M_sparsity, t_adv->dim, "double", 8, "M_mat");
  op_par_loop(M, "M", Tracer.map->from, 
              op_arg_mat(M_mat, OP_ALL, Tracer.map, OP_ALL, Tracer.map, 
                         t_adv->dim, "double", OP_INC), 
              op_arg_dat(Coordinate.dat, OP_ALL, Coordinate.map, 
                         Coordinate.dat->dim, "double", OP_READ));
  op_dat adv_rhs_vec = op_decl_vec(t_adv, "adv_rhs_vec");
  op_par_loop(adv_rhs, "adv_rhs", Tracer.map->from, 
              op_arg_dat(adv_rhs_vec, OP_ALL, Tracer.map, t_adv->dim, 
                         "double", OP_INC), 
              op_arg_dat(Coordinate.dat, OP_ALL, Coordinate.map, 
                         Coordinate.dat->dim, "double", OP_READ), 
              op_arg_dat(Velocity.dat, OP_ALL, Velocity.map, 
                         Velocity.dat->dim, "double", OP_READ), 
              op_arg_dat(Tracer.dat, OP_ALL, Tracer.map, Tracer.dat->dim, 
                         "double", OP_READ));
  op_solve(M_mat, adv_rhs_vec, t_adv);
  op_free_vec(adv_rhs_vec);
  op_free_mat(M_mat);
  op_sparsity A_sparsity = op_decl_sparsity(Tracer.map, Tracer.map, "A_sparsity");
  op_mat A_mat = op_decl_mat(A_sparsity, Tracer.dat->dim, "double", 8, "A_mat");
  op_par_loop(A, "A", Tracer.map->from, 
              op_arg_mat(A_mat, OP_ALL, Tracer.map, OP_ALL, Tracer.map, 
                         Tracer.dat->dim, "double", OP_INC), 
              op_arg_dat(Coordinate.dat, OP_ALL, Coordinate.map, 
                         Coordinate.dat->dim, "double", OP_READ));
  op_dat diff_rhs_vec = op_decl_vec(Tracer.dat, "diff_rhs_vec");
  op_par_loop(diff_rhs, "diff_rhs", Tracer.map->from, 
              op_arg_dat(diff_rhs_vec, OP_ALL, Tracer.map, Tracer.dat->dim, 
                         "double", OP_INC), 
              op_arg_dat(Coordinate.dat, OP_ALL, Coordinate.map, 
                         Coordinate.dat->dim, "double", OP_READ), 
              op_arg_dat(t_adv, OP_ALL, Tracer.map, t_adv->dim, "double", 
                         OP_READ));
  op_solve(A_mat, diff_rhs_vec, Tracer.dat);
  op_free_vec(diff_rhs_vec);
  op_free_mat(A_mat);
  op_free_vec(t_adv);
}

extern "C" void return_fields_()
{
  void* state = get_state();
  op_field_struct Tracer = extract_op_scalar_field(state, "Tracer", 0);
  op_fetch_data(Tracer.dat);
}


