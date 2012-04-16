// Generated by the Manycore Form Compiler.
// https://github.com/gmarkall/manycore_form_compiler


#include "op_lib_cpp.h"
#include "op_seq_mat.h"
#include "ufl_utilities.h"

void rhs_0(double** localTensor, double* dt, double* c0[2], double* c1[1], double* c2[2])
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
  double c_q2[6][2];
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
      c_q2[i_g][i_d_0] = 0.0;
      for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
      {
        c_q2[i_g][i_d_0] += c2[q_r_0][i_d_0] * CG1[q_r_0][i_g];
      }
    }
  }
  for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
  {
    for(int i_g = 0; i_g < 6; i_g++)
    {
      double ST0 = 0.0;
      double ST1 = 0.0;
      double ST2 = 0.0;
      ST0 += CG1[i_r_0][i_g] * c_q1[i_g];
      double l40[2][2] = { { c_q0[i_g][1][1], -1 * c_q0[i_g][0][1] }, { -1 * c_q0[i_g][1][0], c_q0[i_g][0][0] } };
      ST2 += c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0];
      for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
      {
        for(int i_d_4 = 0; i_d_4 < 2; i_d_4++)
        {
          ST1 += c_q2[i_g][i_d_0] * (l40[i_d_4][i_d_0] / (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0])) * d_CG1[i_r_0][i_g][i_d_4];
        }
      }
      localTensor[i_r_0][0] += ST2 * (c_q1[i_g] * *dt * ST1 + ST0) * w[i_g];
    }
  }
}


void Mass_0(double* localTensor, double* dt, double* c0[2], int i_r_0, int i_r_1)
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
    ST3 += CG1[i_r_0][i_g] * CG1[i_r_1][i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
    localTensor[0] += ST3 * w[i_g];
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
  op_field_struct Velocity = extract_op_vector_field(state, "Velocity", 8, 0);
  op_field_struct Coordinate = extract_op_vector_field(state, "Coordinate", 10, 0);
  op_field_struct Tracer = extract_op_scalar_field(state, "Tracer", 6, 0);
  op_sparsity Mass_sparsity = op_decl_sparsity(Tracer.map, Tracer.map, "Mass_sparsity");
  op_mat Mass_mat = op_decl_mat(Mass_sparsity, Tracer.dat->dim, "double", 8, "Mass_mat");
  op_par_loop(Mass, "Mass", op_iteration_space(Tracer.map->from, 3, 3), 
              op_arg_mat(Mass_mat, OP_ALL, Tracer.map, OP_ALL, Tracer.map, 
                         Tracer.dat->dim, "double", OP_INC), 
              op_arg_gbl(dt_pointer, 1, "double", OP_INC), 
              op_arg_dat(Coordinate.dat, OP_ALL, Coordinate.map, 
                         Coordinate.dat->dim, "double", OP_READ));
  op_dat rhs_vec = op_decl_vec(Tracer.dat, "rhs_vec");
  op_par_loop(rhs, "rhs", Tracer.map->from, 
              op_arg_dat(rhs_vec, OP_ALL, Tracer.map, Tracer.dat->dim, 
                         "double", OP_INC), 
              op_arg_gbl(dt_pointer, 1, "double", OP_INC), 
              op_arg_dat(Coordinate.dat, OP_ALL, Coordinate.map, 
                         Coordinate.dat->dim, "double", OP_READ), 
              op_arg_dat(Tracer.dat, OP_ALL, Tracer.map, Tracer.dat->dim, 
                         "double", OP_READ), 
              op_arg_dat(Velocity.dat, OP_ALL, Velocity.map, 
                         Velocity.dat->dim, "double", OP_READ));
  op_solve(Mass_mat, rhs_vec, Tracer.dat);
  op_free_vec(rhs_vec);
  op_free_mat(Mass_mat);
}

extern "C" void return_fields_()
{
  void* state = get_state();
  op_field_struct Tracer = extract_op_scalar_field(state, "Tracer", 6, 0);
  op_fetch_data(Tracer.dat);
}


