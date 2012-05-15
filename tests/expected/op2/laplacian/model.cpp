// Generated by the Manycore Form Compiler.
// https://github.com/gmarkall/manycore_form_compiler


#ifdef __EDG__

#include "OP2_OXFORD.h"

#else

#include "op_lib_cpp.h"
#include "op_seq_mat.h"
#include "ufl_utilities.h"

#endif

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
    double ST1 = 0.0;
    double ST0 = 0.0;
    ST1 += c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0];
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
    localTensor[0] += ST0 * ST1 * w[i_g];
  }
}


void RHS_0(double** localTensor, double* dt, double* c0[2], double* c1[1])
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
  double c_q1[6];
  double c_q0[6][2][2];
  for(int i_g = 0; i_g < 6; i_g++)
  {
    c_q1[i_g] = 0.0;
    for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
    {
      c_q1[i_g] += c1[q_r_0][0] * CG1[q_r_0][i_g];
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
  for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
  {
    for(int i_g = 0; i_g < 6; i_g++)
    {
      double ST2 = 0.0;
      ST2 += CG1[i_r_0][i_g] * c_q1[i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
      localTensor[i_r_0][0] += ST2 * w[i_g];
    }
  }
}


#ifdef __EDG__

extern "C" void initialise_rose_()
{
  op_set Coordinate_elements = op_decl_set(0, "Coordinate_elements");
  op_set Coordinate_dofs = op_decl_set(0, "Coordinate_dofs");
  op_map Coordinate_element_dofs = op_decl_map(Coordinate_elements, Coordinate_dofs, 3, 0, 
              "Coordinate_element_dofs");
  op_dat Coordinate = op_decl_dat(Coordinate_dofs, 2, "double", (double*)(0), "Coordinate");
  op_set Tracer_dofs = op_decl_set(0, "Tracer_dofs");
  op_map Tracer_element_dofs = op_decl_map(Coordinate_elements, Tracer_dofs, 3, 0, "Tracer_element_dofs");
  op_dat Tracer = op_decl_dat(Tracer_dofs, 1, "double", (double*)(0), "Tracer");
}

#endif

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
  op_field_struct Coordinate_field;
  op_map Coordinate_element_dofs;
  op_dat Coordinate;
  Coordinate_field = extract_op_vector_field(state, "Coordinate", 10, 0);
  Coordinate_element_dofs = Coordinate_field.map;
  Coordinate = Coordinate_field.dat;
  op_field_struct Tracer_field;
  op_map Tracer_element_dofs;
  op_dat Tracer;
  Tracer_field = extract_op_scalar_field(state, "Tracer", 6, 0);
  Tracer_element_dofs = Tracer_field.map;
  Tracer = Tracer_field.dat;
  op_set Coordinate_elements;
  Coordinate_elements = Coordinate_element_dofs->from;
  op_sparsity A_sparsity = op_decl_sparsity(Tracer_element_dofs, Tracer_element_dofs, "A_sparsity");
  op_mat A_mat = op_decl_mat(A_sparsity, 2, "double", 8, "A_mat");
  op_par_loop(A_0, "A_0", op_iteration_space(Coordinate_elements, 3, 3), 
              op_arg_mat(A_mat, op_i(1), Tracer_element_dofs, op_i(2), 
                         Tracer_element_dofs, 2, "double", OP_INC), 
              op_arg_gbl(dt_pointer, 1, "double", OP_INC), 
              op_arg_dat(Coordinate, OP_ALL, Coordinate_element_dofs, 2, 
                         "double", OP_READ));
  op_dat RHS_vec = op_decl_vec(Tracer, "RHS_vec");
  op_par_loop(RHS_0, "RHS_0", Coordinate_elements, 
              op_arg_dat(RHS_vec, OP_ALL, Tracer_element_dofs, 2, "double", 
                         OP_INC), 
              op_arg_gbl(dt_pointer, 1, "double", OP_INC), 
              op_arg_dat(Coordinate, OP_ALL, Coordinate_element_dofs, 2, 
                         "double", OP_READ), 
              op_arg_dat(Tracer, OP_ALL, Tracer_element_dofs, 2, "double", 
                         OP_READ));
  op_solve(A_mat, RHS_vec, Tracer);
  op_free_vec(RHS_vec);
  op_free_mat(A_mat);
}

extern "C" void return_fields_()
{
  void* state = get_state();
  op_field_struct Tracer = extract_op_scalar_field(state, "Tracer", 6, 0);
  op_fetch_data(Tracer.dat);
}


