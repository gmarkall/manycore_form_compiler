// Generated by the Manycore Form Compiler.
// https://github.com/gmarkall/manycore_form_compiler


#ifdef __EDG__

#include "OP2_OXFORD.h"

#else

#include "op_lib_cpp.h"
#include "op_seq_mat.h"
#include "ufl_utilities.h"

#endif

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
}

extern "C" void return_fields_()
{

}


