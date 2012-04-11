// Generated by the Manycore Form Compiler.
// https://github.com/gmarkall/manycore_form_compiler


#include "op_lib_cpp.h"
#include "op_seq_mat.h"

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
  op_set elements = get_op_element_set();
  op_field_struct Tracer = extract_op_scalar_field("Tracer", 0);
  op_field_struct Coordinate = extract_op_vector_field("Coordinate", 0);
}

extern "C" void return_fields_()
{

}


