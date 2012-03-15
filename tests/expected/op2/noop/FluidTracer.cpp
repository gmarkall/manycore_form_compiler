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
  op_dat Tracer_data = get_op_dat("Tracer");
  op_map Tracer_map = get_op_map("Tracer");
  op_set Tracer_set = get_op_set("Tracer");
  op_dat Coordinate_data = get_op_dat("Coordinate");
  op_map Coordinate_map = get_op_map("Coordinate");
  op_set Coordinate_set = get_op_set("Coordinate");
}

extern "C" void return_fields_()
{

}


