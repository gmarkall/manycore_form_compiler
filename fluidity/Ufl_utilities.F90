!    Copyright (C) 2006 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
!    Prof. C Pain
!    Applied Modelling and Computation Group
!    Department of Earth Science and Engineering
!    Imperial College London
!
!    C.Pain@Imperial.ac.uk
!
!    This library is free software; you can redistribute it and/or
!    modify it under the terms of the GNU Lesser General Public
!    License as published by the Free Software Foundation,
!    version 2.1 of the License.
!
!    This library is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
!    Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public
!    License along with this library; if not, write to the Free Software
!    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
!    USA

#include "fdebug.h"
module ufl_utilities
  use spud
  use fields
  use state_module
  use FLDebug
  use populate_state_module
  use write_state_module
  use populate_state_module
  use timeloop_utilities
  use sparsity_patterns
  use sparsity_patterns_meshes
  use solvers
  use diagnostic_fields_wrapper
  use linked_lists
  use iso_c_binding
  implicit none

  type(state_type), dimension(:), pointer :: state

contains

  type(c_ptr) function get_state() bind(c)
    ! FIXME: always return first state since we're not supporting multi-phase
    get_state = c_loc(state(1))
  end function get_state

  subroutine extract_scalar_field_wrapper(field_name_buf, field_name_len, conn_findrm, conn_findrm_size, &
         & conn_colm, conn_colm_size, at_findrm, at_findrm_size, at_colm, &
         & at_colm_size, num_ele, num_nodes, ndglno, loc, dim, ngi, n, &
         & dn, degree, weight, val) bind(c)
    !!< Parameters
    !!<
    !!< Field name buffer
    character(kind=c_char), dimension(32), intent(in) :: field_name_buf
    integer(c_int), intent(in) :: field_name_len
    !!< ConnSparsity data
    type(c_ptr), intent(out) :: conn_findrm
    integer(c_int), intent(out) :: conn_findrm_size
    type(c_ptr), intent(out) :: conn_colm
    integer(c_int), intent(out) :: conn_colm_size
    !!< ATSparsity data
    type(c_ptr), intent(out) :: at_findrm
    integer(c_int), intent(out) :: at_findrm_size
    type(c_ptr), intent(out) :: at_colm
    integer(c_int), intent(out) :: at_colm_size
    !!< Mesh data
    integer(c_int), intent(out) :: num_ele, num_nodes
    type(c_ptr), intent(out) :: ndglno
    !!< Element data
    integer(c_int), intent(out) :: loc, dim, ngi
    type(c_ptr), intent(out) :: n
    type(c_ptr), intent(out) :: dn
    !!< Quadrature data
    integer(c_int), intent(out) :: degree
    type(c_ptr), intent(out) :: weight
    !!< Field data
    type(c_ptr), intent(out) :: val

    !!< Local vars
    !!<
    character(len=field_name_len) :: field_name
    type(scalar_field), pointer :: field
    type(csr_sparsity) :: conn_sparsity, at_sparsity
    integer :: stat, c

    type(ilist), dimension(:), pointer :: ATList
    integer, dimension(:), pointer :: t_ele
    integer :: ele, row_count, ele_count, i
    integer(c_int), pointer :: tmp_int_ptr
    real(c_double), pointer :: tmp_real_ptr

    do c=1,field_name_len
      field_name(c:c) = field_name_buf(c)
    end do

    print *,"Extracting scalar field: ",field_name

    field => extract_scalar_field(state, field_name, stat=stat)
    if (stat/=0) then
      FLAbort("Tried to extract non-existent/non-allocated scalar field!")
    end if

    ! Connectivity sparsity
    conn_sparsity = get_csr_sparsity_firstorder(state, field%mesh, field%mesh)
    
    tmp_int_ptr => conn_sparsity%findrm(1)
    conn_findrm = c_loc(tmp_int_ptr)

    tmp_int_ptr => conn_sparsity%colm(1)
    conn_colm = c_loc(tmp_int_ptr)

    conn_findrm_size = size(conn_sparsity%findrm)
    conn_colm_size = size(conn_sparsity%colm)

    ! AT Sparsity
    row_count = node_count(field%mesh)
    ele_count = element_count(field%mesh)
    allocate(ATList(row_count))

    do ele=1, ele_count
      t_ele=>ele_nodes(field, ele)
      call insert_ascending(ATList(t_ele(1)),(ele-1)*3+1)
      call insert_ascending(ATList(t_ele(2)),(ele-1)*3+2)
      call insert_ascending(ATList(t_ele(3)),(ele-1)*3+3)
    end do

    ! This gets lost when the program exits. Think of a nice way
    ! to deallocate it at the end. Same problem for vector field.
    at_sparsity = lists2csr_sparsity(ATList, "AT")
    at_sparsity%columns=ele_count*3
    at_sparsity%sorted_rows=.true.

    do i=1,row_count
      call flush_list(ATList(i))
    end do

    deallocate(ATList)

    tmp_int_ptr => at_sparsity%findrm(1)
    at_findrm = c_loc(tmp_int_ptr)

    tmp_int_ptr => at_sparsity%colm(1)
    at_colm = c_loc(tmp_int_ptr)

    at_findrm_size = size(at_sparsity%findrm)
    at_colm_size = size(at_sparsity%colm)

    ! Mesh
    num_ele = ele_count
    num_nodes = node_count(field%mesh)

    tmp_int_ptr => field%mesh%ndglno(1)
    ndglno = c_loc(tmp_int_ptr)

    ! Element
    loc = field%mesh%shape%loc
    dim = field%mesh%shape%dim

    tmp_real_ptr => field%mesh%shape%n(1,1)
    n = c_loc(tmp_real_ptr)

    tmp_real_ptr => field%mesh%shape%dn(1,1,1)
    dn = c_loc(tmp_real_ptr)

    ! Quadrature
    degree = field%mesh%shape%quadrature%degree
    ngi = field%mesh%shape%quadrature%ngi

    tmp_real_ptr => field%mesh%shape%quadrature%weight(1)
    weight = c_loc(tmp_real_ptr)

    ! Field
    tmp_real_ptr => field%val(1)
    val = c_loc(tmp_real_ptr)
  
    print *,"Finished extraction wrapper for scalar field: ",field_name

  end subroutine extract_scalar_field_wrapper

  subroutine extract_vector_field_wrapper(field_name_buf, field_name_len, conn_findrm, conn_findrm_size, &
         & conn_colm, conn_colm_size, at_findrm, at_findrm_size, at_colm, &
         & at_colm_size, num_ele, num_nodes, ndglno, loc, dim, ngi, n, &
         & dn, degree, weight, val) bind(c)
    !!< Parameters
    !!<
    !!< Field name buffer
    character(kind=c_char), dimension(32), intent(in) :: field_name_buf
    integer(c_int), intent(in) :: field_name_len
    !!< ConnSparsity data
    type(c_ptr), intent(out) :: conn_findrm
    integer(c_int), intent(out) :: conn_findrm_size
    type(c_ptr), intent(out) :: conn_colm
    integer(c_int), intent(out) :: conn_colm_size
    !!< ATSparsity data
    type(c_ptr), intent(out) :: at_findrm
    integer(c_int), intent(out) :: at_findrm_size
    type(c_ptr), intent(out) :: at_colm
    integer(c_int), intent(out) :: at_colm_size
    !!< Mesh data
    integer(c_int), intent(out) :: num_ele, num_nodes
    type(c_ptr), intent(out) :: ndglno
    !!< Element data
    integer(c_int), intent(out) :: loc, dim, ngi
    type(c_ptr), intent(out) :: n
    type(c_ptr), intent(out) :: dn
    !!< Quadrature data
    integer(c_int), intent(out) :: degree
    type(c_ptr), intent(out) :: weight
    !!< Field data
    type(c_ptr), intent(out) :: val

    !!< Local vars
    !!<
    character(len=field_name_len) :: field_name
    type(vector_field), pointer :: field
    type(csr_sparsity) :: conn_sparsity, at_sparsity
    integer :: stat, c

    type(ilist), dimension(:), pointer :: ATList
    integer, dimension(:), pointer :: t_ele
    integer :: ele, row_count, ele_count, i
    integer(c_int), pointer :: tmp_int_ptr
    real(c_double), pointer :: tmp_real_ptr

    do c=1,field_name_len
      field_name(c:c) = field_name_buf(c)
    end do

    print *,"Extracting vector field: ",field_name

    field => extract_vector_field(state, field_name, stat=stat)
    if (stat/=0) then
      FLAbort("Tried to extract non-existent/non-allocated vector field!")
    end if

    ! Connectivity sparsity
    conn_sparsity = get_csr_sparsity_firstorder(state, field%mesh, field%mesh)
    
    tmp_int_ptr => conn_sparsity%findrm(1)
    conn_findrm = c_loc(tmp_int_ptr)

    tmp_int_ptr => conn_sparsity%colm(1)
    conn_colm = c_loc(tmp_int_ptr)

    conn_findrm_size = size(conn_sparsity%findrm)
    conn_colm_size = size(conn_sparsity%colm)

    ! AT Sparsity
    row_count = node_count(field%mesh)
    ele_count = element_count(field%mesh)
    allocate(ATList(row_count))

    do ele=1, ele_count
      t_ele=>ele_nodes(field, ele)
      call insert_ascending(ATList(t_ele(1)),(ele-1)*3+1)
      call insert_ascending(ATList(t_ele(2)),(ele-1)*3+2)
      call insert_ascending(ATList(t_ele(3)),(ele-1)*3+3)
    end do

    at_sparsity = lists2csr_sparsity(ATList, "AT")
    at_sparsity%columns=ele_count*3
    at_sparsity%sorted_rows=.true.

    do i=1,row_count
      call flush_list(ATList(i))
    end do

    deallocate(ATList)

    tmp_int_ptr => at_sparsity%findrm(1)
    at_findrm = c_loc(tmp_int_ptr)

    tmp_int_ptr => at_sparsity%colm(1)
    at_colm = c_loc(tmp_int_ptr)

    at_findrm_size = size(at_sparsity%findrm)
    at_colm_size = size(at_sparsity%colm)

    ! Mesh
    num_ele = ele_count
    num_nodes = node_count(field%mesh)

    tmp_int_ptr => field%mesh%ndglno(1)
    ndglno = c_loc(tmp_int_ptr)

    ! Element
    loc = field%mesh%shape%loc
    dim = field%mesh%shape%dim

    tmp_real_ptr => field%mesh%shape%n(1,1)
    n = c_loc(tmp_real_ptr)

    tmp_real_ptr => field%mesh%shape%dn(1,1,1)
    dn = c_loc(tmp_real_ptr)

    ! Quadrature
    degree = field%mesh%shape%quadrature%degree
    ngi = field%mesh%shape%quadrature%ngi

    tmp_real_ptr => field%mesh%shape%quadrature%weight(1)
    weight = c_loc(tmp_real_ptr)

    ! Field
    tmp_real_ptr => field%val(1,1)
    val = c_loc(tmp_real_ptr)

    print *,"Finished extraction wrapper for vector field: ",field_name

  end subroutine extract_vector_field_wrapper

end module ufl_utilities
