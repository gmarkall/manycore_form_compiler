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

program test_program
  use ufl_utilities
  implicit none

  integer :: i
  integer, dimension(9) :: ndglno
  integer, dimension(18) :: ndglno_v, expected
  type(csr_sparsity) :: sparsity
  type(csr_matrix) :: matrix

  ndglno = (/ 1, 2, 4, 5, 4, 2, 2, 3, 5 /)
  expected = (/ 1, 2, 3, 4, 7, 8, 9, 10, 7, 8, 3, 4, 3, 4, 5, 6, 9, 10 /)

  sparsity = make_sparsity_from_ndglno(ndglno, ndglno, 5, 3, 3, "test")
  call allocate(matrix, sparsity)
  matrix%val=1

  call matrix2file("sparsity", matrix)

  ndglno_v = make_vector_numbering(ndglno, 3, 3, 2)

  do i=1, size(ndglno_v)
    if (ndglno_v(i) .ne. expected(i)) then
      print *, "Mismatch in data at element ", i, ", ", ndglno_v(i), " vs ", expected(i)
      stop 1
    end if
  end do

  print *, "Test passed."

end program test_program
