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
program ufl_program
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
  use linked_lists
  use solvers
  use diagnostic_fields_wrapper
  use iso_c_binding
  use ufl_utilities
  implicit none
#ifdef HAVE_PETSC
#include "finclude/petsc.h"
#endif

  real :: dt
  integer :: timestep
  real*4, dimension(2) :: tarray
  real*4 :: start, finish

#ifdef HAVE_PETSC
  integer :: ierr
  call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
#endif

  call python_init()
  call read_command_line()

  call populate_state(state)

  ! No support for multiphase or multimaterial at this stage.
  if (size(state)/=1) then
     FLAbort("Multiple material_phases are not supported")
  end if

  ! Always output the initial conditions.
  call output_state

  call get_option("/timestepping/current_time", current_time)
  call get_option("/timestepping/timestep", dt)

  call ETIME(tarray, start)

  call initialise_gpu

  timestep=0
  !timestep_loop: do 
     timestep=timestep+1
  !   ewrite (1,'(a,i0)') "Start of timestep ",timestep
     
     call run_model

     call calculate_diagnostic_variables(state)

  !   if (simulation_completed(current_time, timestep)) exit timestep_loop     

  !   call advance_current_time(current_time, dt)

  !   if (do_write_state(current_time, timestep)) then
  !      call output_state
  !   end if

  !end do timestep_loop
  
  call finalise_gpu

  call ETIME(tarray, finish)
  print *,"Simulation time: ",(finish-start)

  ! One last dump
  call output_state

contains

  subroutine read_command_line()
    ! Read the input filename.

    character(len=1024) :: argument
    integer :: status, argn, level

    call set_global_debug_level(0)

    argn=1
    do

       call get_command_argument(argn, value=argument, status=status)
       argn=argn+1

       if (status/=0) then
          call usage
          stop
       end if

       if (argument=="-v") then
          call get_command_argument(argn, value=argument, status=status)
          argn=argn+1

          if (status/=0) then
             call usage
             stop
          end if

          read(argument, "(i1)", err=666) level
          call set_global_debug_level(level)

          ! Go back to picj up the command line.
          cycle
       end if

       exit
    end do

    call load_options(argument)

    return

666 call usage
    stop

  end subroutine read_command_line

  subroutine usage

    write (0,*) "usage: ufl_program [-v n] <options_file>"
    write (0,*) ""
    write (0,*) "-v n sets the verbosity of debugging"
  end subroutine usage

  subroutine advance_current_time(current_time, dt)
    real, intent(inout) :: current_time, dt
    
    ! Adaptive timestepping could go here.

    current_time=current_time + dt

  end subroutine advance_current_time

  subroutine output_state

    integer, save :: dump_no=0

    call write_state(dump_no, state)
    
  end subroutine output_state


end program ufl_program
