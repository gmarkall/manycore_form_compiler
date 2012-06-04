# This file is part of the Manycore Form Compiler.
#
# The Manycore Form Compiler is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# The Manycore Form Compiler is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# the Manycore Form Compiler.  If not, see <http://www.gnu.org/licenses>
#
# Copyright (c) 2011, Graham Markall <grm08@doc.ic.ac.uk> and others. Please see
# the AUTHORS file in the main source directory for a full list of copyright
# holders.

from copy import copy

# MCFC libs
from assembler import *
from codegeneration import *
from formutils import extractCoordinates, numBasisFunctions
from uflnamespace import domain2num_vertices as d2v

# Global paramters
# FIXME: Default to double for now
real_kind = Literal('double')

# Profiling
elapsed = Variable('elapsed_time', Real())
opTimer = lambda time: FunctionCall('op_timers_core', [Literal(0), AddressOfOp(time)])

startTiming = lambda: opTimer(elapsed)

def endTiming(time):
    return [
        MinusAssignmentOp(time, elapsed),
        opTimer(elapsed),
        PlusAssignmentOp(time, elapsed)
        ]
# OP2 data types
# FIXME: introduce Type 'Struct' for these?
OpSet = Class('op_set')
OpMap = Class('op_map')
OpDat = Class('op_dat')
OpMat = Class('op_mat')
OpSparsity = Class('op_sparsity')
OpFieldStruct = Class('op_field_struct')
OpAll = Constant('OP_ALL')
OpInc = Constant('OP_INC')
OpRead = Constant('OP_READ')

# OP2 functions
opInit = lambda diags: FunctionCall('op_init', [Literal(0), Literal(0), diags])
opExit = lambda : FunctionCall('op_exit', [])
# The set sizes are inherently runtime, so we pass a constant 0
opDeclSet = lambda name: \
    FunctionCall('op_decl_set',  [Literal(0), Literal(name)]);
# We pass a NULL pointer for the data since these are only fake calls
opDeclMap = lambda from_set, to_set, dim, name: \
    FunctionCall('op_decl_map',  [from_set, to_set, Literal(dim), Literal(0), Literal(name)]);
# We pass a NULL pointer for the data since these are only fake calls
opDeclDat = lambda dataset, dim, name: \
    FunctionCall('op_decl_dat', \
            [dataset, Literal(dim), real_kind, Cast(Pointer(Real()), Literal(0)), Literal(name)]);
opDeclVec = lambda origin, name: \
    FunctionCall('op_decl_vec', [origin, Literal(name)])
opDeclSparsity = lambda rowmap, colmap, name: \
    FunctionCall('op_decl_sparsity', [rowmap, colmap, Literal(name)])
opDeclMat = lambda sparsity, dim, name: \
    FunctionCall('op_decl_mat', [sparsity, dim, real_kind, Literal(8), Literal(name)])
opFreeVec = lambda vec: \
    FunctionCall('op_free_vec', [vec])
opFreeMat = lambda mat: \
    FunctionCall('op_free_mat', [mat])
opArgGbl = lambda dat, dim, access: \
    FunctionCall('op_arg_gbl', [dat, dim, real_kind, access])
opArgDat = lambda dat, index, mapping, dim, access: \
    FunctionCall('op_arg_dat', [dat, index, mapping, dim, real_kind, access])
opArgMat = lambda mat, rowindex, rowmap, colindex, colmap, dim, access: \
    FunctionCall('op_arg_mat', [mat, rowindex, rowmap, colindex, colmap, dim, real_kind, access])
opFetchData = lambda dat: FunctionCall('op_fetch_data', [dat])
opIterationSpace = lambda iterset, dims: \
    FunctionCall('op_iteration_space', [iterset, Literal(dims[0]), Literal(dims[1])])
opI = lambda index: FunctionCall('op_i', [index])
opParLoop = lambda kernel, iterspace, arguments: \
    FunctionCall('op_par_loop', [FunctionPointer(kernel), Literal(kernel), iterspace] + arguments)
opSolve = lambda A, b, x: FunctionCall('op_solve', [A, b, x])

# Opaque pointer to fluidity state
state = Variable('state', Pointer(Void()))
rank2type = { 0: 'scalar', 1: 'vector', 2: 'tensor' }
# Fluidity OP2 state functions
opExtractField = lambda fieldname, rank, codim=0: \
        FunctionCall('extract_op_%s_field' % rank2type[rank], \
        [state, Literal(fieldname), Literal(len(fieldname)), Literal(codim)])

class Field:

    def __init__(self, rank, name, dim, loc):
        self.rank = rank
        self.name = name
        self.dim = dim**rank
        self.loc = loc
        self.dof_set = Variable(name+'_dofs', OpSet)
        self.map = Variable(name+'_element_dofs', OpMap)
        self.dat = Variable(name, OpDat)

    def buildFakeInitialiser(self, scope, elem_set):
        scope.append(InitialisationOp(self.dof_set, opDeclSet(self.dof_set.name())))
        scope.append(InitialisationOp(self.map, \
                opDeclMap(elem_set, self.dof_set, self.loc, self.map.name())))
        scope.append(InitialisationOp(self.dat, \
                opDeclDat(self.dof_set, self.dim, self.name)))

    def buildInitialiser(self, scope):
        # Get OP2 data structures
        var = Variable(self.name + '_field', OpFieldStruct)
        scope.append(Declaration(var))
        scope.append(Declaration(self.map))
        scope.append(Declaration(self.dat))
        # FIXME: We stupidly default to requesting co-dimension 0.
        # This should be infered from the integral's measure.
        scope.append(AssignmentOp(var, opExtractField(self.name, self.rank)))
        scope.append(AssignmentOp(self.map, MemberAccess(var, 'map')))
        # The field dof set is not currently needed
        #scope.append(InitialisationOp(self.dof_set, MemberAccess(self.map, 'to', True)))
        scope.append(AssignmentOp(self.dat, MemberAccess(var, 'dat')))

class Op2AssemblerBackend(AssemblerBackend):

    def compile(self, equation):

        self._eq = equation

        # Mesh constants
        # FIXME: We'll need the same for facets once we support facet integrals
        cell = equation.state[1]['Coordinate'].cell()
        self.loc = d2v[cell.domain()]
        self.dim = cell.d
        # Element set (used by all maps)
        self.elem_set = Variable('Coordinate_elements', OpSet)
        # Create field meta data for all fields used
        self._fields = dict((field, Field(rank, field, self.dim, self.loc)) \
                for rank, field in equation.state.accessedFields().values())

        # Build declarations
        declarations = GlobalScope()
        declarations.append(self._buildFakeInitialiser())
        declarations.append(self._buildInitialiser())
        declarations.append(self._buildRunModel())
        declarations.append(self._buildReturnFields())
        declarations.append(self._buildFinaliser())

        # Build definitions
        # This comes last since it requires information from earlier steps
        definitions = self._buildHeadersAndGlobals()

        return definitions, declarations

    def _buildFakeInitialiser(self):
        """Build fake initialisers for all used OP2 data structures for the
           ROSE source-to-source translator to analyse. Since we don't have
           runtime information about set sizes and the actual data pointers,
           we use dummy values. This function is never called by the Fluidity
           driver routines."""

        func = FunctionDefinition(Void(), 'initialise_rose_')
        func.setExternC(True)

        # The element set is always defined on the Coordinate field
        func.append(InitialisationOp(self.elem_set, opDeclSet(self.elem_set.name())))
        for f in self._fields.values():
            f.buildFakeInitialiser(func, self.elem_set)

        # Make sure only the ROSE frontend sees this function since it would
        # otherwise require OP2 C library functions to be available in the OP2
        # library Fluidity is linked against
        return PreprocessorScope('__EDG__', func)

    def _buildInitialiser(self):

        func = FunctionDefinition(Void(), 'initialise_gpu_')
        func.setExternC(True)

        # OP2 Initialisation
        # FIXME: We don't currently pass argc, argv
        func.append(opInit(Literal(2)))

        return func

    def _buildFinaliser(self):
        func = FunctionDefinition(Void(), 'finalise_gpu_')
        func.setExternC(True)

        func.append(opExit())

        if self._eq.opts['profiling']:
            for name, (assemble_time, solve_time) in self._counters.items():
                func.append(FunctionCall('printf', [Literal(name+" assembly time (ms): %f\\n"), assemble_time]))
                func.append(FunctionCall('printf', [Literal(name+" solve time (ms): %f\\n"), solve_time]))

        return func

    def _buildHeadersAndGlobals(self):
        seq_includes = GlobalScope([Include('op_lib_cpp.h'), Include('op_seq_mat.h'), Include('ufl_utilities.h')])
        rose_includes = GlobalScope([Include('OP2_OXFORD.h')])

        if self._eq.opts['profiling']:
            prof_decl = GlobalScope(PreprocessorScope('__EDG__', rose_includes, seq_includes))
            for assemble_time, solve_time in self._counters.values():
                prof_decl.append(InitialisationOp(assemble_time, Literal(0.0)))
                prof_decl.append(InitialisationOp(solve_time, Literal(0.0)))
            return prof_decl
        else:
            return PreprocessorScope('__EDG__', rose_includes, seq_includes)

    def _buildRunModel(self):

        def makeParameterListAndGetters(form, staticParameters):
            # Figure out which parameters to pass
            params = list(staticParameters)

            for coeff in form.form_data().original_coefficients:
                # find which field this coefficient came from, then get data for that field
                field = self._eq.getInputCoeffName(extractCoordinates(coeff).count())
                f = self._fields[field]
                params.append(opArgDat(f.dat, OpAll, f.map, Literal(f.dim), OpRead))

            return params

        dtp = Variable('dt_pointer', Pointer(Real()))
        func = FunctionDefinition(Void(), 'run_model_', [dtp])
        func.setExternC(True)

        # Get a handle to Fluidity state to pass to extractor functions
        func.append(InitialisationOp(state, FunctionCall('get_state')))

        # Element set (used by all maps) is always defined on the Coordinate field
        coord_field = self._fields['Coordinate']
        for f in self._fields.values():
            f.buildInitialiser(func)
        func.append(Declaration(self.elem_set))
        func.append(AssignmentOp(self.elem_set, MemberAccess(coord_field.map, 'from', True)))

        # If the coefficient is not written back to state, insert a
        # temporary field to solve for
        temp_dats = []
        for fieldname in self._eq.getTmpCoeffNames():
            from_field = self._eq.getFieldFromCoeff(fieldname)
            # Get field data for orginal coefficient (dat, map)
            field = copy(self._fields[from_field])
            call = opDeclVec(field.dat, fieldname)
            field.dat = Variable(fieldname, OpDat)
            func.append(AssignmentOp(Declaration(field.dat), call))
            # The temporary dat has the same associate map as the
            # origin it has been derived from
            self._fields[fieldname] = field
            temp_dats.append(field.dat)

        if self._eq.opts['profiling']:
            # Declar elapsed time
            func.append(InitialisationOp(elapsed, Literal(0.0)))
            self._counters = {}

        for count, forms in self._eq.solves.items():
            # Unpack the bits of information we want
            matform, vecform = forms
            matname, vecname = matform.form_data().name, vecform.form_data().name
            f = self._fields[self._eq.getResultCoeffName(count)]

            if self._eq.opts['profiling']:
                # Performance counter variables
                assemble_time = Variable(f.name+'_assemble_time', Real())
                solve_time = Variable(f.name+'_solve_time', Real())
                self._counters[f.name] = (assemble_time, solve_time)
                # Start timing assembly
                func.append(startTiming())

            # Get sparsity of the field we're solving for
            sparsity = Variable(matname+'_sparsity', OpSparsity)
            call = opDeclSparsity(f.map, f.map, sparsity.name())
            func.append(AssignmentOp(Declaration(sparsity), call))

            # Call the op_par_loops
            dtArg = opArgGbl(dtp, Literal(1), OpInc)

            # Create a matrix
            matrix = Variable(matname+'_mat', OpMat)
            decl = opDeclMat(sparsity, Literal(f.dim), matrix.name())
            func.append(AssignmentOp(Declaration(matrix), decl))
            # Matrix
            # FIXME: should use mappings from the sparsity instead
            # FIXME: we need to use the op_i corresponding to which mapping we pass
            # (i.e. row or column map)
            matArg = opArgMat(matrix, opI(Literal(1)), f.map, opI(Literal(2)), f.map, Literal(f.dim), OpInc)
            arguments = makeParameterListAndGetters(matform, [matArg, dtArg])
            itbounds = (numBasisFunctions(matform.form_data()),)*2
            # FIXME: To properly support multiple integrals, we need to get
            # the mappings per integral
            for _, name in matform.form_data().named_integrals:
                func.append(opParLoop(name, opIterationSpace(self.elem_set, itbounds), arguments))

            # Create the resulting vector
            vector = Variable(vecname+'_vec', OpDat)
            func.append(AssignmentOp(Declaration(vector),
                opDeclVec(f.dat, vector.name())))
            # Vector
            datArg = opArgDat(vector, OpAll, f.map, Literal(f.dim), OpInc)
            arguments = makeParameterListAndGetters(vecform, [datArg, dtArg])
            # FIXME: To properly support multiple integrals, we need to get
            # the mappings per integral
            for _, name in vecform.form_data().named_integrals:
                func.append(opParLoop(name, self.elem_set, arguments))

            if self._eq.opts['profiling']:
                # Stop timing assembly
                func.append(endTiming(assemble_time))
                # Start timing solve
                func.append(startTiming())

            # Solve
            func.append(opSolve(matrix, vector, f.dat))

            if self._eq.opts['profiling']:
                # Stop timing solve
                func.append(endTiming(solve_time))

            # Free temporaries
            func.append(opFreeVec(vector))
            func.append(opFreeMat(matrix))

        for dat in temp_dats:
            func.append(opFreeVec(dat))

        return func

    def _buildReturnFields(self):

        func = FunctionDefinition(Void(), 'return_fields_', [])
        func.setExternC(True)

        # Get a handle to Fluidity state to pass to extractor functions
        if self._eq.getReturnedFieldNames() and self._eq.getResultCoeffNames():
            func.append(InitialisationOp(state, FunctionCall('get_state')))

        # Transfer all fields solved for on the GPU and written back to state
        for rank, field in self._eq.getReturnedFieldNames():
            # Sanity check: only copy back fields that were solved for
            if field in self._eq.getResultCoeffNames():
                var = Variable(field, OpFieldStruct)
                func.append(AssignmentOp(Declaration(var), opExtractField(field, rank)))
                func.append(opFetchData(MemberAccess(var, 'dat')))

        return func

# vim:sw=4:ts=4:sts=4:et
