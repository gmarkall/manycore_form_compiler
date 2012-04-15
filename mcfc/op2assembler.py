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

# MCFC libs
from assembler import *
from codegeneration import *
from formutils import extractCoordinates

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
opDeclVec = lambda origin, name: \
    FunctionCall('op_decl_vec', [origin, Literal(name)])
opDeclSparsity = lambda rowmap, colmap, name: \
    FunctionCall('op_decl_sparsity', [rowmap, colmap, Literal(name)])
# FIXME: Default to double for now
opDeclMat = lambda sparsity, dim, name: \
    FunctionCall('op_decl_mat', [sparsity, dim, Literal('double'), Literal(8), Literal(name)])
opFreeVec = lambda vec: \
    FunctionCall('op_free_vec', [vec])
opFreeMat = lambda mat: \
    FunctionCall('op_free_mat', [mat])
opArgDat = lambda dat, index, mapping, access: \
    FunctionCall('op_arg_dat', [dat, index, mapping, access])
opArgMat = lambda mat, rowindex, rowmap, colindex, colmap, access: \
    FunctionCall('op_arg_mat', [mat, rowindex, rowmap, colindex, colmap, access])
opFetchData = lambda dat: FunctionCall('op_fetch_data', [dat])
opParLoop = lambda kernel, iterationset, arguments: \
    FunctionCall('op_par_loop', [FunctionPointer(kernel), Literal(kernel), iterationset] + arguments)
opSolve = lambda A, b, x: FunctionCall('op_solve', [A, b, x])

# Opaque pointer to fluidity state
state = Variable('state', Pointer(Void()))
rank2type = { 0: 'scalar', 1: 'vector', 2: 'tensor' }
# Fluidity OP2 state functions
opExtractField = lambda fieldname, rank, codim=0: \
        FunctionCall('extract_op_%s_field' % rank2type[rank], [state, fieldname, Literal(codim)])

def extractOpFieldData(scope, field, rank):
    # Get OP2 data structures
    var = Variable(field, OpFieldStruct)
    # FIXME: We stupidly default to requesting co-dimension 0.
    # This should be infered from the integral's measure.
    scope.append(InitialisationOp(var, opExtractField(Literal(field), rank)))
    return Member(var, 'dat'), Member(var, 'map')

class Op2AssemblerBackend(AssemblerBackend):

    def compile(self, equation):

        self._eq = equation

        # Build declarations
        declarations = GlobalScope()
        declarations.append(self._buildInitialiser())
        declarations.append(self._buildFinaliser())
        declarations.append(self._buildRunModel())
        declarations.append(self._buildReturnFields())

        # Build definitions
        # This comes last since it requires information from earlier steps
        definitions = self._buildHeadersAndGlobals()

        return definitions, declarations

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

        return func

    def _buildHeadersAndGlobals(self):
        scope = GlobalScope()
        scope.append(Include('op_lib_cpp.h'))
        scope.append(Include('op_seq_mat.h'))
        scope.append(Include('ufl_utilities.h'))

        return scope

    def _buildRunModel(self):

        def makeParameterListAndGetters(form, staticParameters):
            # Figure out which parameters to pass
            params = list(staticParameters)

            for coeff in form.form_data().original_coefficients:
                # find which field this coefficient came from, then get data for that field
                field = self._eq.getInputCoeffName(extractCoordinates(coeff).count())
                mdat, mmap = field_data[field]
                params.append(opArgDat(mdat, OpAll, mmap, OpRead))

            return params

        dtp = Variable('dt_pointer', Pointer(Real()))
        func = FunctionDefinition(Void(), 'run_model_', [dtp])
        func.setExternC(True)

        # Get a handle to Fluidity state to pass to extractor functions
        func.append(InitialisationOp(state, FunctionCall('get_state')))

        # op_field_data struct per field solved for
        field_data = {}
        # Extract op_dat, op_map for accessed fields
        for rank, field in self._eq.state.accessedFields().values():
            field_data[field] = extractOpFieldData(func, field, rank)

        # If the coefficient is not written back to state, insert a
        # temporary field to solve for
        temp_dats = []
        for field in self._eq.getTmpCoeffNames():
            # Get field data for orginal coefficient (dat, map)
            orig_data = field_data[self._eq.getFieldFromCoeff(field)]
            datVar = Variable(field, OpDat)
            call = opDeclVec(orig_data[0], field)
            func.append(AssignmentOp(Declaration(datVar), call))
            # The temporary dat has the same associate map as the
            # origin it has been derived from
            field_data[field] = datVar, orig_data[1]
            temp_dats.append(datVar)

        for count, forms in self._eq.solves.items():
            # Unpack the bits of information we want
            matform, vecform = forms
            matname = matform.form_data().name
            vecname = vecform.form_data().name
            mdat, mmap = field_data[self._eq.getResultCoeffName(count)]

            # Get sparsity of the field we're solving for
            sparsity = Variable(matname+'_sparsity', OpSparsity)
            call = opDeclSparsity(mmap, mmap, sparsity.name())
            func.append(AssignmentOp(Declaration(sparsity), call))

            # Call the op_par_loops

            # Create a matrix
            matrix = Variable(matname+'_mat', OpMat)
            decl = opDeclMat(sparsity, Member(mdat, 'dim'), matrix.name())
            func.append(AssignmentOp(Declaration(matrix), decl))
            # Matrix
            # FIXME: should use mappings from the sparsity instead
            matArg = opArgMat(matrix, OpAll, mmap, OpAll, mmap, OpInc)
            arguments = makeParameterListAndGetters(matform, [matArg])
            func.append(opParLoop(matname, Member(mmap, 'from'), arguments))

            # Create the resulting vector
            vector = Variable(vecname+'_vec', OpDat)
            func.append(AssignmentOp(Declaration(vector),
                opDeclVec(mdat, vector.name())))
            # Vector
            datArg = opArgDat(vector, OpAll, mmap, OpInc)
            arguments = makeParameterListAndGetters(vecform, [datArg])
            func.append(opParLoop(vecname, Member(mmap, 'from'), arguments))

            # Solve
            func.append(opSolve(matrix, vector, mdat))

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
                func.append(AssignmentOp(Declaration(var), opExtractField(Literal(field), rank)))
                func.append(opFetchData(Member(var, 'dat')))

        return func

# vim:sw=4:ts=4:sts=4:et
