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
OpAll = Constant('OP_ALL')
OpInc = Constant('OP_INC')
OpRead = Constant('OP_READ')

# OP2 functions
opInit = lambda diags: FunctionCall('op_init', [Literal(0), Literal(0), diags])
opExit = lambda : FunctionCall('op_exit', [])
opCloneDat = lambda origin, name: \
    FunctionCall('op_clone_dat', [origin, Literal(name)])
opDeclSparsity = lambda rowmap, colmap: \
    FunctionCall('op_decl_sparsity', [rowmap, colmap])
opDeclMat = lambda sparsity: \
    FunctionCall('op_decl_mat', [sparsity])
opArgDat = lambda dat, index, mapping, access: \
    FunctionCall('op_arg_dat', [dat, index, mapping, access])
opArgMat = lambda mat, rowindex, rowmap, colindex, colmap, access: \
    FunctionCall('op_arg_mat', [mat, rowindex, rowmap, colindex, colmap, access])
opFetchData = lambda dat: FunctionCall('op_fetch_data', [dat])
opParLoop = lambda kernel, iterationset, arguments: \
    FunctionCall('op_par_loop', [FunctionPointer(kernel), Literal(kernel), iterationset] + arguments)
opSolve = lambda A, b, x: FunctionCall('op_solve', [A, b, x])

# Fluidity OP2 state functions
opGetDat = lambda fieldname: FunctionCall('get_op_dat', [fieldname])
opSetDat = lambda fieldname, dat: FunctionCall('set_op_dat', [fieldname, dat])
opGetMap = lambda fieldname: FunctionCall('get_op_map', [fieldname])
opGetSet = lambda fieldname: FunctionCall('get_op_set', [fieldname])
opGetElementSet = lambda : FunctionCall('get_op_element_set', [])

def extractOpFieldData(scope, field):
    # Get op_dat
    datVar = Variable(field+'_data', OpDat)
    scope.append(AssignmentOp(datVar, opGetDat(Literal(field))))
    # Get op_map
    mapVar = Variable(field+'_map', OpMap)
    scope.append(AssignmentOp(mapVar, opGetMap(Literal(field))))
    # Get op_set (currently unused)
    setVar = Variable(field+'_set', OpSet)
    scope.append(AssignmentOp(setVar, opGetSet(Literal(field))))

    return datVar, mapVar, setVar

# Global Variables
elements         = Variable('elements',          OpSet)

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

        # Get element set
        func.append(AssignmentOp(elements, opGetElementSet()))

        # Triplets of op_dat, op_map, op_set per field solved for
        self._field_data = {}
        # Extract op_dat, op_map, op_set for accessed fields
        # FIXME: Do we need different call for different ranks?
        for rank, field in self._eq.state.accessedFields().values():
            self._field_data[field] = extractOpFieldData(func, field)

        # If the coefficient is not written back to state, insert a
        # temporary field for it
        for field in self._eq.getTmpCoeffNames():
            # Get field data for orginal coefficient (dat, map, set)
            orig_data = self._field_data[self._eq.getFieldFromCoeff(field)]
            datVar = Variable(field, OpDat)
            call = opCloneDat(orig_data[0], field)
            func.append(AssignmentOp(datVar, call))
            # The temporary dat has the same associate map and set as the
            # origin it has been derived from
            self._field_data[field] = datVar, orig_data[1], orig_data[2]

        # Tuples of sparsity, matrix per field solved for
        self._solve_data = {}
        # FIXME: This will stupidly create a sparsity for each coefficient
        # solved for (which is not necessary in the general case)
        for field in self._eq.getResultCoeffNames():

            # Get sparsity of the field we're solving for
            sparsity = Variable(field+'_sparsity', OpSparsity)
            fieldmap = self._field_data[field][1]
            call = opDeclSparsity(fieldmap, fieldmap)
            func.append(AssignmentOp(sparsity, call))

            # Create a matrix
            matrix = Variable(field+'_mat', OpMat)
            func.append(AssignmentOp(matrix, opDeclMat(sparsity)))

            # Create the resulting vector
            vector = Variable(field+'_vec', OpDat)
            func.append(AssignmentOp(vector,
                opCloneDat(self._field_data[field][0], vector.name())))

            # Remember the sparsity variable information
            self._solve_data[field] = sparsity, matrix, vector

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

        # Declare vars in global scope:
        # op_dat, op_map, op_set for fields; op_sparsity, op_mat for solves
        scope.append(Declaration(elements))
        # We need to make sure declarations are unique
        declared_vars = set()
        for data in self._field_data.values() + self._solve_data.values():
            for var in data:
                if var not in declared_vars:
                    scope.append(Declaration(var))
                    declared_vars.add(var)

        return scope

    def _buildRunModel(self):

        dtp = Variable('dt_pointer', Pointer(Real()))
        func = FunctionDefinition(Void(), 'run_model_', [dtp])
        func.setExternC(True)

        for count, forms in self._eq.solves.items():
            # Unpack the bits of information we want
            result = self._eq.getResultCoeffName(count)
            matrixform, vectorform = forms
            sparsity, matrix, vector = self._solve_data[result]
            mdat, mmap, _ = self._field_data[result]

            # Call the op_par_loops

            # Matrix
            # FIXME: should use mappings from the sparsity instead
            matArg = opArgMat(matrix, OpAll, mmap, OpAll, mmap, OpInc)
            arguments = self._makeParameterListAndGetters(matrixform, [matArg])
            func.append(opParLoop(matrixform.form_data().name, elements, arguments))

            # Vector
            datArg = opArgDat(vector, OpAll, mmap, OpInc)
            arguments = self._makeParameterListAndGetters(vectorform, [datArg])
            func.append(opParLoop(vectorform.form_data().name, elements, arguments))

            # Solve
            func.append(opSolve(matrix, vector, mdat))

        return func

    def _buildReturnFields(self):

        func = FunctionDefinition(Void(), 'return_fields_', [])
        func.setExternC(True)
        # Transfer all fields solved for on the GPU and written back to state
        for field in self._eq.getReturnedFieldNames():
            # Sanity check: only copy back fields that were solved for
            if field in self._eq.getResultCoeffNames():
                dat = self._field_data[field][0]
                func.append(opFetchData(dat))
                func.append(opSetDat(Literal(field), dat))

        return func

    def _makeParameterListAndGetters(self, form, staticParameters):
        # Figure out which parameters to pass
        params = list(staticParameters)

        for coeff in form.form_data().original_coefficients:
            # find which field this coefficient came from, then get data for that field
            field = self._eq.getInputCoeffName(extractCoordinates(coeff).count())
            mdat, mmap, _ = self._field_data[field]
            params.append(opArgDat(mdat, OpAll, mmap, OpRead))

        return params

# vim:sw=4:ts=4:sts=4:et
