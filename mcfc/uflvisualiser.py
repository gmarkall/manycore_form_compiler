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

import visualiser

ASTVisualiser = visualiser.ASTVisualiser
ReprVisualiser = visualiser.ReprVisualiser

class ObjectVisualiser(visualiser.ObjectVisualiser):

    def _ignoreattr(self, attr):
        # We have to ignore 'Transposed' to prevent an infinite recursion
        # on a rank 2 tensor (i.e. taking the transpose of a transpose of
        # a transpose... - you get the picture), so ignore attribute 'T'
        # Ignore attributes ending in *_domains since they are deprecated
        return attr == 'T' or attr.endswith('_domains')

# vim:sw=4:ts=4:sts=4:et
