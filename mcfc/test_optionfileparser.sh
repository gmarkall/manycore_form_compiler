#! /bin/bash

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

extensions=(flml swml adml)
usage="usage: $0 FLUIDITY_DIR\n\n
  or set the environment variable FLUIDITY_DIR to point to your\n
  Fluidity source directory"

[[ $1 ]] && FLUIDITY_DIR=$1
[[ -z $FLUIDITY_DIR ]] && echo -e $usage && exit -1

for ext in $extensions; do
  for f in $FLUIDITY_DIR/tests/*/*.$ext; do
    python optionfileparser.py $f
    if [[ $? != 0 ]]; then
      echo $f >> failed
      #echo Failed for $f
      #exit -1
    fi
  done
done

