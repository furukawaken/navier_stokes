#!/bin/bash

CXXFLAGS="-O2 -Xpreprocessor -fopenmp -lomp"
VERSION="-std=c++17"
NETCDF_INCLUDE_PATH="-I/opt/homebrew/Cellar/netcdf/4.9.2_1/include"
NETCDF_LIB="-L/opt/homebrew/Cellar/netcdf/4.9.2_1/lib"
SOURCE_FILES="main.cpp ../src/BoundaryConditionChecker.cpp ../src/BCModificationToolsNeumann.cpp ../src/BoundaryForcingBuilderCellCentered.cpp ../src/BoundaryForcingBuilderStaggered.cpp ../src/BoundaryOperations.cpp ../src/DataWriter.cpp ../src/IndexMapping.cpp ../src/InitialFunctionFactory.cpp ../src/InitialVelocityFactory.cpp ../src/MathFunctions.cpp ../src/Parameters.cpp ../src/ParameterConverter.cpp ../src/PressureManager.cpp ../src/SteadyForcingFactory.cpp ../src/TripletListFactory.cpp"
EXECUTABLE="main.exe"
COMPILATION_LOG="../exe/compilation.log"
EXECUTION_LOG="../exe/out.log"
ERROR_LOG="../exe/error.log"

cd ../src

# Compile with debugging symbols and redirect compilation output and errors
clang++ $CXXFLAGS $VERSION $NETCDF_INCLUDE_PATH $NETCDF_LIB -lnetcdf -o $EXECUTABLE $SOURCE_FILES > $COMPILATION_LOG 2> $ERROR_LOG

# Check compilation status
if [ $? -ne 0 ]; then
    echo "Compilation failed. Check the error log for details." | tee -a $COMPILATION_LOG
    exit 1
fi

# Run the program and redirect output and errors
./$EXECUTABLE > $EXECUTION_LOG 2>> $ERROR_LOG

# Check execution status
if [ $? -ne 0 ]; then
    echo "Execution failed. Check the error log for details." | tee -a $ERROR_LOG
    exit 1
fi

echo "Execution successful."  # Optionally, you can add this line to indicate successful execution.