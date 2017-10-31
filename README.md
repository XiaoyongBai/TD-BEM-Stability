# TD-BEM-Stability
Eigenvalue analysis tools to evaluate time-domain BEM's time marching stability. 

### External libraries ###
* MPI for parallel computing
* Slepc to solve for Eigenvalue
* Slepc depends on Petsc

### Work Flow ###
* Step 1: read the boundary element matrices, i.e., G and H matrices
* Step 2: Form the hybrid amplification matrix
* Step 3: Find the spectral radius of the amplification matrix
* Step 4: output the spectral radius, i.e., the eigenvalue with the largest absolute value

