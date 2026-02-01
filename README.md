# PARCO-2-2026-236410

# deliverable2-ParCo
# Distributed Sparse Matrix Vector Multiplication (SpMV) Benchmark (MPI)

This project implements a distributed Sparse Matrix Vector Multiplication SpMV algorithm using the CSR (Compressed Sparse Row) format and the MPI standard for distributed memory systems.

## Features

- **Distributed SpMV Implementation**: MPI parallelization with row distribution based on the number of non-zero (NNZ) elements to ensure optimal load balancing.
- **CSR Format**: Efficient handling of large-scale sparse matrices.
- **Scaling Analysis**: Support for both Strong Scaling (fixed problem size, increasing processes) and Weak Scaling (constant load per process).
- **Real World Matrix Support**: Parser for the Matrix Market (`.mtx`) format, tested on SuiteSparse datasets (`apache2`, `G3_circuit`).
- **Performance Metrics**: Detailed timing breakdown (Computation vs. Communication), GFLOPS, speedup, parallel efficiency and more.

## Local Compilation and Execution

Ensure you have an MPI environment installed (OpenMPI).

### Compilation
To compile the project with O3 optimization:
```bash
mpicxx -O3 -o deliverable2.x deliverable2.cpp
```

## Execution

### 1. Local Execution (WSL/Linux)
To run the benchmark on your local machine, use the `mpirun` command. You must specify the number of processes and the path to the Matrix Market file.

```bash
# Basic execution with 4 processes and 1NP reference of apache2
mpirun -np 4 ./deliverable2.x apache2.mtx 0.0448266
# Execution with 8 processes with G3_circuit, doesn't compare with 1NP because reference wasnt given in parameters
mpirun -np 8 ./deliverable2.x G3_circuit.mtx 
# Execution with 32 processes with a synthetic matrix
mpirun -np 32 ./deliverable2.x  
```

## HPC Deployment (Unitn Cluster)


The benchmark is designed to run on a cluster using the PBS  scheduler. Resource allocation is handled via `.pbs` script files.
Follow these steps to transfer, compile, and execute the project on the HPC cluster.

### 1. Project Setup
Log in to the cluster and create a dedicated directory for the deliverable:
```bash
# On the HPC terminal
mkdir deliverable2
cd deliverable2
# On your local machine
scp deliverable2.cpp *.pbs username@hpc.unitn.it:/home/deliverable2/

### Compilation (HPC terminal)
To compile the project with O3 optimization:
module load gcc91
module load openmpi-4.0.4
mpicxx -O3 -o deliverable2.x deliverable2.cpp
module purge
```


### 1. Job Script Configuration

Examples:
```bash
#Submitting the job with apache2 matrix, 32 NP and 1NP reference I measured
qsub del2p32.pbs
#Submitting the job with G3_circuit matrix, 2 NP and 1NP reference I measured
qsub del2p2G3.pbs
#Submitting the job with synthetic matrix, 48 NP
qsub del2p48ws.pbs

#Check job status
qstat jobid
```

### Check output of run
```bash
cat deliverable2.out
```
## Or errors
```bash
cat deliverable2.err
```