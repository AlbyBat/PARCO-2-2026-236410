#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>
#include <numeric>
#include <unordered_map>
#include <set>

/*=====================
    DATA STRUCTURES
 =====================*/

/*for storing matrix data in coordinate format */
struct COOMatrix {
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<double> values;
    int num_rows, num_cols, nnz;

    COOMatrix() : num_rows(0), num_cols(0), nnz(0) {}
};

/*for storing the local portion of the matrix in compressed sparse reow format */
struct CSRMatrixLocal {
    std::vector<int> row_ptr;
    std::vector<int> col_indices;
    std::vector<double> values;
    int local_rows;      //number of local rows (cyclic)
    int num_cols;        //number of global columns
    int nnz;
};

/*for managing MPI buffers and indices for nonblocking communication */
struct CommPlan {
    std::vector<int> send_indices;    //local indices of x_local that need to be sent to other ranks
    std::vector<double> send_buffer;  //untermediate buffer to store packed values before the MPI send
    std::vector<int> send_counts;     //number of elements to send to each specific MPI rank
    std::vector<int> send_displs;     //memory offsets in send_buffer for each destination rank

    std::vector<int> recv_counts;     //number of elements to receive from each specific MPI rank
    std::vector<int> recv_displs;     //memory offsets in the x_ghost vector for each source rank
};

/*for storing Matrix Market file metadata */
struct MatrixMarketHeader {
    std::string object, format, field, symmetry;
    int num_rows, num_cols, nnz;
    MatrixMarketHeader() : num_rows(0), num_cols(0), nnz(0) {}
};

/*=====================
    MAIN CLASS
 =====================*/
class DistributedSpMV {
    private:
        int rank, size;
        MPI_Comm comm;

    public:
        DistributedSpMV(MPI_Comm c = MPI_COMM_WORLD) : comm(c) {
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);
        }

/*for parsing the Matrix Market file header and broadcasting dimensions */
    MatrixMarketHeader read_header(const std::string& filename) {
        MatrixMarketHeader header;
        if(rank==0){
            std::ifstream file(filename);
            if(!file.is_open()){ std::cerr<<"Cannot open "<<filename<<"\n"; MPI_Abort(comm,1);}
            std::string line;
            std::getline(file,line);
            std::istringstream iss(line);
            iss >> header.object >> header.format >> header.field >> header.symmetry;
            while(std::getline(file,line)){
                if(line[0]!='%'){
                    std::istringstream iss2(line);
                    iss2 >> header.num_rows >> header.num_cols >> header.nnz;
                    break;
                }
            }
            file.close();
        }
        int header_data[3];
        if(rank==0){header_data[0]=header.num_rows; header_data[1]=header.num_cols; header_data[2]=header.nnz;}
        MPI_Bcast(header_data,3,MPI_INT,0,comm);
        if(rank!=0){header.num_rows=header_data[0]; header.num_cols=header_data[1]; header.nnz=header_data[2];}
        return header;
    }

    /*for sequential reading of the file followed by MPI_Scatterv distribution */
   COOMatrix read_and_distribute(const std::string& filename) {
    MatrixMarketHeader header = read_header(filename);

    COOMatrix local;
    local.num_rows = header.num_rows;
    local.num_cols = header.num_cols;

    std::vector<int> nnz_counts(size, 0), displs(size, 0);
    std::vector<int> all_rows, all_cols;
    std::vector<double> all_vals;

    if (rank == 0) {
        std::ifstream file(filename);
        if (!file.is_open()) { 
            std::cerr << "Cannot open " << filename << "\n"; 
            MPI_Abort(comm, 1); 
        }
        
        std::string line;
        do { std::getline(file, line); } while (line[0] == '%');
        //temporary buffer
        struct Entry { int r, c; double v; };
        std::vector<std::vector<Entry>> temp_buffers(size);

        int r, c; double v;
        for (int i = 0; i < header.nnz; i++) {
            if (!(file >> r >> c >> v)) break;
            r--; c--; //0-based
            int owner = r % size; //cyclic
            temp_buffers[owner].push_back({r, c, v});
            nnz_counts[owner]++;
        }
        file.close();

        all_rows.resize(header.nnz);
        all_cols.resize(header.nnz);
        all_vals.resize(header.nnz);

        int global_idx = 0;
        for (int p = 0; p < size; p++) {
            displs[p] = global_idx;
            for (const auto& e : temp_buffers[p]) {
                all_rows[global_idx] = e.r;
                all_cols[global_idx] = e.c;
                all_vals[global_idx] = e.v;
                global_idx++;
            }
        }
    }

    MPI_Bcast(nnz_counts.data(), size, MPI_INT, 0, comm);
    MPI_Bcast(displs.data(), size, MPI_INT, 0, comm);

    int local_nnz = nnz_counts[rank];
    local.row_indices.resize(local_nnz);
    local.col_indices.resize(local_nnz);
    local.values.resize(local_nnz);

    MPI_Scatterv(rank == 0 ? all_rows.data() : nullptr, nnz_counts.data(), displs.data(), MPI_INT,
                 local.row_indices.data(), local_nnz, MPI_INT, 0, comm);
    MPI_Scatterv(rank == 0 ? all_cols.data() : nullptr, nnz_counts.data(), displs.data(), MPI_INT,
                 local.col_indices.data(), local_nnz, MPI_INT, 0, comm);
    MPI_Scatterv(rank == 0 ? all_vals.data() : nullptr, nnz_counts.data(), displs.data(), MPI_DOUBLE,
                 local.values.data(), local_nnz, MPI_DOUBLE, 0, comm);

    local.nnz = local_nnz;
    return local;
}

    /*for efficient parallel reading using MPI IO */
    COOMatrix read_parallel(const std::string& filename){
        MatrixMarketHeader header = read_header(filename);
        COOMatrix local;
        local.num_rows = header.num_rows;
        local.num_cols = header.num_cols;

        MPI_File fh;
        MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        
        MPI_Offset filesize;
        MPI_File_get_size(fh, &filesize);

        MPI_Offset data_offset = 0;
        if (rank == 0) {
            std::ifstream file(filename);
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty() && line[0] != '%') {
                    data_offset = file.tellg();
                    break;
                }
            }
        }
        MPI_Bcast(&data_offset, 1, MPI_OFFSET, 0, comm);

        MPI_Offset total_data_size = filesize - data_offset;
        MPI_Offset chunk_size = total_data_size / size;
        MPI_Offset start = data_offset + rank * chunk_size;
        MPI_Offset end = (rank == size - 1) ? filesize : data_offset + (rank + 1) * chunk_size;

        char char_before = '\n';
        if (rank > 0) {
            MPI_File_read_at(fh, start - 1, &char_before, 1, MPI_CHAR, MPI_STATUS_IGNORE);
        }

        MPI_Offset local_read_size = end - start;
        std::vector<char> buffer(local_read_size + 1, 0);
        MPI_File_read_at_all(fh, start, buffer.data(), local_read_size, MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&fh);

        std::string content(buffer.data(), local_read_size);
        std::istringstream iss(content);
        std::string line;

        if (char_before != '\n' && char_before != '\r') {
            std::getline(iss, line);
        }

        while (std::getline(iss, line)) {
            if (line.empty()) continue;
            int r, c; double v;
            if (std::istringstream(line) >> r >> c >> v) {
                r--; c--; // 0-based
                
                if (r % size == rank) {//cyclic
                    local.row_indices.push_back(r);
                    local.col_indices.push_back(c);
                    local.values.push_back(v);
                }
            }
        }
        local.nnz = local.row_indices.size();
    return local;
}

/*for converting the local COO data to CSR format with cyclic distribution */
    CSRMatrixLocal coo_to_csr_local(const COOMatrix& coo, int global_nrows, int rank, int size){
        CSRMatrixLocal csr;
        csr.num_cols = coo.num_cols;
        csr.nnz = coo.nnz;

        csr.local_rows = (global_nrows + size - 1 - rank) / size;

        csr.row_ptr.assign(csr.local_rows + 1, 0);
        csr.col_indices.resize(coo.nnz);
        csr.values.resize(coo.nnz);

        // count
        for(int i = 0; i < coo.nnz; i++){
            int gr = coo.row_indices[i];
            if(gr % size == rank){
                int lr = (gr - rank) / size;
                csr.row_ptr[lr + 1]++;
            }
        }

        for(int i = 0; i < csr.local_rows; i++)
            csr.row_ptr[i + 1] += csr.row_ptr[i];

        std::vector<int> offset = csr.row_ptr;
        for(int i = 0; i < coo.nnz; i++){
            int gr = coo.row_indices[i];
            if(gr % size == rank){
                int lr = (gr - rank) / size;
                int pos = offset[lr]++;
                csr.col_indices[pos] = coo.col_indices[i];
                csr.values[pos] = coo.values[i];
            }
        }

        return csr;
    }

/*for the sparse matrix vector multiplication using non blocking communication overlap */
    std::vector<double> spmv(const CSRMatrixLocal& A, const std::vector<double>& x_local, std::vector<double>& x_ghost,
                                 const std::unordered_map<int,int>& ghost_map, CommPlan& plan, int rank, int size, MPI_Comm comm){
        
        std::vector<double> y(A.local_rows, 0.0);
        std::vector<MPI_Request> requests;

        for (int i = 0; i < size; i++) {
            if (plan.recv_counts[i] > 0) {
                MPI_Request r;
                MPI_Irecv(&x_ghost[plan.recv_displs[i]], plan.recv_counts[i], MPI_DOUBLE, i, 10, comm, &r);
                requests.push_back(r);
            }
        }

        for (size_t i = 0; i < plan.send_indices.size(); i++) {
            plan.send_buffer[i] = x_local[plan.send_indices[i]];
        }
        for (int i = 0; i < size; i++) {
            if (plan.send_counts[i] > 0) {
                MPI_Request r;
                MPI_Isend(&plan.send_buffer[plan.send_displs[i]], plan.send_counts[i], MPI_DOUBLE, i, 10, comm, &r);
                requests.push_back(r);
            }
        }

        for (int i = 0; i < A.local_rows; i++) {
            for (int jj = A.row_ptr[i]; jj < A.row_ptr[i+1]; jj++) {
                int col = A.col_indices[jj];
                if (col % size == rank) {
                    y[i] += A.values[jj] * x_local[col / size];
                }
            }
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        for (int i = 0; i < A.local_rows; i++) {
            for (int jj = A.row_ptr[i]; jj < A.row_ptr[i+1]; jj++) {
                int col = A.col_indices[jj];
                if (col % size != rank) {
                    y[i] += A.values[jj] * x_ghost[ghost_map.at(col)];
                }
            }
        }

        return y;
    }


   /*for collecting local result segments into a global vector on rank 0 */
    std::vector<double> gather_results(const std::vector<double>& y_local, int local_rows, int global_nrows, int rank, int size, MPI_Comm comm) {
        std::vector<int> recvcounts(size);
        MPI_Gather(&local_rows, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, comm);

        std::vector<int> displs(size, 0);
        std::vector<double> all_raw_vals;

        if (rank == 0) {
            for (int i = 1; i < size; i++) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
            all_raw_vals.resize(displs[size-1] + recvcounts[size-1]);
        }

        MPI_Gatherv(y_local.data(), local_rows, MPI_DOUBLE,
                    rank == 0 ? all_raw_vals.data() : nullptr,
                    recvcounts.data(), displs.data(), MPI_DOUBLE, 0, comm);

        std::vector<double> y_global;
        if (rank == 0) {
            y_global.assign(global_nrows, 0.0);
            for (int r = 0; r < size; r++) {
                for (int i = 0; i < recvcounts[r]; i++) {
                    int global_idx = r + i * size; 
                    if (global_idx < global_nrows) {
                        y_global[global_idx] = all_raw_vals[displs[r] + i];
                    }
                }
            }
        }
        return y_global;
    }
};

/*for calculating the communication plan, indices, and ghost maps */
    void setup_communication(const CSRMatrixLocal& A,
                                int rank, int size,
                                std::vector<double>& x_ghost,
                                std::unordered_map<int,int>& ghost_map,
                                CommPlan& plan,
                                MPI_Comm comm) {

        std::set<int> needed_cols;
        for (int k = 0; k < A.nnz; k++) {
            if (A.col_indices[k] % size != rank)
                needed_cols.insert(A.col_indices[k]);
        }

        ghost_map.clear();
        plan.recv_counts.assign(size, 0);
        int idx = 0;
        for (int col : needed_cols) {
            ghost_map[col] = idx++;
            plan.recv_counts[col % size]++;
        }
        x_ghost.resize(needed_cols.size());
        
        plan.recv_displs.assign(size + 1, 0);
        for (int i = 0; i < size; i++) 
            plan.recv_displs[i+1] = plan.recv_displs[i] + plan.recv_counts[i];

        plan.send_counts.assign(size, 0);
        MPI_Alltoall(plan.recv_counts.data(), 1, MPI_INT, 
                    plan.send_counts.data(), 1, MPI_INT, comm);

        plan.send_displs.assign(size + 1, 0);
        for (int i = 0; i < size; i++)
            plan.send_displs[i+1] = plan.send_displs[i] + plan.send_counts[i];

        std::vector<int> recv_indices_global(needed_cols.begin(), needed_cols.end());
        std::vector<int> send_indices_global(plan.send_displs[size]);

        MPI_Alltoallv(recv_indices_global.data(), plan.recv_counts.data(), plan.recv_displs.data(), MPI_INT,
                    send_indices_global.data(), plan.send_counts.data(), plan.send_displs.data(), MPI_INT, comm);


        plan.send_indices.resize(send_indices_global.size());
        for (size_t i = 0; i < send_indices_global.size(); i++) {
            plan.send_indices[i] = send_indices_global[i] / size; //cyclic
        }
        plan.send_buffer.resize(plan.send_indices.size());
    }

/*for analyzing the workload distribution across ranks */
    void gather_load_balance_info(int rank, int size, int local_nnz) {
        std::vector<int> all;
        if (rank==0) all.resize(size);
        MPI_Gather(&local_nnz,1,MPI_INT,rank==0?all.data():nullptr,1,MPI_INT,0,MPI_COMM_WORLD);

        if(rank==0){
            int min = *std::min_element(all.begin(),all.end());
            int max = *std::max_element(all.begin(),all.end());
            double avg = std::accumulate(all.begin(),all.end(),0.0)/size;
            std::cout << "\nLoad Balance Analysis:\n";
            std::cout << "Min NNZ per rank: " << min << "\n";
            std::cout << "Max NNZ per rank: " << max << "\n";
            std::cout << "Avg NNZ per rank: " << avg << "\n";
            std::cout << "Imbalance ratio: " << (max-min)/avg << "\n";
        }
    }

    #include <random>

/*for generating a synthetic random matrix for weak scaling tests 
(Nlocal*rank) rows *   (Nlocal*rank) columns */  
COOMatrix generate_synthetic_coo(int local_rows_per_rank, int global_cols, int nnz_per_row, int rank, int size) {
    COOMatrix local;
    local.num_rows = local_rows_per_rank * size; // Total global rows
    local.num_cols = global_cols;
    
    //seed with rank
    std::mt19937 gen(1337 + rank);
    std::uniform_int_distribution<int> col_dist(0, global_cols - 1);
    std::uniform_real_distribution<double> val_dist(0.1, 1.0);

    for (int i = 0; i < local_rows_per_rank; i++) {
        //global row index based on cyclic distribution again
        int global_row = rank + i * size;
        
        //random column indices for this row
        for (int j = 0; j < nnz_per_row; j++) {
            local.row_indices.push_back(global_row);
            local.col_indices.push_back(col_dist(gen));
            local.values.push_back(val_dist(gen));
        }
    }
    local.nnz = local.row_indices.size();
    return local;
}

/* =====================
         

            MAIN



 =====================*/
int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);


    DistributedSpMV dsp;
    COOMatrix local_coo;

    if(argc < 2){
        int local_rows_base = 100000; //workload per rank
        int nnz_per_row = 1;
        int global_cols = local_rows_base * size; 
        if(rank == 0) std::cout << "Running Weak Scaling with " << local_rows_base << " rows per rank\n";
        local_coo = generate_synthetic_coo(local_rows_base, global_cols, nnz_per_row, rank, size);
    }

    /* =======================
            SEQ. READ
       ======================= */
    double read_time = 0.0,t0 = 0.0;
    if (argc >= 2) {   
    t0 = MPI_Wtime();
    local_coo = dsp.read_and_distribute(argv[1]);//loads the matrix using the scatterv method
    read_time = MPI_Wtime() - t0;
    }
    /* =======================
            COO --> CSR 
       ======================= */
    double t1 = MPI_Wtime();
    CSRMatrixLocal local_csr = dsp.coo_to_csr_local(local_coo, local_coo.num_rows, rank, size);//converts the local data to CSR format
    double convert_time = MPI_Wtime() - t1;
    int global_ncols = local_csr.num_cols;
    MPI_Bcast(&global_ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* =======================
      SETUP VECTOR AND COMMS
       ======================= */
    int local_x_size = (global_ncols + size - 1 - rank) / size;
    std::vector<double> x_local(local_x_size, 1.0);//all values 1.0

    std::vector<double> x_ghost;
    std::unordered_map<int,int> ghost_map;
    CommPlan plan; 
    setup_communication(local_csr, rank, size, x_ghost, ghost_map, plan, MPI_COMM_WORLD);//prepares the MPI communication plan once for all execution

    std::vector<double> y_local(local_csr.local_rows, 0.0);

    /* =======================
                SPMV
       ======================= */
    int iterations = 20;
    MPI_Barrier(MPI_COMM_WORLD);

    double t_spmv_start = MPI_Wtime();
    for(int i = 0; i < iterations; i++) {
        y_local = dsp.spmv(local_csr,
                                x_local,
                                x_ghost,
                                ghost_map,
                                plan,
                                rank,
                                size,
                                MPI_COMM_WORLD);//executes the iterative benchmark with communication overlap
    }
    double t_spmv_end = MPI_Wtime();

    double avg_spmv_time_local = (t_spmv_end - t_spmv_start) / iterations;

    /* =======================
       SLOWEST SPMV TIME (GLOBAL)
       ======================= */
    double max_avg_spmv_time = 0.0;
    MPI_Allreduce(&avg_spmv_time_local,
                  &max_avg_spmv_time,
                  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    /* =======================
       GATHER WITH GLOBAL ROW ORDER
       ======================= */
    double t_gather_start = MPI_Wtime();
    std::vector<double> y_global =
        dsp.gather_results(y_local,
                           local_csr.local_rows,
                           local_coo.num_rows,
                           rank,
                           size,
                           MPI_COMM_WORLD);//distributed result back to rank 0

    double gather_time = MPI_Wtime() - t_gather_start;

    /* =======================
                 NNZ
       ======================= */
    long local_nnz = static_cast<long>(local_csr.nnz);
    long global_nnz = 0;
    MPI_Reduce(&local_nnz, &global_nnz,
               1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    /* =======================
       MEMORY AND COMMUNICATION CALC.
       ======================= */
    size_t mem_bytes =
        local_csr.row_ptr.size()   * sizeof(int) +
        local_csr.col_indices.size()* sizeof(int) +
        local_csr.values.size()    * sizeof(double) +
        x_local.size()                   * sizeof(double) +
        y_local.size()             * sizeof(double);

    double mem_mb = mem_bytes / (1024.0 * 1024.0);
    double sent_kb = (local_csr.local_rows * sizeof(double)) / 1024.0;

    double total_comp_comm = avg_spmv_time_local + gather_time; 
    double comp_percentage = (avg_spmv_time_local / total_comp_comm) * 100.0; 
    double comm_percentage = (gather_time / total_comp_comm) * 100.0;

    /* =======================
       PARALLEL READ MPI IO
       ======================= */
       double read_par_time = 0.0, t_par=0.0;
    if (argc >= 2) {    
    MPI_Barrier(MPI_COMM_WORLD);
    t_par = MPI_Wtime();
    dsp.read_parallel(argv[1]);//performs the MPI IO parallel read benchmark
    read_par_time = MPI_Wtime() - t_par;
    }
    /* =======================
       OUTPUT IN ORDER
       ======================= */
    if(rank == 0){
        std::cout << "\n=== Distributed SpMV Performance ===\n";
        std::cout << "Processes:          " << size << "\n";
        std::cout << "Matrix dimensions:  "
                  << local_coo.num_rows << " x "
                  << local_csr.num_cols << "\n";
        std::cout << "Global NNZ:         " << global_nnz << "\n\n";
        std::cout << "Breakdown: " << comp_percentage << "% Comp, " << comm_percentage << "% Comm\n";
        std::cout << "Read time (seq):    " << read_time << " s\n";
        std::cout << "Read time (MPI-IO): " << read_par_time << " s\n";
        std::cout << "Conversion time:    " << convert_time << " s\n";
        std::cout << "SpMV time (max):    " << max_avg_spmv_time << " s\n";
        std::cout << "Gather time:        " << gather_time << " s\n";
        std::cout << "Total time:         " << (MPI_Wtime()-t0) << " s\n";

        if(!y_global.empty()){
            std::cout << "\nFirst 10 elements of result vector:\n";
            for(int i = 0; i < std::min(10,(int)y_global.size()); i++)
                std::cout << y_global[i] << " ";
            std::cout << "\n";
        }

        double gflops =
            (2.0 * global_nnz) / (max_avg_spmv_time * 1e9);

        std::cout << "\n=== Performance Metrics ===\n";
        std::cout << "Global Performance: " << gflops << " GFLOP/s\n";

        if (argc > 2) {
            double T1_reference = std::atof(argv[2]);
            //speedup on spmv time with 1p, calculated on different run and given in parameters
            double speedup = T1_reference / max_avg_spmv_time;
            double efficiency = (speedup / size) * 100.0;

            std::cout << "Reference T1:       " << T1_reference << " s\n";
            std::cout << "Real Speedup:       " << speedup << "x\n";
            std::cout << "Parallel Efficiency:" << efficiency << " %\n";
        }
    }

    /* =======================
       LOAD BALANCE BETWEEN P.
       ======================= */
    gather_load_balance_info(rank, size, local_csr.nnz);//reports the nnz distribution and other

    /* =======================
       EACH RANK SPECIFIC OUTPUT
       ======================= */
    std::cout << "[Rank " << rank << "] "
              << "Local NNZ: " << local_csr.nnz
              << ", Memory: " << mem_mb << " MB"
              << ", Sent: " << sent_kb << " KB"
              << ", MPI-IO Read: " << read_par_time << " s\n";

    MPI_Finalize();//shut down mpi
    return 0;
}
