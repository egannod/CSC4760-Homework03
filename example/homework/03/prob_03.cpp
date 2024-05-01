
#include <iostream>
#include <mpi.h>

int main(int argc, char** argv)
{
    int rank, size;
    int color1, color2;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Number of rows and columns
    int P = 4; //rows
    int Q = 2; //columns
    int M = 8; // length of vector x
    
    // Calculate process coordinates
    int row = rank / Q;
    int col = rank % Q;

    // calculate local vector size
    int local_size = (M + P - 1) / P;

    // Allocate memory for local vector
    int* local_x = (rank == 0) ? new int[M] : new int[local_size];

    // Create row and column communicators
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, rank, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, rank, &col_comm);

    // Initialize local vector for process 0
    if (rank == 0)
    {
        for (int i = 0; i < M; i++)
        {
            local_x[i] = i+1;
        }
    }


    // Scatter vector x adjacent column
    MPI_Scatter(local_x, local_size, MPI_INT, local_x, local_size, MPI_INT, 0, col_comm);

    // copy local vector to all processes in the same row
    MPI_Bcast(local_x, local_size, MPI_INT, 0, row_comm);

    // define local vector y
    int* local_y = new int[local_size];  
    // intialize local vector y to all zeros
    for (int i = 0; i < local_size; i++)
    {
        local_y[i] = 0;
    }
    std::cout << "Rank: " << rank << " row: " << row << " col: " << col << std::endl;
    for (int i=0; i < local_size; i++)
    {
    // x local to global: given that this element is (p,i), what is its global index I?
    int I = i*rank + rank;

    // so to what (qhat,jhat) does this element of the original global vector go?
    int qhat = I%Q;
    int jhat = I/Q;

    if(qhat == col)  // great, this process has an element of y!
    { 
        local_y[jhat] = local_x[i];
    }
    }

    // use MPI_AllReduce to gather all local_y vectors into global_y
    int* global_y = new int[local_size];
    MPI_Allreduce(local_y, global_y, local_size, MPI_INT, MPI_SUM, col_comm);

    // Print local vector
    std::cout << "local_x: ";
    for (int i = 0; i < local_size; i++)
    {
        std::cout << local_x[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "local_y: ";
    for (int i = 0; i < local_size; i++)
    {
        std::cout << local_y[i] << " ";
    }
    std::cout << std::endl;

    // Print global vector
    std::cout << "global_y: ";
    for (int i = 0; i < local_size; i++)
    {
        std::cout << global_y[i] << " ";
    }
    std::cout << std::endl;

    // Free communicators and memory
    delete[] local_x;
    delete[] local_y;
    delete[] global_y;

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}