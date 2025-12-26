#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <vector>
#include <immintrin.h>
#include <mpi.h>

using namespace std;

constexpr double DT = 0.001;
constexpr double G  = 1.0;
constexpr double SOFT = 0.1;
constexpr int SEED = 1234;

enum Mode { BENCH, VISUAL };

// Structure to bundle data for MPI transmission
// We send x, y, z, m together to minimize MPI call overhead
struct ParticleData {
    double x, y, z, m;
};

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc < 4) {
        if(rank == 0) cerr << "Usage: " << argv[0] << " <N> <STEPS> <bench|visual> [SAVE_INTERVAL]\n";
        MPI_Finalize();
        return 1;
    }

    const int N_GLOBAL = atoi(argv[1]);
    const int STEPS = atoi(argv[2]);
    
    Mode mode = (strcmp(argv[3], "visual") == 0) ? VISUAL : BENCH;
    int saveInterval = (mode == VISUAL && argc > 4) ? atoi(argv[4]) : 1;

    // --- 1. Load Balancing (Arbitrary N support) ---
    // Calculate how many particles each rank gets.
    int base_N = N_GLOBAL / size;
    int remainder = N_GLOBAL % size;
    
    int my_N = base_N + (rank < remainder ? 1 : 0);
    
    // Calculate global offsets (needed for consistent seeding and output)
    std::vector<int> all_counts(size);
    std::vector<int> all_displs(size);
    
    // Each rank calculates the distribution for everyone to know offsets
    int current_disp = 0;
    for(int r=0; r<size; r++) {
        all_counts[r] = base_N + (r < remainder ? 1 : 0);
        all_displs[r] = current_disp;
        current_disp += all_counts[r];
    }
    int my_offset = all_displs[rank];
    
    // Determine the maximum N any rank has (for buffer allocation)
    int max_N = base_N + (remainder > 0 ? 1 : 0);

    // --- 2. Memory Allocation (Aligned for AVX-512) ---
    // Local particles (Updated every step)
    size_t bytes = my_N * sizeof(double);
    if (bytes % 64 != 0) bytes += 64 - (bytes % 64);

    double* __restrict__ x  = (double*) aligned_alloc(64, bytes);
    double* __restrict__ y  = (double*) aligned_alloc(64, bytes);
    double* __restrict__ z  = (double*) aligned_alloc(64, bytes);
    double* __restrict__ vx = (double*) aligned_alloc(64, bytes);
    double* __restrict__ vy = (double*) aligned_alloc(64, bytes);
    double* __restrict__ vz = (double*) aligned_alloc(64, bytes);
    double* __restrict__ m  = (double*) aligned_alloc(64, bytes);

    // Buffers for the Ring Algorithm (Incoming particles from other ranks)
    // We strictly use Struct of Arrays (SoA) for computation, but for MPI 
    // it is easier to send an Array of Structs (AoS) or packed arrays.
    // Here we will use packed arrays to keep the AVX loops efficient.
    
    size_t buf_bytes = max_N * sizeof(double);
    if (buf_bytes % 64 != 0) buf_bytes += 64 - (buf_bytes % 64);

    // Two buffers: One to send, One to receive
    double* send_x = (double*) aligned_alloc(64, buf_bytes);
    double* send_y = (double*) aligned_alloc(64, buf_bytes);
    double* send_z = (double*) aligned_alloc(64, buf_bytes);
    double* send_m = (double*) aligned_alloc(64, buf_bytes);

    double* recv_x = (double*) aligned_alloc(64, buf_bytes);
    double* recv_y = (double*) aligned_alloc(64, buf_bytes);
    double* recv_z = (double*) aligned_alloc(64, buf_bytes);
    double* recv_m = (double*) aligned_alloc(64, buf_bytes);

    // --- 3. Initialization ---
    // To match serial output exactly, we must run the RNG sequence 
    // for all preceding particles.
    srand(SEED);
    
    // Skip RNG states for particles before my_offset
    for(int i=0; i<my_offset; i++) {
        rand(); rand(); rand(); // x,y,z
        rand(); rand(); rand(); // vx,vy,vz
        rand();                 // m
    }

    // Initialize my particles
    for(int i=0; i<my_N; i++){
        x[i]=rand()/(double)RAND_MAX*2.0-1.0;
        y[i]=rand()/(double)RAND_MAX*2.0-1.0;
        z[i]=rand()/(double)RAND_MAX*2.0-1.0;
        vx[i]=rand()/(double)RAND_MAX-0.5;
        vy[i]=rand()/(double)RAND_MAX-0.5;
        vz[i]=rand()/(double)RAND_MAX-0.5;
        m[i]=rand()/(double)RAND_MAX*0.9+0.1;
    }
    
    // Fast forward RNG for remaining particles (to keep state clean if needed, optional)
    // Not strictly necessary unless we re-seed later.

    ofstream out;
    if(mode == VISUAL && rank == 0) {
        out.open("nbody_output.csv");
        out << "step,i,x,y,z,vx,vy,vz,m\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = chrono::high_resolution_clock::now();

    const __m512d vSoft = _mm512_set1_pd(SOFT);
    const __m512d v1p5  = _mm512_set1_pd(1.5);
    const __m512d vHalf = _mm512_set1_pd(0.5);
    const __m512d vDt   = _mm512_set1_pd(DT);

    // --- 4. Main Time Loop ---
    for(int step=0; step<STEPS; step++)
    {
        // 1. Prepare "Send" buffer with MY particles for the first ring step
        memcpy(send_x, x, my_N * sizeof(double));
        memcpy(send_y, y, my_N * sizeof(double));
        memcpy(send_z, z, my_N * sizeof(double));
        memcpy(send_m, m, my_N * sizeof(double));

        int current_send_count = my_N;

        // Force Accumulators
        // We use a temporary array or update directly?
        // To use AVX, we can't easily scatter-add to global vx/vy/vz.
        // We accumulate forces in registers inside the kernel, then write to temporary arrays.
        // Or simpler: We just update vx/vy/vz at the very end of the step?
        // The serial code updates v incrementally. We can do that here too.
        
        // We need to accumulate Accelerations (or Force/mass)
        // Since we update V at the end of the full ring cycle, we need temp buffers for Forces.
        // Actually, the serial code updates vx[i] += fx * dt/m immediately after processing a tile.
        // We can do exactly that.

        // --- Ring Algorithm ---
        int src = (rank - 1 + size) % size;
        int dst = (rank + 1) % size;

        for(int r=0; r<size; r++)
        {
            // Determine who owns the particles currently in 'send_x' (conceptually, for logic)
            // But we actually process the data we HOLD.
            
            // In the first step (r=0), we have OUR OWN particles in send_x.
            // We compute forces between (x,y,z) [My particles] and (send_x, send_y...) [Other particles].
            
            // NOTE: On step 0, we are comparing My Particles vs My Particles.
            // On step 1, we have received neighbor's particles.
            
            // Pointers to the "Other" particles we are interacting with
            double *ox, *oy, *oz, *om;
            int other_N;

            // Communication (Overlap attempt or simple blocking)
            // Blocking SendRecv is usually safest and highly optimized for Ring.
            if(r > 0) {
                // If not the first step, we need to receive new data.
                // We send what we just processed to the right, receive new from left.
                
                // Get the count of particles coming from the left
                int incoming_rank = (rank - r + size) % size;
                int incoming_count = all_counts[incoming_rank];

                MPI_Sendrecv(send_x, current_send_count, MPI_DOUBLE, dst, 0,
                             recv_x, incoming_count,     MPI_DOUBLE, src, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                             
                MPI_Sendrecv(send_y, current_send_count, MPI_DOUBLE, dst, 1,
                             recv_y, incoming_count,     MPI_DOUBLE, src, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Sendrecv(send_z, current_send_count, MPI_DOUBLE, dst, 2,
                             recv_z, incoming_count,     MPI_DOUBLE, src, 2,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Sendrecv(send_m, current_send_count, MPI_DOUBLE, dst, 3,
                             recv_m, incoming_count,     MPI_DOUBLE, src, 3,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Swap pointers: The received data becomes the data to send next
                std::swap(send_x, recv_x);
                std::swap(send_y, recv_y);
                std::swap(send_z, recv_z);
                std::swap(send_m, recv_m);
                
                other_N = incoming_count;
                current_send_count = incoming_count;
            } else {
                // Step 0: Interact with self
                ox = x; oy = y; oz = z; om = m;
                other_N = my_N;
                // Prepare send_x for next iteration is already done (it contains my data)
            }

            // Adjust pointers if r > 0 because of the swap
            if (r > 0) {
                ox = send_x; oy = send_y; oz = send_z; om = send_m;
            }

            // --- AVX-512 Computation Kernel ---
            // i loops over MY particles
            for(int i=0; i<my_N; i++)
            {
                __m512d v_xi  = _mm512_set1_pd(x[i]);
                __m512d v_yi  = _mm512_set1_pd(y[i]);
                __m512d v_zi  = _mm512_set1_pd(z[i]);
                __m512d v_Gmi = _mm512_set1_pd(G * m[i]);

                __m512d fx0=_mm512_setzero_pd(), fy0=_mm512_setzero_pd(), fz0=_mm512_setzero_pd();
                __m512d fx1=_mm512_setzero_pd(), fy1=_mm512_setzero_pd(), fz1=_mm512_setzero_pd();

                int j = 0;
                // j loops over OTHER particles (from buffer or local)
                for(; j <= other_N - 16; j+=16)
                {
                    _mm_prefetch((char*)&ox[j+64], _MM_HINT_T0);
                    _mm_prefetch((char*)&oy[j+64], _MM_HINT_T0);
                    _mm_prefetch((char*)&oz[j+64], _MM_HINT_T0);
                    _mm_prefetch((char*)&om[j+64], _MM_HINT_T0);

                    __m512d x0=_mm512_load_pd(&ox[j]);   __m512d x1=_mm512_load_pd(&ox[j+8]);
                    __m512d y0=_mm512_load_pd(&oy[j]);   __m512d y1=_mm512_load_pd(&oy[j+8]);
                    __m512d z0=_mm512_load_pd(&oz[j]);   __m512d z1=_mm512_load_pd(&oz[j+8]);
                    __m512d m0=_mm512_load_pd(&om[j]);   __m512d m1=_mm512_load_pd(&om[j+8]);

                    __m512d dx0=_mm512_sub_pd(x0,v_xi); __m512d dx1=_mm512_sub_pd(x1,v_xi);
                    __m512d dy0=_mm512_sub_pd(y0,v_yi); __m512d dy1=_mm512_sub_pd(y1,v_yi);
                    __m512d dz0=_mm512_sub_pd(z0,v_zi); __m512d dz1=_mm512_sub_pd(z1,v_zi);

                    __m512d r20 = _mm512_fmadd_pd(dx0,dx0,vSoft);
                    r20 = _mm512_fmadd_pd(dy0,dy0,r20);
                    r20 = _mm512_fmadd_pd(dz0,dz0,r20);

                    __m512d r21 = _mm512_fmadd_pd(dx1,dx1,vSoft);
                    r21 = _mm512_fmadd_pd(dy1,dy1,r21);
                    r21 = _mm512_fmadd_pd(dz1,dz1,r21);

                    __m512d inv0 = _mm512_rsqrt14_pd(r20);
                    __m512d inv1 = _mm512_rsqrt14_pd(r21);

                    __m512d t0 = _mm512_mul_pd(r20, _mm512_mul_pd(inv0, inv0));
                    t0 = _mm512_fnmadd_pd(vHalf, t0, v1p5); 
                    inv0 = _mm512_mul_pd(inv0, t0);

                    __m512d t1 = _mm512_mul_pd(r21, _mm512_mul_pd(inv1, inv1));
                    t1 = _mm512_fnmadd_pd(vHalf, t1, v1p5);
                    inv1 = _mm512_mul_pd(inv1, t1);

                    __m512d inv30 = _mm512_mul_pd(inv0,_mm512_mul_pd(inv0,inv0));
                    __m512d inv31 = _mm512_mul_pd(inv1,_mm512_mul_pd(inv1,inv1));

                    __m512d f0 = _mm512_mul_pd(_mm512_mul_pd(v_Gmi,m0),inv30);
                    __m512d f1 = _mm512_mul_pd(_mm512_mul_pd(v_Gmi,m1),inv31);

                    fx0 = _mm512_fmadd_pd(f0,dx0,fx0); fx1 = _mm512_fmadd_pd(f1,dx1,fx1);
                    fy0 = _mm512_fmadd_pd(f0,dy0,fy0); fy1 = _mm512_fmadd_pd(f1,dy1,fy1);
                    fz0 = _mm512_fmadd_pd(f0,dz0,fz0); fz1 = _mm512_fmadd_pd(f1,dz1,fz1);
                }

                double fx = _mm512_reduce_add_pd(_mm512_add_pd(fx0,fx1));
                double fy = _mm512_reduce_add_pd(_mm512_add_pd(fy0,fy1));
                double fz = _mm512_reduce_add_pd(_mm512_add_pd(fz0,fz1));

                // Cleanup remainder
                for(; j < other_N; j++){
                    // Don't calculate self-self force in step 0
                    if (r==0 && i==j) continue; 

                    double dx=ox[j]-x[i], dy=oy[j]-y[i], dz=oz[j]-z[i];
                    double r2=dx*dx+dy*dy+dz*dz+SOFT;
                    double inv=1.0/sqrt(r2);
                    double inv3=inv*inv*inv;
                    double f=G*m[i]*om[j]*inv3;
                    fx+=f*dx; fy+=f*dy; fz+=f*dz;
                }

                double invMdt = DT/m[i];
                vx[i] += fx * invMdt;
                vy[i] += fy * invMdt;
                vz[i] += fz * invMdt;
            } // End Particle Loop
        } // End Ring Loop

        // --- Position Update ---
        int i=0;
        for(; i<=my_N-8; i+=8) {
            __m512d v_x  = _mm512_load_pd(&x[i]);
            __m512d v_y  = _mm512_load_pd(&y[i]);
            __m512d v_z  = _mm512_load_pd(&z[i]);
            __m512d v_vx = _mm512_load_pd(&vx[i]);
            __m512d v_vy = _mm512_load_pd(&vy[i]);
            __m512d v_vz = _mm512_load_pd(&vz[i]);

            _mm512_store_pd(&x[i], _mm512_fmadd_pd(v_vx, vDt, v_x));
            _mm512_store_pd(&y[i], _mm512_fmadd_pd(v_vy, vDt, v_y));
            _mm512_store_pd(&z[i], _mm512_fmadd_pd(v_vz, vDt, v_z));
        }
        for(; i<my_N; i++) {
            x[i]+=vx[i]*DT;
            y[i]+=vy[i]*DT;
            z[i]+=vz[i]*DT;
        }

        // --- Visualization Output (Slow, but requested) ---
        if(mode==VISUAL && step%saveInterval==0){
            // Gather all data to Rank 0 to write to file
            // Note: For massive N, this will crash memory. Assuming reasonable N for visual.
            
            // Arrays to hold collected data at Rank 0
            vector<double> g_x, g_y, g_z, g_vx, g_vy, g_vz, g_m;
            if(rank == 0) {
                g_x.resize(N_GLOBAL); g_y.resize(N_GLOBAL); g_z.resize(N_GLOBAL);
                g_vx.resize(N_GLOBAL); g_vy.resize(N_GLOBAL); g_vz.resize(N_GLOBAL); g_m.resize(N_GLOBAL);
            }

            MPI_Gatherv(x, my_N, MPI_DOUBLE, g_x.data(), all_counts.data(), all_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(y, my_N, MPI_DOUBLE, g_y.data(), all_counts.data(), all_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(z, my_N, MPI_DOUBLE, g_z.data(), all_counts.data(), all_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(vx, my_N, MPI_DOUBLE, g_vx.data(), all_counts.data(), all_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(vy, my_N, MPI_DOUBLE, g_vy.data(), all_counts.data(), all_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(vz, my_N, MPI_DOUBLE, g_vz.data(), all_counts.data(), all_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(m, my_N, MPI_DOUBLE, g_m.data(), all_counts.data(), all_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if(rank == 0) {
                for(int k=0; k<N_GLOBAL; k++) {
                    out<<step<<","<<k<<","<<g_x[k]<<","<<g_y[k]<<","<<g_z[k]<<","
                       <<g_vx[k]<<","<<g_vy[k]<<","<<g_vz[k]<<","<<g_m[k]<<"\n";
                }
            }
        }
    }

    auto end=chrono::high_resolution_clock::now();
    double t=chrono::duration<double>(end-start).count();
    
    // Global Reduce max time (so we report the slowest rank's time)
    double max_t;
    MPI_Reduce(&t, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        double ops=(double)N_GLOBAL*N_GLOBAL*20*STEPS;
        cout<<"Time: "<<max_t<<" s\n";
        cout<<"GFLOPs: "<<(ops/max_t)/1e9<<"\n";
    }

    MPI_Finalize();
}
