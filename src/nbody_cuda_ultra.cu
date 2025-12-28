#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <cstdlib>

// --- Constants ---
constexpr double DT = 0.001;
constexpr double G  = 1.0;
constexpr double SOFT = 0.1;
constexpr int SEED = 1234;
constexpr int BLOCK = 256;

// --- CUDA Kernel ---
__global__ void nbody_kernel(
    double* __restrict__ x,
    double* __restrict__ y,
    double* __restrict__ z,
    double* __restrict__ vx,
    double* __restrict__ vy,
    double* __restrict__ vz,
    double* __restrict__ m,
    int N)
{
    __shared__ double sx[BLOCK], sy[BLOCK], sz[BLOCK], sm[BLOCK];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Registers for the current particle 'i'
    double xi, yi, zi, vxi, vyi, vzi;
    
    // Only load if 'i' is valid, but ALL threads participate in loading Shared Memory
    if(i < N) {
        xi = x[i]; yi = y[i]; zi = z[i];
        vxi = vx[i]; vyi = vy[i]; vzi = vz[i];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    // Tile over all particles in steps of BLOCK
    for(int tile = 0; tile < N; tile += BLOCK)
    {
        int j = tile + threadIdx.x;
        
        // --- 1. Load Shared Memory with Padding logic ---
        if(j < N){
            sx[threadIdx.x] = x[j];
            sy[threadIdx.x] = y[j];
            sz[threadIdx.x] = z[j];
            sm[threadIdx.x] = m[j];
        } else {
            // FIX: Zero out mass/pos for out-of-bound threads
            // This prevents "Ghost Particles" (NaNs/garbage data)
            sx[threadIdx.x] = 0.0;
            sy[threadIdx.x] = 0.0;
            sz[threadIdx.x] = 0.0;
            sm[threadIdx.x] = 0.0;
        }
        
        __syncthreads();

        // --- 2. Compute Force ---
        if(i < N) {
            #pragma unroll 8
            for(int k = 0; k < BLOCK; ++k)
            {
                double dx = sx[k] - xi;
                double dy = sy[k] - yi;
                double dz = sz[k] - zi;

                double dist2 = dx*dx + dy*dy + dz*dz + SOFT;
                double inv   = rsqrt(dist2);
                double inv3  = inv * inv * inv;

                double f = G * sm[k] * inv3;

                Fx += f * dx;
                Fy += f * dy;
                Fz += f * dz;
            }
        }
        __syncthreads();
    }

    // --- 3. Update Position & Velocity ---
    if(i < N) {
        vxi += Fx * DT;
        vyi += Fy * DT;
        vzi += Fz * DT;

        xi += vxi * DT;
        yi += vyi * DT;
        zi += vzi * DT;

        x[i] = xi;  y[i] = yi;  z[i] = zi;
        vx[i] = vxi; vy[i] = vyi; vz[i] = vzi;
    }
}

int main(int argc, char** argv)
{
    // Defaults
    int N = 10000;
    int STEPS = 1000;
    int SAVE_INTERVAL = 10; 
    bool benchmark = true;

    // Argument Parsing
    if(argc > 1) N = atoi(argv[1]);
    if(argc > 2) STEPS = atoi(argv[2]);
    if(argc > 3 && std::string(argv[3]) == "visual") benchmark = false;
    // FIX: Read the save interval from command line
    if(argc > 4) SAVE_INTERVAL = atoi(argv[4]);

    size_t bytes = N * sizeof(double);

    std::cout << "Running CUDA N-Body: N=" << N 
              << ", Steps=" << STEPS 
              << ", Mode=" << (benchmark ? "Benchmark" : "Visual") 
              << ", Interval=" << SAVE_INTERVAL << "\n";

    // Host Vectors
    std::vector<double> x(N), y(N), z(N), vx(N), vy(N), vz(N), m(N);

    // Initialization
    srand(SEED);
    for(int i=0;i<N;i++){
        x[i]  = rand()/(double)RAND_MAX * 2.0 - 1.0;
        y[i]  = rand()/(double)RAND_MAX * 2.0 - 1.0;
        z[i]  = rand()/(double)RAND_MAX * 2.0 - 1.0;
        vx[i] = rand()/(double)RAND_MAX - 0.5;
        vy[i] = rand()/(double)RAND_MAX - 0.5;
        vz[i] = rand()/(double)RAND_MAX - 0.5;
        m[i]  = rand()/(double)RAND_MAX * 0.9 + 0.1;
    }

    // Device Pointers
    double *dx,*dy,*dz,*dvx,*dvy,*dvz,*dm;
    cudaMalloc(&dx,bytes); cudaMalloc(&dy,bytes); cudaMalloc(&dz,bytes);
    cudaMalloc(&dvx,bytes); cudaMalloc(&dvy,bytes); cudaMalloc(&dvz,bytes);
    cudaMalloc(&dm,bytes);

    // Host -> Device Copy
    cudaMemcpy(dx,x.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dy,y.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dz,z.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dvx,vx.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dvy,vy.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dvz,vz.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dm,m.data(),bytes,cudaMemcpyHostToDevice);

    dim3 block(BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK);

    std::ofstream out;
    if(!benchmark){
        std::string filename = "nbody_cuda_output_N" + std::to_string(N) + ".csv";
        out.open(filename);
        out << "step,i,x,y,z\n";
    }

    // Timer Setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // --- Main Loop ---
    for(int s=0;s<STEPS;s++){
        nbody_kernel<<<grid,block>>>(dx,dy,dz,dvx,dvy,dvz,dm,N);

        // FIX: Added logic to copy Y and Z, and use dynamic SAVE_INTERVAL
        if(!benchmark && s % SAVE_INTERVAL == 0){
            cudaMemcpy(x.data(), dx, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(y.data(), dy, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(z.data(), dz, bytes, cudaMemcpyDeviceToHost);

            for(int i=0;i<N;i++)
                out << s << "," << i << "," << x[i] << "," << y[i] << "," << z[i] << "\n";
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Performance Metrics
    float ms; cudaEventElapsedTime(&ms,start,stop);
    double sec = ms / 1000.0;
    double ops = (double)N * N * 20.0 * STEPS;

    std::cout << "Time: "   << sec << " s\n";
    std::cout << "GFLOPs: " << (ops / sec) / 1e9 << "\n";

    if(!benchmark) out.close();

    // Cleanup
    cudaFree(dx); cudaFree(dy); cudaFree(dz);
    cudaFree(dvx); cudaFree(dvy); cudaFree(dvz); cudaFree(dm);
    
    return 0;
}
