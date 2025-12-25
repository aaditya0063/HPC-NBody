#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

using namespace std;

constexpr double DT = 0.001;
constexpr double G  = 1.0;
constexpr double SOFT = 0.1;
constexpr int SEED = 1234;
// TILE_SIZE: 1024 particles * 32 bytes = 32KB (Fits exactly in L1 Cache)
constexpr int TILE_SIZE = 1024; 

enum Mode { BENCH, VISUAL };

int main(int argc, char** argv)
{
    if(argc < 4) {
        cerr << "Usage: " << argv[0] << " <N> <STEPS> <bench|visual> [SAVE_INTERVAL]\n";
        return 1;
    }

    const int N = atoi(argv[1]);
    const int STEPS = atoi(argv[2]);
    const double dt = DT;

    Mode mode = (strcmp(argv[3], "visual") == 0) ? VISUAL : BENCH;
    int saveInterval = (mode == VISUAL && argc > 4) ? atoi(argv[4]) : 1;

    size_t bytes = N * sizeof(double);
    if (bytes % 64 != 0) bytes += 64 - (bytes % 64);

    double* __restrict__ x  = (double*) aligned_alloc(64, bytes);
    double* __restrict__ y  = (double*) aligned_alloc(64, bytes);
    double* __restrict__ z  = (double*) aligned_alloc(64, bytes);
    double* __restrict__ vx = (double*) aligned_alloc(64, bytes);
    double* __restrict__ vy = (double*) aligned_alloc(64, bytes);
    double* __restrict__ vz = (double*) aligned_alloc(64, bytes);
    double* __restrict__ m  = (double*) aligned_alloc(64, bytes);

    srand(SEED);
    for(int i=0;i<N;i++){
        x[i]=rand()/(double)RAND_MAX*2.0-1.0;
        y[i]=rand()/(double)RAND_MAX*2.0-1.0;
        z[i]=rand()/(double)RAND_MAX*2.0-1.0;
        vx[i]=rand()/(double)RAND_MAX-0.5;
        vy[i]=rand()/(double)RAND_MAX-0.5;
        vz[i]=rand()/(double)RAND_MAX-0.5;
        m[i]=rand()/(double)RAND_MAX*0.9+0.1;
    }

    ofstream out;
    if(mode == VISUAL) {
        out.open("nbody_output.csv");
        out << "step,i,x,y,z,vx,vy,vz,m\n";
    }

    auto start = chrono::high_resolution_clock::now();

    const __m512d vSoft = _mm512_set1_pd(SOFT);
    const __m512d v1p5  = _mm512_set1_pd(1.5);
    const __m512d vHalf = _mm512_set1_pd(0.5);
    const __m512d vDt   = _mm512_set1_pd(dt);

    for(int step=0; step<STEPS; step++)
    {
        // --- TILING IMPLEMENTATION ---
        // We iterate through chunks (tiles) of J to keep them in Cache
        for(int jj=0; jj < N; jj += TILE_SIZE)
        {
            // Calculate the end of the current tile
            int j_end = std::min(jj + TILE_SIZE, N);

            // Process all i particles against this J-Tile
            for(int i=0; i<N; i++)
            {
                __m512d v_xi  = _mm512_set1_pd(x[i]);
                __m512d v_yi  = _mm512_set1_pd(y[i]);
                __m512d v_zi  = _mm512_set1_pd(z[i]);
                __m512d v_Gmi = _mm512_set1_pd(G * m[i]);

                __m512d fx0=_mm512_setzero_pd(), fy0=_mm512_setzero_pd(), fz0=_mm512_setzero_pd();
                __m512d fx1=_mm512_setzero_pd(), fy1=_mm512_setzero_pd(), fz1=_mm512_setzero_pd();

                int j = jj;
                // Unrolled Loop (2x) inside the Tile
                // Ensure we don't go past the end of the tile (j_end)
                for(; j <= j_end - 16; j+=16)
                {
                    // Prefetch within the tile
                    _mm_prefetch((char*)&x[j+64], _MM_HINT_T0);
                    _mm_prefetch((char*)&y[j+64], _MM_HINT_T0);
                    _mm_prefetch((char*)&z[j+64], _MM_HINT_T0);
                    _mm_prefetch((char*)&m[j+64], _MM_HINT_T0);

                    __m512d x0=_mm512_load_pd(&x[j]);   __m512d x1=_mm512_load_pd(&x[j+8]);
                    __m512d y0=_mm512_load_pd(&y[j]);   __m512d y1=_mm512_load_pd(&y[j+8]);
                    __m512d z0=_mm512_load_pd(&z[j]);   __m512d z1=_mm512_load_pd(&z[j+8]);
                    __m512d m0=_mm512_load_pd(&m[j]);   __m512d m1=_mm512_load_pd(&m[j+8]);

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

                    // Newton-Raphson
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

                // Cleanup loop for this tile
                for(; j < j_end; j++){
                    double dx=x[j]-x[i], dy=y[j]-y[i], dz=z[j]-z[i];
                    double r2=dx*dx+dy*dy+dz*dz+SOFT;
                    double inv=1.0/sqrt(r2);
                    double inv3=inv*inv*inv;
                    double f=G*m[i]*m[j]*inv3;
                    fx+=f*dx; fy+=f*dy; fz+=f*dz;
                }

                // --- INCREMENTAL VELOCITY UPDATE ---
                // We update velocity 'partially' for every tile.
                // v_new = v_old + (F_tile * dt / m)
                double invMdt = dt/m[i];
                vx[i] += fx * invMdt;
                vy[i] += fy * invMdt;
                vz[i] += fz * invMdt;
            }
        }

        // Vectorized Position Update
        int i=0;
        for(; i<=N-8; i+=8) {
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
        for(; i<N; i++) {
            x[i]+=vx[i]*dt;
            y[i]+=vy[i]*dt;
            z[i]+=vz[i]*dt;
        }

        if(mode==VISUAL && step%saveInterval==0){
            for(int k=0;k<N;k++)
                out<<step<<","<<k<<","<<x[k]<<","<<y[k]<<","<<z[k]<<","<<vx[k]<<","<<vy[k]<<","<<vz[k]<<","<<m[k]<<"\n";
        }
    }

    auto end=chrono::high_resolution_clock::now();
    double t=chrono::duration<double>(end-start).count();
    double ops=(double)N*N*20*STEPS;
    cout<<"Time: "<<t<<" s\n";
    cout<<"GFLOPs: "<<(ops/t)/1e9<<"\n";
}
