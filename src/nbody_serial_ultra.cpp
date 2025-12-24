#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <immintrin.h>

using namespace std;

constexpr double DT = 0.001;
constexpr double G  = 1.0;
constexpr double SOFT = 0.1;
constexpr int SEED = 1234;

int main(int argc, char** argv)
{
    const int N = atoi(argv[1]);
    const int STEPS = atoi(argv[2]);
    const double dt = DT;

    double* __restrict__ x  = (double*) aligned_alloc(64, N*sizeof(double));
    double* __restrict__ y  = (double*) aligned_alloc(64, N*sizeof(double));
    double* __restrict__ z  = (double*) aligned_alloc(64, N*sizeof(double));
    double* __restrict__ vx = (double*) aligned_alloc(64, N*sizeof(double));
    double* __restrict__ vy = (double*) aligned_alloc(64, N*sizeof(double));
    double* __restrict__ vz = (double*) aligned_alloc(64, N*sizeof(double));
    double* __restrict__ m  = (double*) aligned_alloc(64, N*sizeof(double));

    __builtin_assume_aligned(x, 64);
    __builtin_assume_aligned(y, 64);
    __builtin_assume_aligned(z, 64);
    __builtin_assume_aligned(vx,64);
    __builtin_assume_aligned(vy,64);
    __builtin_assume_aligned(vz,64);
    __builtin_assume_aligned(m, 64);

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

    auto start = chrono::high_resolution_clock::now();

    for(int step=0; step<STEPS; ++step)
    {
        for(int i=0;i<N;i++)
        {
            const double xi=x[i], yi=y[i], zi=z[i];
            const double mi=m[i];
            const double Gmi = G * mi;

            double fx0=0, fy0=0, fz0=0;
            double fx1=0, fy1=0, fz1=0;
            double fx2=0, fy2=0, fz2=0;
            double fx3=0, fy3=0, fz3=0;

            int j=0;
            for(; j<=N-4; j+=4)
            {
                __builtin_prefetch(&x[j+32], 0, 1);
                __builtin_prefetch(&y[j+32], 0, 1);
                __builtin_prefetch(&z[j+32], 0, 1);
                __builtin_prefetch(&m[j+32], 0, 1);

                double dx0=x[j]-xi,   dy0=y[j]-yi,   dz0=z[j]-zi;
                double dx1=x[j+1]-xi, dy1=y[j+1]-yi, dz1=z[j+1]-zi;
                double dx2=x[j+2]-xi, dy2=y[j+2]-yi, dz2=z[j+2]-zi;
                double dx3=x[j+3]-xi, dy3=y[j+3]-yi, dz3=z[j+3]-zi;

                double r20 = dx0*dx0 + dy0*dy0 + dz0*dz0 + SOFT;
                double r21 = dx1*dx1 + dy1*dy1 + dz1*dz1 + SOFT;
                double r22 = dx2*dx2 + dy2*dy2 + dz2*dz2 + SOFT;
                double r23 = dx3*dx3 + dy3*dy3 + dz3*dz3 + SOFT;

                double inv0 = 1.0 / __builtin_sqrt(r20);
                double inv1 = 1.0 / __builtin_sqrt(r21);
                double inv2 = 1.0 / __builtin_sqrt(r22);
                double inv3 = 1.0 / __builtin_sqrt(r23);

                double inv30 = inv0*inv0*inv0;
                double inv31 = inv1*inv1*inv1;
                double inv32 = inv2*inv2*inv2;
                double inv33 = inv3*inv3*inv3;

                double f0 = Gmi * m[j]   * inv30;
                double f1 = Gmi * m[j+1] * inv31;
                double f2 = Gmi * m[j+2] * inv32;
                double f3 = Gmi * m[j+3] * inv33;

                fx0 += f0*dx0; fy0 += f0*dy0; fz0 += f0*dz0;
                fx1 += f1*dx1; fy1 += f1*dy1; fz1 += f1*dz1;
                fx2 += f2*dx2; fy2 += f2*dy2; fz2 += f2*dz2;
                fx3 += f3*dx3; fy3 += f3*dy3; fz3 += f3*dz3;
            }

            double fx = fx0+fx1+fx2+fx3;
            double fy = fy0+fy1+fy2+fy3;
            double fz = fz0+fz1+fz2+fz3;

            for(; j<N; j++){
                double dx=x[j]-xi, dy=y[j]-yi, dz=z[j]-zi;
                double r2 = dx*dx + dy*dy + dz*dz + SOFT;
                double inv = 1.0 / __builtin_sqrt(r2);
                double inv3 = inv*inv*inv;
                double f = Gmi * m[j] * inv3;
                fx += f*dx; fy += f*dy; fz += f*dz;
            }

            const double invMdt = dt / mi;
            vx[i] += fx * invMdt;
            vy[i] += fy * invMdt;
            vz[i] += fz * invMdt;

            x[i] += vx[i] * dt;
            y[i] += vy[i] * dt;
            z[i] += vz[i] * dt;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double t = chrono::duration<double>(end-start).count();

    double ops = double(N)*N*20.0*STEPS;
    cout << "Time: " << t << " s\n";
    cout << "GFLOPs: " << (ops/t)/1e9 << "\n";
}

