#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <string>
#include <sstream>

const double DT = 0.001;
const double G = 1.0;
const double SOFTENING = 0.1;
const int SEED = 1234;

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

void initialize_bodies(std::vector<Body>& bodies) {
    std::srand(SEED);

    int N = bodies.size();
    for (int i = 0; i < N; i++) {
        bodies[i].x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        bodies[i].y = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        bodies[i].z = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

        bodies[i].vx = ((double)rand() / RAND_MAX) - 0.5;
        bodies[i].vy = ((double)rand() / RAND_MAX) - 0.5;
        bodies[i].vz = ((double)rand() / RAND_MAX) - 0.5;

        bodies[i].mass = ((double)rand() / RAND_MAX) * 0.9 + 0.1;
    }
}

void update_physics(std::vector<Body>& bodies) {
    int N = bodies.size();

    std::vector<double> fx(N, 0.0), fy(N, 0.0), fz(N, 0.0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) continue;

            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dz = bodies[j].z - bodies[i].z;

            double distSq = dx*dx + dy*dy + dz*dz + SOFTENING;
            double dist = std::sqrt(distSq);

            double f = (G * bodies[i].mass * bodies[j].mass) / (dist * dist * dist);

            fx[i] += f * dx;
            fy[i] += f * dy;
            fz[i] += f * dz;
        }
    }

    for (int i = 0; i < N; i++) {
        double ax = fx[i] / bodies[i].mass;
        double ay = fy[i] / bodies[i].mass;
        double az = fz[i] / bodies[i].mass;

        bodies[i].vx += ax * DT;
        bodies[i].vy += ay * DT;
        bodies[i].vz += az * DT;

        bodies[i].x += bodies[i].vx * DT;
        bodies[i].y += bodies[i].vy * DT;
        bodies[i].z += bodies[i].vz * DT;
    }
}

int main(int argc, char* argv[]) {
    // Default parameters
    int n_bodies = 100;
    int n_steps = 1000;
    bool benchmark_mode = true; 
    int save_interval = 1;
    std::string mode_input = "bench";

    // 1. Parse Arguments
    if (argc >= 2) n_bodies = std::atoi(argv[1]);
    if (argc >= 3) n_steps = std::atoi(argv[2]);
    
    if (argc >= 4) {
        mode_input = std::string(argv[3]);
        if (mode_input == "visual") {
            benchmark_mode = false;
        }
    }
    
    if (!benchmark_mode && argc >= 5) {
        save_interval = std::atoi(argv[4]);
    }

    std::string output_filename = "nbody_output_serial_N" + std::to_string(n_bodies) + ".csv";

    if (!benchmark_mode) {
        std::cout << "\n--- N-Body Serial Simulation ---" << std::endl;
        std::cout << "Bodies: " << n_bodies << " | Steps: " << n_steps << std::endl;
        std::cout << "Mode:   VISUALIZATION" << std::endl;
        std::cout << "Output: " << output_filename << std::endl;
    }

    // 2. Initialize
    std::vector<Body> bodies(n_bodies);
    initialize_bodies(bodies);

    // 3. Open Trajectory File (Only in Visual Mode)
    std::ofstream outfile;
    if (!benchmark_mode) {
        outfile.open(output_filename);
        outfile << "step,i,x,y,z,vx,vy,vz,m" << std::endl;
    }

    // 4. Main Simulation Loop
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < n_steps; step++) {

        // Save state if in Visual mode and interval matches
        if (!benchmark_mode && (step % save_interval == 0)) {
            for (int i = 0; i < n_bodies; i++) {
                outfile << step << "," << i << ","
                        << bodies[i].x << "," << bodies[i].y << "," << bodies[i].z << ","
                        << bodies[i].vx << "," << bodies[i].vy << "," << bodies[i].vz << ","
                        << bodies[i].mass << "\n";
            }
        }

        update_physics(bodies);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double time_seconds = elapsed.count();

    if (!benchmark_mode) {
        outfile.close();
    }

    // 5. Performance Metrics
    double total_ops = (double)n_bodies * n_bodies * 20.0 * n_steps;
    double gflops = (total_ops / time_seconds) / 1e9;

    std::cout << "Time:   " << time_seconds << " s" << std::endl;
    std::cout << "GFLOPs: " << gflops << "\n";

    return 0;
}
