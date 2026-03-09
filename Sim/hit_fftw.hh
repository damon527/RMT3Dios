#ifndef HIT_FFTW_HH
#define HIT_FFTW_HH

#include <vector>
#include <cmath>

class hit_fftw_backend {
public:
    bool enabled;
    int nx, ny, nz, nzc;
    int kf2;
    double lx, ly, lz;
    double target_energy;
    int init_spectrum_type;

    double energy_total, energy_low, energy_high, target_error;
    double var_comp[3];
    double div_l2;
    std::vector<double> shell_E;

    // Debug/verification state for Stage-2 closure checks.
    bool debug_enabled;
    long update_calls;
    long fft_forward_calls;
    long fft_inverse_force_calls;
    long lowk_rescaled_modes;
    long highk_rescaled_modes;
    double alpha_last;
    double forcing_l2_last;
    double div_phys_l2_last;

    hit_fftw_backend();
    ~hit_fftw_backend();

    bool setup(int nx_, int ny_, int nz_, double lx_, double ly_, double lz_, int kf2_, double target_energy_, int init_spectrum_type_);
    void initialize(unsigned int seed);

    void set_velocity_field(const std::vector<double> &ux, const std::vector<double> &uy, const std::vector<double> &uz);
    void update_lowk_forcing(double dt);

    void sample_initial_velocity(double xh, double yh, double zh, double &u, double &v, double &w) const;
    void sample_forcing(double xh, double yh, double zh, double &fx, double &fy, double &fz) const;

    // Minimal standalone self-check used when full solver build is unavailable.
    static int run_selftest();

private:
    std::vector<double> u0[3], force[3], ucur[3];
#ifdef HIT_USE_FFTW
    void *plan_r2c[3];
    void *plan_c2r_init[3];
    void *plan_c2r_force[3];
    void *uhat[3];
    void *uhat_new[3];
    void *duhat[3];
    std::vector<char> lowk_mask;
#endif
};

#endif