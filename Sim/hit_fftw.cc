#include "hit_fftw.hh"
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#ifdef HIT_USE_FFTW
#include <fftw3.h>
#endif

#ifdef HIT_FFTW_DEBUG
#define HIT_DBG(...) do { std::fprintf(stderr, __VA_ARGS__); } while(0)
#else
#define HIT_DBG(...) do {} while(0)
#endif

hit_fftw_backend::hit_fftw_backend() : enabled(false), nx(0), ny(0), nz(0), nzc(0), kf2(0), lx(1), ly(1), lz(1),
    target_energy(0), init_spectrum_type(0), energy_total(0), energy_low(0), energy_high(0), target_error(0), div_l2(0),
    debug_enabled(false), update_calls(0), fft_forward_calls(0), fft_inverse_force_calls(0),
    lowk_rescaled_modes(0), highk_rescaled_modes(0), alpha_last(1.0), forcing_l2_last(0.0), div_phys_l2_last(0.0) {
    var_comp[0]=var_comp[1]=var_comp[2]=0;
#ifdef HIT_USE_FFTW
    for(int i=0;i<3;i++){ plan_r2c[i]=plan_c2r_init[i]=plan_c2r_force[i]=NULL; uhat[i]=uhat_new[i]=duhat[i]=NULL; }
#endif
#ifdef HIT_FFTW_DEBUG
    debug_enabled = true;
#endif
}

hit_fftw_backend::~hit_fftw_backend(){
#ifdef HIT_USE_FFTW
    for(int i=0;i<3;i++){
        if(plan_r2c[i]) fftw_destroy_plan((fftw_plan)plan_r2c[i]);
        if(plan_c2r_init[i]) fftw_destroy_plan((fftw_plan)plan_c2r_init[i]);
        if(plan_c2r_force[i]) fftw_destroy_plan((fftw_plan)plan_c2r_force[i]);
        if(uhat[i]) fftw_free(uhat[i]);
        if(uhat_new[i]) fftw_free(uhat_new[i]);
        if(duhat[i]) fftw_free(duhat[i]);
    }
#endif
}

bool hit_fftw_backend::setup(int nx_, int ny_, int nz_, double lx_, double ly_, double lz_, int kf2_, double target_energy_, int init_spectrum_type_){
#ifndef HIT_USE_FFTW
    (void)nx_;(void)ny_;(void)nz_;(void)lx_;(void)ly_;(void)lz_;(void)kf2_;(void)target_energy_;(void)init_spectrum_type_;
    enabled=false;
    return false;
#else
    nx=nx_; ny=ny_; nz=nz_; nzc=nz/2+1;
    lx=lx_; ly=ly_; lz=lz_;
    kf2=kf2_;
    target_energy=target_energy_;
    init_spectrum_type=init_spectrum_type_;
    if(nx<=0||ny<=0||nz<=0||target_energy<=0){ enabled=false; return false; }

    size_t nr = (size_t)nx*ny*nz;
    size_t nc = (size_t)nx*ny*nzc;
    for(int c=0;c<3;c++){
        u0[c].assign(nr,0.0);
        ucur[c].assign(nr,0.0);
        force[c].assign(nr,0.0);
        uhat[c] = fftw_malloc(sizeof(fftw_complex)*nc);
        uhat_new[c] = fftw_malloc(sizeof(fftw_complex)*nc);
        duhat[c] = fftw_malloc(sizeof(fftw_complex)*nc);
        plan_r2c[c] = fftw_plan_dft_r2c_3d(nx,ny,nz,u0[c].data(),(fftw_complex*)uhat[c],FFTW_MEASURE);
        plan_c2r_init[c] = fftw_plan_dft_c2r_3d(nx,ny,nz,(fftw_complex*)uhat[c],u0[c].data(),FFTW_MEASURE);
        plan_c2r_force[c] = fftw_plan_dft_c2r_3d(nx,ny,nz,(fftw_complex*)duhat[c],force[c].data(),FFTW_MEASURE);
    }

    lowk_mask.assign(nc,0);
    int max_kb = (int)(sqrt((nx/2.0)*(nx/2.0) + (ny/2.0)*(ny/2.0) + (nz/2.0)*(nz/2.0))) + 2;
    shell_E.assign(max_kb,0.0);
    long lowk_count = 0;
    for(int i=0;i<nx;i++){
        int kx=(i<=nx/2)?i:i-nx;
        for(int j=0;j<ny;j++){
            int ky=(j<=ny/2)?j:j-ny;
            for(int k=0;k<nzc;k++){
                int kz=k;
                int kk = kx*kx+ky*ky+kz*kz;
                size_t id=((size_t)i*ny + j)*nzc + k;
                if(kk<=kf2*kf2 && kk>0) {
                    lowk_mask[id]=1;
                    lowk_count++;
                }
            }
        }
    }
    enabled=true;
    HIT_DBG("[HIT_FFTW] setup nx=%d ny=%d nz=%d kf2=%d lowk_modes=%ld target_energy=%g\n", nx, ny, nz, kf2, lowk_count, target_energy);
    return true;
#endif
}

void hit_fftw_backend::initialize(unsigned int seed){
#ifdef HIT_USE_FFTW
    if(!enabled) return;
    srand(seed);
    size_t nc=(size_t)nx*ny*nzc;
    for(int c=0;c<3;c++){
        fftw_complex *uh=(fftw_complex*)uhat[c];
        for(size_t q=0;q<nc;q++){ uh[q][0]=0; uh[q][1]=0; }
    }
    // random divergence-free low-k init
    for(int i=0;i<nx;i++){
        int kx=(i<=nx/2)?i:i-nx;
        double kxw=2*M_PI*kx/lx;
        for(int j=0;j<ny;j++){
            int ky=(j<=ny/2)?j:j-ny;
            double kyw=2*M_PI*ky/ly;
            for(int k=0;k<nzc;k++){
                int kz=k;
                if(kx==0&&ky==0&&kz==0) continue;
                int kk=kx*kx+ky*ky+kz*kz;
                if(kk>kf2*kf2) continue;
                double kzw=2*M_PI*kz/lz;
                double kmag=sqrt(kxw*kxw+kyw*kyw+kzw*kzw);
                double kh[3]={kxw/kmag,kyw/kmag,kzw/kmag};
                double a[3]={0,0,1}; if(fabs(kh[2])>0.9){a[1]=1;a[2]=0;}
                double e1[3]={a[1]*kh[2]-a[2]*kh[1],a[2]*kh[0]-a[0]*kh[2],a[0]*kh[1]-a[1]*kh[0]};
                double n1=sqrt(e1[0]*e1[0]+e1[1]*e1[1]+e1[2]*e1[2]); e1[0]/=n1;e1[1]/=n1;e1[2]/=n1;
                double e2[3]={kh[1]*e1[2]-kh[2]*e1[1],kh[2]*e1[0]-kh[0]*e1[2],kh[0]*e1[1]-kh[1]*e1[0]};
                double kmag_i = sqrt((double)kk);
                double w = 1.0;
                if(init_spectrum_type==1) {
                    w = pow((kmag_i>1e-12)?kmag_i:1.0, -5.0/3.0);
                } else if(init_spectrum_type==2) {
                    // Rogallo-style low-k peak: E(k)~k^4 exp(-2 (k/kp)^2)
                    double kp = (kf2>1)?(0.5*kf2):1.0;
                    double r = kmag_i/kp;
                    w = pow(r,4.0)*exp(-2.0*r*r);
                }
                double amp=sqrt((w>0)?w:0.0);
                double p1=2*M_PI*(rand()/(double)RAND_MAX), p2=2*M_PI*(rand()/(double)RAND_MAX);
                double a1=amp*cos(p1), b1=amp*sin(p1), a2=amp*cos(p2), b2=amp*sin(p2);
                size_t id=((size_t)i*ny+j)*nzc+k;
                for(int c=0;c<3;c++){
                    fftw_complex *uh=(fftw_complex*)uhat[c];
                    double re=a1*e1[c]+a2*e2[c], im=b1*e1[c]+b2*e2[c];
                    uh[id][0]=re; uh[id][1]=im;
                }
            }
        }
    }
    // Explicit divergence-free projection and zero-mode removal in spectral space.
    for(int i=0;i<nx;i++){
        int kx=(i<=nx/2)?i:i-nx;
        double kxw=2*M_PI*kx/lx;
        for(int j=0;j<ny;j++){
            int ky=(j<=ny/2)?j:j-ny;
            double kyw=2*M_PI*ky/ly;
            for(int k=0;k<nzc;k++){
                int kz=k;
                double kzw=2*M_PI*kz/lz;
                size_t id=((size_t)i*ny+j)*nzc+k;
                fftw_complex *u0h=(fftw_complex*)uhat[0], *u1h=(fftw_complex*)uhat[1], *u2h=(fftw_complex*)uhat[2];
                double kk = kxw*kxw+kyw*kyw+kzw*kzw;
                if(kk<1e-20) {
                    u0h[id][0]=u0h[id][1]=0.0;
                    u1h[id][0]=u1h[id][1]=0.0;
                    u2h[id][0]=u2h[id][1]=0.0;
                    continue;
                }
                double d_re = kxw*u0h[id][0] + kyw*u1h[id][0] + kzw*u2h[id][0];
                double d_im = kxw*u0h[id][1] + kyw*u1h[id][1] + kzw*u2h[id][1];
                u0h[id][0] -= kxw*d_re/kk; u0h[id][1] -= kxw*d_im/kk;
                u1h[id][0] -= kyw*d_re/kk; u1h[id][1] -= kyw*d_im/kk;
                u2h[id][0] -= kzw*d_re/kk; u2h[id][1] -= kzw*d_im/kk;
            }
        }
    }

    // scale to target energy
    double el=0;
    for(int c=0;c<3;c++){
        fftw_complex *uh=(fftw_complex*)uhat[c];
        for(size_t id=0;id<nc;id++) if(lowk_mask[id]) el += uh[id][0]*uh[id][0]+uh[id][1]*uh[id][1];
    }
    if(el>0){
        double fac=sqrt(target_energy/el);
        for(int c=0;c<3;c++){ fftw_complex*uh=(fftw_complex*)uhat[c]; for(size_t id=0;id<nc;id++){uh[id][0]*=fac; uh[id][1]*=fac;}}
    }
    size_t nr=(size_t)nx*ny*nz;
    for(int c=0;c<3;c++){
        fftw_execute((fftw_plan)plan_c2r_init[c]);
        for(size_t id=0;id<nr;id++) u0[c][id] /= nr;
    }
    HIT_DBG("[HIT_FFTW] initialize seed=%u\n", seed);
#endif
}

void hit_fftw_backend::set_velocity_field(const std::vector<double> &ux,const std::vector<double> &uy,const std::vector<double> &uz){
    if(!enabled) return;
    ucur[0]=ux; ucur[1]=uy; ucur[2]=uz;
}

void hit_fftw_backend::update_lowk_forcing(double dt){
#ifdef HIT_USE_FFTW
    if(!enabled || dt<=0) return;
    update_calls++;
    size_t nr=(size_t)nx*ny*nz, nc=(size_t)nx*ny*nzc;
    // forward
    u0[0]=ucur[0]; u0[1]=ucur[1]; u0[2]=ucur[2];
    for(int c=0;c<3;c++) {
        fftw_execute((fftw_plan)plan_r2c[c]);
        fft_forward_calls++;
    }

    // Spectral projection to enforce incompressibility before diagnostics/forcing.
    for(int i=0;i<nx;i++){
      int kx=(i<=nx/2)?i:i-nx; double kxw=2*M_PI*kx/lx;
      for(int j=0;j<ny;j++){
        int ky=(j<=ny/2)?j:j-ny; double kyw=2*M_PI*ky/ly;
        for(int k=0;k<nzc;k++){
          int kz=k; double kzw=2*M_PI*kz/lz;
          size_t id=((size_t)i*ny+j)*nzc+k;
          fftw_complex *u0h=(fftw_complex*)uhat[0], *u1h=(fftw_complex*)uhat[1], *u2h=(fftw_complex*)uhat[2];
          double kk = kxw*kxw+kyw*kyw+kzw*kzw;
          if(kk<1e-20) {
            u0h[id][0]=u0h[id][1]=0.0;
            u1h[id][0]=u1h[id][1]=0.0;
            u2h[id][0]=u2h[id][1]=0.0;
            continue;
          }
          double d_re = kxw*u0h[id][0] + kyw*u1h[id][0] + kzw*u2h[id][0];
          double d_im = kxw*u0h[id][1] + kyw*u1h[id][1] + kzw*u2h[id][1];
          u0h[id][0] -= kxw*d_re/kk; u0h[id][1] -= kxw*d_im/kk;
          u1h[id][0] -= kyw*d_re/kk; u1h[id][1] -= kyw*d_im/kk;
          u2h[id][0] -= kzw*d_re/kk; u2h[id][1] -= kzw*d_im/kk;
        }
      }
    }

    // Physical-space variance diagnostics.
    var_comp[0]=var_comp[1]=var_comp[2]=0;
    for(size_t id=0;id<nr;id++){
        double u=ucur[0][id],v=ucur[1][id],w=ucur[2][id];
        var_comp[0]+=u*u; var_comp[1]+=v*v; var_comp[2]+=w*w;
    }
    var_comp[0]/=nr; var_comp[1]/=nr; var_comp[2]/=nr;

    // Spectral energies (with real-to-complex multiplicity correction on kz).
    energy_total=energy_low=0;
    std::fill(shell_E.begin(), shell_E.end(), 0.0);
    std::vector<int> cnt(shell_E.size(),0);
    for(int i=0;i<nx;i++){
      int kx=(i<=nx/2)?i:i-nx;
      for(int j=0;j<ny;j++){
        int ky=(j<=ny/2)?j:j-ny;
        for(int k=0;k<nzc;k++){
          int kz=k;
          int kb=(int)(sqrt((double)(kx*kx+ky*ky+kz*kz))+0.5);
          size_t id=((size_t)i*ny+j)*nzc+k;
          double em=0;
          for(int c=0;c<3;c++){
            fftw_complex *uh=(fftw_complex*)uhat[c];
            em += uh[id][0]*uh[id][0]+uh[id][1]*uh[id][1];
          }
          double wz = (k==0 || (nz%2==0 && k==nz/2)) ? 1.0 : 2.0;
          double ecell = 0.5*wz*em/(nr*nr);
          energy_total += ecell;
          if(lowk_mask[id]) energy_low += ecell;
          if(kb>=0 && kb<(int)shell_E.size()){ shell_E[kb]+=ecell; cnt[kb]++; }
        }
      }
    }
    for(size_t i=0;i<shell_E.size();i++) if(cnt[i]>0) shell_E[i]/=cnt[i];
    energy_high = energy_total - energy_low; if(energy_high<0) energy_high=0;

    // spectralDNS-style update() mapping:
    // energy_new=energy_total, energy_lower=energy_low, energy_upper=energy_high
    // alpha = sqrt((target_energy-energy_upper)/energy_lower) with numerical floor.
    double denom = (energy_low>1e-14)?energy_low:1e-14;
    double alpha2 = (target_energy - energy_high) / denom;
    if(alpha2<0) alpha2=0;
    double alpha=sqrt(alpha2);
    alpha_last = alpha;
    target_error = target_energy - energy_total;

    lowk_rescaled_modes = 0;
    highk_rescaled_modes = 0;

    // Only low-k modes are rescaled; high-k increment is exactly zero.
    // The physical-space forcing is dU/dt from inverse FFT of this spectral increment.
    // build duhat = (alpha-1)*lowk*uhat
    for(int c=0;c<3;c++){
      fftw_complex *uh=(fftw_complex*)uhat[c], *du=(fftw_complex*)duhat[c];
      for(size_t id=0;id<nc;id++){
        if(lowk_mask[id]) {
          du[id][0]=(alpha-1)*uh[id][0];
          du[id][1]=(alpha-1)*uh[id][1];
          lowk_rescaled_modes++;
        }
        else {
          du[id][0]=0;
          du[id][1]=0;
          highk_rescaled_modes++;
        }
      }
      fftw_execute((fftw_plan)plan_c2r_force[c]);
      fft_inverse_force_calls++;
      for(size_t id=0;id<nr;id++) force[c][id] = force[c][id] / (nr*dt);
    }

    // divergence norm (spectral)
    double div2=0;
    for(int i=0;i<nx;i++){
      int kx=(i<=nx/2)?i:i-nx; double kxw=2*M_PI*kx/lx;
      for(int j=0;j<ny;j++){
        int ky=(j<=ny/2)?j:j-ny; double kyw=2*M_PI*ky/ly;
        for(int k=0;k<nzc;k++){
          int kz=k; double kzw=2*M_PI*kz/lz;
          size_t id=((size_t)i*ny+j)*nzc+k;
          fftw_complex *u0h=(fftw_complex*)uhat[0], *u1h=(fftw_complex*)uhat[1], *u2h=(fftw_complex*)uhat[2];
          double re = -(kxw*u0h[id][1] + kyw*u1h[id][1] + kzw*u2h[id][1]);
          double im =  (kxw*u0h[id][0] + kyw*u1h[id][0] + kzw*u2h[id][0]);
          div2 += (re*re+im*im)/(nr*nr);
        }
      }
    }
    div_l2 = sqrt(div2);

    // Physical-space divergence norm from current velocity field.
    double divp2 = 0;
    for(int i=0;i<nx;i++){
      int ip = (i+1)%nx;
      int im = (i+nx-1)%nx;
      for(int j=0;j<ny;j++){
        int jp=(j+1)%ny;
        int jm=(j+ny-1)%ny;
        for(int k=0;k<nz;k++){
          int kp=(k+1)%nz;
          int km=(k+nz-1)%nz;
          size_t id=((size_t)i*ny+j)*nz+k;
          size_t id_xp=((size_t)ip*ny+j)*nz+k, id_xm=((size_t)im*ny+j)*nz+k;
          size_t id_yp=((size_t)i*ny+jp)*nz+k, id_ym=((size_t)i*ny+jm)*nz+k;
          size_t id_zp=((size_t)i*ny+j)*nz+kp, id_zm=((size_t)i*ny+j)*nz+km;
          double dudx = (ucur[0][id_xp]-ucur[0][id_xm])*(0.5*nx/lx);
          double dvdy = (ucur[1][id_yp]-ucur[1][id_ym])*(0.5*ny/ly);
          double dwdz = (ucur[2][id_zp]-ucur[2][id_zm])*(0.5*nz/lz);
          double divv = dudx + dvdy + dwdz;
          divp2 += divv*divv;
        }
      }
    }
    div_phys_l2_last = sqrt(divp2/nr);

    forcing_l2_last = 0;
    for(size_t id=0;id<nr;id++) {
        forcing_l2_last += force[0][id]*force[0][id] + force[1][id]*force[1][id] + force[2][id]*force[2][id];
    }
    forcing_l2_last = sqrt(forcing_l2_last/nr);

    HIT_DBG("[HIT_FFTW] update=%ld E=%g Elow=%g Ehigh=%g alpha=%g lowk_scaled=%ld highk_zeroed=%ld |f|_l2=%g div_phys=%g fwd=%ld inv=%ld\n",
        update_calls, energy_total, energy_low, energy_high, alpha_last,
        lowk_rescaled_modes, highk_rescaled_modes, forcing_l2_last, div_phys_l2_last,
        fft_forward_calls, fft_inverse_force_calls);
#endif
}

static inline int wrapi(int i, int n){ i%=n; if(i<0) i+=n; return i; }

void hit_fftw_backend::sample_initial_velocity(double xh,double yh,double zh,double &u,double &v,double &w) const {
    if(!enabled) return;
    int i=wrapi((int)floor(xh*nx),nx), j=wrapi((int)floor(yh*ny),ny), k=wrapi((int)floor(zh*nz),nz);
    size_t id=((size_t)i*ny+j)*nz+k;
    u += u0[0][id]; v += u0[1][id]; w += u0[2][id];
}

void hit_fftw_backend::sample_forcing(double xh,double yh,double zh,double &fx,double &fy,double &fz) const {
    if(!enabled) {fx=fy=fz=0; return;}
    int i=wrapi((int)floor(xh*nx),nx), j=wrapi((int)floor(yh*ny),ny), k=wrapi((int)floor(zh*nz),nz);
    size_t id=((size_t)i*ny+j)*nz+k;
    fx = force[0][id]; fy = force[1][id]; fz = force[2][id];
}

int hit_fftw_backend::run_selftest(){
#ifndef HIT_USE_FFTW
    std::fprintf(stderr, "[HIT_FFTW_SELFTEST] HIT_USE_FFTW is not enabled.\n");
    return 2;
#else
    const int nx_t = 8, ny_t = 8, nz_t = 8;
    const double lx_t = 1.0, ly_t = 1.0, lz_t = 1.0;
    const int kf2_t = 2;
    const double target_e_t = 0.2;

    hit_fftw_backend b;
    if(!b.setup(nx_t, ny_t, nz_t, lx_t, ly_t, lz_t, kf2_t, target_e_t, 0)) {
        std::fprintf(stderr, "[HIT_FFTW_SELFTEST] setup failed.\n");
        return 3;
    }
    b.initialize(123);

    size_t n = (size_t)nx_t*ny_t*nz_t;
    std::vector<double> ux(n,0), uy(n,0), uz(n,0);
    size_t id = 0;
    for(int k=0;k<nz_t;k++) for(int j=0;j<ny_t;j++) for(int i=0;i<nx_t;i++,id++) {
        double x=(i+0.5)/nx_t, y=(j+0.5)/ny_t, z=(k+0.5)/nz_t;
        ux[id] = sin(2*M_PI*x) * cos(2*M_PI*y);
        uy[id] = -cos(2*M_PI*x) * sin(2*M_PI*y);
        uz[id] = 0.1*sin(2*M_PI*z);
    }

    b.set_velocity_field(ux,uy,uz);
    b.update_lowk_forcing(1e-2);

    double fx=0, fy=0, fz=0;
    b.sample_forcing(0.5,0.5,0.5,fx,fy,fz);

    std::fprintf(stdout,
        "[HIT_FFTW_SELFTEST] update_calls=%ld fft_fwd=%ld fft_inv_force=%ld E=%g Elow=%g Ehigh=%g alpha=%g |f|_l2=%g f(0.5)=(%g,%g,%g)\n",
        b.update_calls, b.fft_forward_calls, b.fft_inverse_force_calls,
        b.energy_total, b.energy_low, b.energy_high, b.alpha_last, b.forcing_l2_last,
        fx, fy, fz);

    if(!(b.fft_forward_calls>0 && b.fft_inverse_force_calls>0)) return 4;
    if(!(b.energy_total>0 && b.energy_low>=0 && b.energy_high>=0)) return 5;
    if(!(b.lowk_rescaled_modes>0)) return 6;
    return 0;
#endif
}