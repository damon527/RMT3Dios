#include "hit_fftw.hh"
#include <cstdio>

int main() {
    int rc = hit_fftw_backend::run_selftest();
    if(rc==0) {
        std::fprintf(stdout, "[HIT_FFTW_SELFTEST] PASS\n");
    } else {
        std::fprintf(stdout, "[HIT_FFTW_SELFTEST] FAIL rc=%d\n", rc);
    }
    return rc;
}