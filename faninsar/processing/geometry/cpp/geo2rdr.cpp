// C++/OpenMP replica of the FanInSAR geo2rdr Newton solver.
//
// Mirrors the pure-Torch backend: along-track azimuth-time seed plus an
// analytic Doppler derivative.  Loaded by Python through ctypes; see
// faninsar.processing.geometry.geo2rdr_backends.cpp_geo2rdr.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

struct HermiteCoeff {
    double c0[3];
    double c1[3];
    double c2[3];
    double c3[3];
};

inline void pack_coefficients(const double* t, const double* p0,
                              const double* p1, const double* v0,
                              const double* v1, double dt, HermiteCoeff& out)
{
    const double inv_dt = 1.0 / dt;
    for (int k = 0; k < 3; ++k) {
        const double slope = (p1[k] - p0[k]) * inv_dt;
        out.c0[k] = (v0[k] + v1[k] - 2.0 * slope) * inv_dt * inv_dt;
        out.c1[k] = (3.0 * slope - 2.0 * v0[k] - v1[k]) * inv_dt;
        out.c2[k] = v0[k];
        out.c3[k] = p0[k];
    }
}

inline int find_interval(const double* times, int n, double t)
{
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (times[mid] <= t) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return std::max(0, std::min(lo, n - 2));
}

inline void evaluate_orbit(const double* times, const HermiteCoeff* coeff,
                           int nstate, double t, double* pos, double* vel,
                           double* acc)
{
    const int interval = find_interval(times, nstate, t);
    const double dt = t - times[interval];
    const HermiteCoeff& c = coeff[interval];
    for (int k = 0; k < 3; ++k) {
        pos[k] = ((c.c0[k] * dt + c.c1[k]) * dt + c.c2[k]) * dt + c.c3[k];
        vel[k] = (3.0 * c.c0[k] * dt + 2.0 * c.c1[k]) * dt + c.c2[k];
        if (acc != nullptr) {
            acc[k] = 6.0 * c.c0[k] * dt + 2.0 * c.c1[k];
        }
    }
}

inline void llh_to_ecef(double lat_deg, double lon_deg, double hae, double* xyz)
{
    constexpr double radius = 6378137.0;
    constexpr double flattening = 1.0 / 298.257223563;
    constexpr double e2 = flattening * (2.0 - flattening);
    const double lat = lat_deg * (M_PI / 180.0);
    const double lon = lon_deg * (M_PI / 180.0);
    const double sin_lat = std::sin(lat);
    const double cos_lat = std::cos(lat);
    const double prime_vertical =
        radius / std::sqrt(1.0 - e2 * sin_lat * sin_lat);
    xyz[0] = (prime_vertical + hae) * cos_lat * std::cos(lon);
    xyz[1] = (prime_vertical + hae) * cos_lat * std::sin(lon);
    xyz[2] = (prime_vertical * (1.0 - e2) + hae) * sin_lat;
}

}  // namespace

extern "C" int geo2rdr_batch(
    const double* lat, const double* lon, const double* hae, const int64_t n,
    const double* orbit_times, const double* orbit_pos, const double* orbit_vel,
    const int64_t nstate, const double sensing_start_s,
    const double azimuth_dt_s, const double starting_range_m,
    const double range_dt_m, const double wavelength_m,
    const double doppler_hz, const double time_tol_s, const int max_iter,
    const int coarse_points, const int use_along_track_seed,
    const int use_analytic_derivative, double* az_index, double* range_index,
    int* converged)
{
    std::vector<HermiteCoeff> coeff(static_cast<size_t>(nstate - 1));
    for (int64_t i = 0; i + 1 < nstate; ++i) {
        pack_coefficients(
            &orbit_times[i], &orbit_pos[3 * i], &orbit_pos[3 * (i + 1)],
            &orbit_vel[3 * i], &orbit_vel[3 * (i + 1)],
            orbit_times[i + 1] - orbit_times[i],
            coeff[static_cast<size_t>(i)]);
    }
    const double tstart = orbit_times[0];
    const double tend = orbit_times[nstate - 1];

#pragma omp parallel for schedule(static)
    for (int64_t index = 0; index < n; ++index) {
        double xyz[3];
        llh_to_ecef(lat[index], lon[index], hae[index], xyz);
        double t = sensing_start_s;

        if (use_along_track_seed) {
            double pos[3];
            double vel[3];
            evaluate_orbit(orbit_times, coeff.data(), nstate, sensing_start_s,
                           pos, vel, nullptr);
            const double speed2 =
                vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2];
            if (speed2 > 0.0) {
                const double proj =
                    ((xyz[0] - pos[0]) * vel[0] + (xyz[1] - pos[1]) * vel[1] +
                     (xyz[2] - pos[2]) * vel[2]) /
                    speed2;
                t = sensing_start_s + proj;
            }
            t = std::max(tstart, std::min(tend, t));
        } else {
            const double dt =
                (tend - tstart) / static_cast<double>(coarse_points - 1);
            double best_r = 1e16;
            double best_t = -1000.0;
            for (int k = 0; k < coarse_points; ++k) {
                const double tt = tstart + k * dt;
                if (tt < tstart || tt > tend) {
                    continue;
                }
                double pos[3];
                double vel[3];
                evaluate_orbit(orbit_times, coeff.data(), nstate, tt, pos, vel,
                               nullptr);
                const double dx = xyz[0] - pos[0];
                const double dy = xyz[1] - pos[1];
                const double dz = xyz[2] - pos[2];
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (r < best_r) {
                    best_r = r;
                    best_t = tt;
                }
            }
            t = (best_t < 0.0) ? 0.5 * (tstart + tend) : best_t;
        }

        double aztime = t;
        double slant_range = 0.0;
        double step = 0.0;
        int done = 0;
        for (int iter = 0; iter < max_iter && !done; ++iter) {
            aztime -= step;
            double pos[3];
            double vel[3];
            double acc[3];
            evaluate_orbit(orbit_times, coeff.data(), nstate, aztime, pos, vel,
                           acc);
            const double rx = xyz[0] - pos[0];
            const double ry = xyz[1] - pos[1];
            const double rz = xyz[2] - pos[2];
            slant_range = std::sqrt(rx * rx + ry * ry + rz * rz);
            const double dopfact = rx * vel[0] + ry * vel[1] + rz * vel[2];
            const double vel2 =
                vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2];
            const double fdop = 0.5 * wavelength_m * doppler_hz;
            const double fn = dopfact - fdop * slant_range;
            double fnprime;
            if (use_analytic_derivative) {
                const double rdot_acc =
                    rx * acc[0] + ry * acc[1] + rz * acc[2];
                fnprime = rdot_acc - vel2 - fdop * (dopfact / slant_range);
            } else {
                fnprime = -vel2 + (fdop / slant_range) * dopfact;
            }
            if (std::fabs(fnprime) < 1e-12) {
                done = 1;
            } else {
                step = fn / fnprime;
                if (std::fabs(step) < time_tol_s) {
                    done = 1;
                }
            }
        }
        aztime -= step;
        az_index[index] = (aztime - sensing_start_s) / azimuth_dt_s;
        range_index[index] = (slant_range - starting_range_m) / range_dt_m;
        converged[index] = done;
    }
    return 0;
}
