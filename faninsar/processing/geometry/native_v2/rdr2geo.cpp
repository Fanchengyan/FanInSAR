#include "native_v2_abi.h"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace faninsar_native_v2 {
namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr double radians_to_degrees = 180.0 / 3.14159265358979323846;

void validate_inputs(const Tensor& azimuth_index, const Tensor& range_index,
                     const Tensor& height_m, const Tensor& orbit_times_s,
                     const Tensor& orbit_positions_m,
                     const Tensor& orbit_velocities_m_s,
                     double azimuth_time_interval_s, double range_spacing_m,
                     double wavelength_m, int64_t max_iter, int64_t extra_iter,
                     double range_tol_m, double doppler_tol_hz) {
  check_vector(azimuth_index, "azimuth_index");
  check_vector(range_index, "range_index");
  check_vector(height_m, "height_m");
  check_vector(orbit_times_s, "orbit_times_s");
  TORCH_CHECK(azimuth_index.numel() == range_index.numel() &&
                  azimuth_index.numel() == height_m.numel(),
              "rdr2geo input arrays must have equal lengths");
  check_orbit(orbit_positions_m, "orbit_positions_m", orbit_times_s.numel());
  check_orbit(orbit_velocities_m_s, "orbit_velocities_m_s", orbit_times_s.numel());
  check_orbit_times(orbit_times_s);
  TORCH_CHECK(std::isfinite(azimuth_time_interval_s) && azimuth_time_interval_s > 0.0,
              "azimuth_time_interval_s must be finite and positive");
  TORCH_CHECK(std::isfinite(range_spacing_m) && range_spacing_m > 0.0,
              "range_spacing_m must be finite and positive");
  TORCH_CHECK(std::isfinite(wavelength_m) && wavelength_m > 0.0,
              "wavelength_m must be finite and positive");
  TORCH_CHECK(max_iter > 0 && extra_iter >= 0, "iteration settings are invalid");
  constexpr int64_t max_budget = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(max_iter <= max_budget && extra_iter <= max_budget - max_iter,
              "iteration budget exceeds int32 capacity");
  TORCH_CHECK(std::isfinite(range_tol_m) && range_tol_m > 0.0 &&
                  std::isfinite(doppler_tol_hz) && doppler_tol_hz > 0.0,
              "solver tolerances must be finite and positive");
}

Vec3 cross(const Vec3& left, const Vec3& right) {
  return {left[1] * right[2] - left[2] * right[1],
          left[2] * right[0] - left[0] * right[2],
          left[0] * right[1] - left[1] * right[0]};
}

Vec3 ecef_to_llh_tcn(const Vec3& xyz) {
  const double e4 = kWgs84EccentricitySquared * kWgs84EccentricitySquared;
  const double a2 = kWgs84SemiMajorAxisM * kWgs84SemiMajorAxisM;
  const double lateral = (xyz[0] * xyz[0] + xyz[1] * xyz[1]) / a2;
  const double polar = (1.0 - kWgs84EccentricitySquared) * xyz[2] * xyz[2] / a2;
  const double reduced = (lateral + polar - e4) / 6.0;
  if (!(reduced > 0.0) || !std::isfinite(reduced)) {
    return {kNan, kNan, kNan};
  }
  const double cubic = e4 * lateral * polar / (4.0 * reduced * reduced * reduced);
  const double cubic_radical = cubic * (2.0 + cubic);
  if (cubic_radical < 0.0 || !std::isfinite(cubic_radical)) {
    return {kNan, kNan, kNan};
  }
  const double root = std::cbrt(1.0 + cubic + std::sqrt(cubic_radical));
  if (!(std::abs(root) > 0.0) || !std::isfinite(root)) {
    return {kNan, kNan, kNan};
  }
  const double u = reduced * (1.0 + root + 1.0 / root);
  const double radial = std::sqrt(u * u + e4 * polar);
  const double w = kWgs84EccentricitySquared * (u + radial - polar) /
                   (2.0 * radial);
  const double k = std::sqrt(u + radial + w * w) - w;
  const double horizontal = std::hypot(xyz[0], xyz[1]);
  const double d = k * horizontal / (k + kWgs84EccentricitySquared);
  return {std::atan2(xyz[2], d), std::atan2(xyz[1], xyz[0]),
          (k + kWgs84EccentricitySquared - 1.0) * std::hypot(d, xyz[2]) / k};
}

double natural_spline_six(const double* values, double fraction) {
  constexpr double second_one[6] = {
      1.6076555023923444, -3.6459330143540667, 2.5837320574162677,
      -0.6889952153110048, 0.1722488038277512, -0.0287081339712919};
  constexpr double second_two[6] = {
      -0.4306220095693780, 2.5837320574162677, -4.3349282296650715,
      2.7559808612440193, -0.6889952153110048, 0.1148325358851674};
  const double fraction_squared = fraction * fraction;
  const double fraction_cubed = fraction_squared * fraction;
  double weights[6];
  for (int index = 0; index < 6; ++index) {
    weights[index] =
        fraction * (-second_one[index] / 3.0 - second_two[index] / 6.0) +
        fraction_squared * (second_one[index] / 2.0) +
        fraction_cubed * (second_two[index] - second_one[index]) / 6.0;
  }
  weights[1] += 1.0 - fraction;
  weights[2] += fraction;
  double result = 0.0;
  for (int index = 0; index < 6; ++index) {
    result += values[index] * weights[index];
  }
  return result;
}

struct DemView {
  const double* samples = nullptr;
  int rows = 0;
  int columns = 0;
};

double sample_dem_six(const DemView& dem, double latitude,
                      double longitude, double latitude_start,
                      double longitude_start, double latitude_spacing,
                      double longitude_spacing) {
  if (!std::isfinite(latitude) || !std::isfinite(longitude) ||
      !std::isfinite(latitude_spacing) || !std::isfinite(longitude_spacing) ||
      latitude_spacing == 0.0 || longitude_spacing == 0.0) {
    return kNan;
  }
  const double row = (latitude - latitude_start) / latitude_spacing;
  const double column = (longitude - longitude_start) / longitude_spacing;
  const int row_base = static_cast<int>(std::floor(row));
  const int column_base = static_cast<int>(std::floor(column));
  // The six-point stencil spans ``base - 1`` through ``base + 4``.  Keep the
  // base inside the supplied two-dimensional DEM instead of assuming a fixed
  // window shape.
  const int rows = dem.rows;
  const int columns = dem.columns;
  if (row_base < 1 || row_base > rows - 5 || column_base < 1 ||
      column_base > columns - 5) {
    return kNan;
  }
  const auto* samples = dem.samples;
  double along_rows[6]{};
  double window[6]{};
  const double column_fraction = column - column_base;
  const double row_fraction = row - row_base;
  for (int row_offset = -1; row_offset <= 4; ++row_offset) {
    for (int column_offset = -1; column_offset <= 4; ++column_offset) {
      window[column_offset + 1] =
          samples[(row_base + row_offset) * columns + column_base +
                  column_offset];
    }
    along_rows[row_offset + 1] = natural_spline_six(window, column_fraction);
  }
  return natural_spline_six(along_rows, row_fraction);
}

struct Rdr2GeoPointResult {
  double latitude = kNan;
  double longitude = kNan;
  double height = kNan;
  double final_height = kNan;
  bool initialized = false;
  bool converged = false;
  int32_t iterations = -1;
  double decision_residual = kNan;
  double final_residual = kNan;
  bool exhausted = false;
  double residual_range = kNan;
  double residual_doppler = kNan;
};

struct Rdr2GeoContext {
  bool initialized = false;
  OrbitState state{};
  double target_range = kNan;
  double satellite_norm = kNan;
  Vec3 normal{};
  Vec3 cross_track{};
  Vec3 along_track{};
  double normal_dot_velocity = kNan;
  double velocity_dot_along = kNan;
  double radius = kNan;
  double ellipsoid_height = kNan;
};

Rdr2GeoContext prepare_rdr2geo_context(
    double azimuth, double range, const double* times, const double* positions,
    const double* velocities, int64_t orbit_count, double sensing_offset_s,
    double azimuth_time_interval_s, double starting_slant_range_m,
    double range_spacing_m) {
  Rdr2GeoContext context;
  if (!std::isfinite(azimuth) || !std::isfinite(range)) return context;
  const double time_s = sensing_offset_s + azimuth * azimuth_time_interval_s;
  if (time_s < times[0] || time_s > times[orbit_count - 1]) return context;
  context.target_range = starting_slant_range_m + range * range_spacing_m;
  context.state = interpolate_orbit(times, positions, velocities, orbit_count,
                                    time_s);
  const double velocity_norm = norm(context.state.velocity);
  context.satellite_norm = norm(context.state.position);
  if (!(velocity_norm > 0.0) || !(context.satellite_norm > 0.0) ||
      !(context.target_range > 0.0) || !std::isfinite(context.target_range)) {
    return context;
  }
  const Vec3 velocity_unit{context.state.velocity[0] / velocity_norm,
                           context.state.velocity[1] / velocity_norm,
                           context.state.velocity[2] / velocity_norm};
  context.normal = {-context.state.position[0] / context.satellite_norm,
                    -context.state.position[1] / context.satellite_norm,
                    -context.state.position[2] / context.satellite_norm};
  context.cross_track = cross(context.normal, context.state.velocity);
  const double cross_track_norm = norm(context.cross_track);
  if (!(cross_track_norm > 0.0) || !std::isfinite(cross_track_norm)) {
    return context;
  }
  context.cross_track = {context.cross_track[0] / cross_track_norm,
                         context.cross_track[1] / cross_track_norm,
                         context.cross_track[2] / cross_track_norm};
  context.along_track = cross(context.cross_track, context.normal);
  const double along_track_norm = norm(context.along_track);
  if (!(along_track_norm > 0.0) || !std::isfinite(along_track_norm)) {
    return context;
  }
  context.along_track = {context.along_track[0] / along_track_norm,
                         context.along_track[1] / along_track_norm,
                         context.along_track[2] / along_track_norm};
  context.normal_dot_velocity = dot(context.normal, velocity_unit);
  context.velocity_dot_along = dot(velocity_unit, context.along_track);
  if (!std::isfinite(context.normal_dot_velocity) ||
      !(context.velocity_dot_along > 1.0e-12)) {
    return context;
  }
  const double minor = kWgs84SemiMajorAxisM *
                       std::sqrt(1.0 - kWgs84EccentricitySquared);
  const double eta = 1.0 / std::sqrt(
      (context.state.position[0] / kWgs84SemiMajorAxisM) *
          (context.state.position[0] / kWgs84SemiMajorAxisM) +
      (context.state.position[1] / kWgs84SemiMajorAxisM) *
          (context.state.position[1] / kWgs84SemiMajorAxisM) +
      (context.state.position[2] / minor) * (context.state.position[2] / minor));
  context.radius = eta * context.satellite_norm;
  context.ellipsoid_height = (1.0 - eta) * context.satellite_norm;
  context.initialized = true;
  return context;
}

Rdr2GeoPointResult solve_rdr2geo_point(
    double azimuth, double range, double height_seed, const double* times,
    const double* positions, const double* velocities, int64_t orbit_count,
    double sensing_offset_s, double azimuth_time_interval_s,
    double starting_slant_range_m, double range_spacing_m, double wavelength_m,
    int64_t budget, int64_t primary_budget, double range_tol_m,
    bool right_looking,
    double dem_height_tol_m, const Rdr2GeoContext* cached_context = nullptr,
    const DemView* dem_view = nullptr, double dem_latitude_start_deg = 0.0,
    double dem_longitude_start_deg = 0.0, double dem_latitude_spacing_deg = 1.0,
    double dem_longitude_spacing_deg = 1.0, double dem_min = -kNan,
    double dem_max = kNan) {
  Rdr2GeoPointResult result;
  if (!std::isfinite(height_seed)) return result;
  const Rdr2GeoContext context = cached_context == nullptr
                                     ? prepare_rdr2geo_context(
                                           azimuth, range, times, positions,
                                           velocities, orbit_count,
                                           sensing_offset_s,
                                           azimuth_time_interval_s,
                                           starting_slant_range_m,
                                           range_spacing_m)
                                     : *cached_context;
  if (!context.initialized) return result;
  result.initialized = true;
  const OrbitState& state = context.state;
  const double target_range = context.target_range;
  const double satellite_norm = context.satellite_norm;
  const Vec3& normal = context.normal;
  const Vec3& cross_track = context.cross_track;
  const Vec3& along_track = context.along_track;
  const double normal_dot_velocity = context.normal_dot_velocity;
  const double velocity_dot_along = context.velocity_dot_along;
  const double radius = context.radius;
  const double ellipsoid_height = context.ellipsoid_height;
  double height = height_seed;
  double latitude_rad = kNan;
  double longitude_rad = kNan;
  double last_range_residual = kNan;
  double last_doppler = kNan;
  bool lane_solved = false;
  bool early_failure = false;
  int64_t attempts_evaluated = 0;
  double previous_height = kNan;
  double previous_fixed_height = kNan;
  double old_latitude_rad = kNan;
  double old_longitude_rad = kNan;
  double old_height = kNan;
  bool aitken_restart = false;
  for (int64_t iteration = 0; iteration < budget; ++iteration) {
    ++attempts_evaluated;
    const bool active = ellipsoid_height - height < target_range;
    if (!active) {
      early_failure = true;
      break;
    }
    const double semi_minor = radius + height;
    const double cos_theta = 0.5 *
        (satellite_norm / target_range + target_range / satellite_norm -
         (semi_minor / satellite_norm) * (semi_minor / target_range));
    const double sin_theta =
        std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    const double gamma = target_range * cos_theta;
    const double alpha = -gamma * normal_dot_velocity /
                         std::max(velocity_dot_along, 1.0e-12);
    const double beta_argument = (target_range * sin_theta) *
                                     (target_range * sin_theta) - alpha * alpha;
    if (!std::isfinite(beta_argument) || beta_argument < -1.0e-6) {
      early_failure = true;
      break;
    }
    const double beta = (right_looking ? 1.0 : -1.0) *
                        std::sqrt(std::max(0.0, beta_argument));
    const Vec3 target_xyz{
        state.position[0] + alpha * along_track[0] + beta * cross_track[0] +
            gamma * normal[0],
        state.position[1] + alpha * along_track[1] + beta * cross_track[1] +
            gamma * normal[1],
        state.position[2] + alpha * along_track[2] + beta * cross_track[2] +
            gamma * normal[2]};
    const Vec3 llh = ecef_to_llh_tcn(target_xyz);
    if (!std::isfinite(llh[0]) || !std::isfinite(llh[1])) {
      early_failure = true;
      break;
    }
    latitude_rad = llh[0];
    longitude_rad = llh[1];
    const double latitude_deg = latitude_rad * radians_to_degrees;
    const double longitude_deg = longitude_rad * radians_to_degrees;
    // Evaluate the range/Doppler residual at the TCN candidate.  The DEM
    // height is sampled from that candidate LLH, then fed back as the next
    // fixed-point state.
    const double dem_height =
        dem_view == nullptr
            ? height_seed
            : sample_dem_six(*dem_view, latitude_deg, longitude_deg,
                             dem_latitude_start_deg, dem_longitude_start_deg,
                             dem_latitude_spacing_deg,
                             dem_longitude_spacing_deg);
    if (!std::isfinite(dem_height)) {
      early_failure = true;
      break;
    }
    const Vec3 dem_xyz = llh_to_ecef(latitude_deg, longitude_deg, dem_height);
    const Vec3 look_vector{dem_xyz[0] - state.position[0],
                           dem_xyz[1] - state.position[1],
                           dem_xyz[2] - state.position[2]};
    const double slant_range = norm(look_vector);
    last_range_residual = slant_range - target_range;
    const double look_norm = norm(look_vector);
    last_doppler = look_norm > 0.0
                       ? 2.0 * dot(state.velocity,
                                   Vec3{look_vector[0] / look_norm,
                                        look_vector[1] / look_norm,
                                        look_vector[2] / look_norm}) /
                             wavelength_m
                       : kNan;
    result.decision_residual = last_range_residual;
    result.residual_range = last_range_residual;
    result.residual_doppler = last_doppler;
    const double next_height = norm(dem_xyz) - radius;
    if (!std::isfinite(next_height)) {
      early_failure = true;
      break;
    }
    double selected_height = next_height;
    const bool dem_height_converged =
        dem_view == nullptr ||
        std::abs(next_height - height) <= dem_height_tol_m;
    const bool newly_solved =
        std::abs(last_range_residual) < range_tol_m && dem_height_converged;
    bool aitken_enabled = false;
    if (dem_view != nullptr && iteration >= 2) {
      const double denominator = next_height - 2.0 * height + previous_height;
      const double step = next_height - height;
      const double previous_step = height - previous_height;
      const double slope = step / previous_step;
      const double candidate =
          previous_height - previous_step * previous_step / denominator;
      const bool finite_history =
          std::isfinite(previous_height) && std::isfinite(height) &&
          std::isfinite(next_height) && std::isfinite(slope) &&
          std::isfinite(candidate) && std::isfinite(dem_min) &&
          std::isfinite(dem_max);
      const bool denominator_safe = std::abs(denominator) > 1.0e-6;
      const bool previous_step_safe = std::abs(previous_step) > 1.0e-12;
      bool accelerated =
          finite_history && denominator_safe && previous_step_safe &&
          slope > -0.95 && slope < 0.95 && dem_min <= dem_max &&
          std::abs(candidate - height) <= std::max(std::abs(step), 1.0e-3) &&
          (candidate - height) * step >= 0.0 && candidate >= dem_min &&
          candidate <= dem_max;
      if (accelerated) {
        const double prior_residual = previous_fixed_height - previous_height;
        const double current_residual = next_height - height;
        const bool sign_change = prior_residual * current_residual <= 0.0;
        const bool between = (candidate - previous_height) *
                                 (candidate - height) <= 0.0;
        accelerated = std::isfinite(previous_fixed_height) &&
                      ((!sign_change) || between);
      }
      aitken_enabled = accelerated && !aitken_restart;
      if (aitken_enabled) selected_height = candidate;
    }
    bool restart_after_damping = false;
    double next_old_height = dem_height;
    if (dem_view != nullptr && iteration >= primary_budget) {
      // Match Torch's extra-budget transition: average the previous LLH/DEM
      // state with the current DEM ECEF point before the next TCN solve.
      const Vec3 old_xyz = llh_to_ecef(
          old_latitude_rad * radians_to_degrees,
          old_longitude_rad * radians_to_degrees, old_height);
      const Vec3 average_xyz{
          0.5 * (old_xyz[0] + dem_xyz[0]),
          0.5 * (old_xyz[1] + dem_xyz[1]),
          0.5 * (old_xyz[2] + dem_xyz[2])};
      const Vec3 average_llh = ecef_to_llh_tcn(average_xyz);
      const double average_height = norm(average_xyz) - radius;
      const bool damping_valid =
          !newly_solved && std::isfinite(old_latitude_rad) &&
          std::isfinite(old_longitude_rad) && std::isfinite(old_height) &&
          std::isfinite(average_llh[0]) && std::isfinite(average_llh[1]) &&
          std::isfinite(average_llh[2]) && std::isfinite(average_height);
      if (damping_valid) {
        latitude_rad = average_llh[0];
        longitude_rad = average_llh[1];
        next_old_height = average_llh[2];
        selected_height = average_height;
        aitken_enabled = false;
        restart_after_damping = !aitken_restart;
      }
    }
    if (newly_solved) {
      lane_solved = true;
      height = next_height;
      break;
    }
    old_latitude_rad = latitude_rad;
    old_longitude_rad = longitude_rad;
    old_height = next_old_height;
    if (restart_after_damping) {
      previous_height = kNan;
      previous_fixed_height = kNan;
      aitken_restart = true;
    } else {
      previous_height = height;
      previous_fixed_height = next_height;
      aitken_restart = false;
    }
    height = selected_height;
  }
  if (!lane_solved) {
    if (early_failure || attempts_evaluated < budget) return result;
    result.exhausted = true;
    result.iterations = static_cast<int32_t>(budget);
  }
  const double final_semi_minor = radius + height;
  const double final_cos_theta = 0.5 *
      (satellite_norm / target_range + target_range / satellite_norm -
       (final_semi_minor / satellite_norm) *
           (final_semi_minor / target_range));
  const double final_sin_theta =
      std::sqrt(std::max(0.0, 1.0 - final_cos_theta * final_cos_theta));
  const double final_gamma = target_range * final_cos_theta;
  const double final_alpha = -final_gamma * normal_dot_velocity /
                             std::max(velocity_dot_along, 1.0e-12);
  const double final_beta_argument =
      (target_range * final_sin_theta) * (target_range * final_sin_theta) -
      final_alpha * final_alpha;
  if (!std::isfinite(final_beta_argument) || final_beta_argument < -1.0e-6) {
    return result;
  }
  const double final_beta = (right_looking ? 1.0 : -1.0) *
                            std::sqrt(std::max(0.0, final_beta_argument));
  const Vec3 final_xyz{
      state.position[0] + final_alpha * along_track[0] +
          final_beta * cross_track[0] + final_gamma * normal[0],
      state.position[1] + final_alpha * along_track[1] +
          final_beta * cross_track[1] + final_gamma * normal[1],
      state.position[2] + final_alpha * along_track[2] +
          final_beta * cross_track[2] + final_gamma * normal[2]};
  const Vec3 final_llh = ecef_to_llh_tcn(final_xyz);
  if (!std::isfinite(final_llh[0]) || !std::isfinite(final_llh[1])) {
    return result;
  }
  result.final_height = final_llh[2];
  const Vec3 final_look{final_xyz[0] - state.position[0],
                        final_xyz[1] - state.position[1],
                        final_xyz[2] - state.position[2]};
  const double final_slant_range = norm(final_look);
  if (!(final_slant_range > 0.0) || !std::isfinite(final_slant_range)) {
    return result;
  }
  const Vec3 final_unit{final_look[0] / final_slant_range,
                        final_look[1] / final_slant_range,
                        final_look[2] / final_slant_range};
  const double final_range_residual = final_slant_range - target_range;
  const double final_doppler =
      2.0 * dot(state.velocity, final_unit) / wavelength_m;
  result.residual_range = final_range_residual;
  result.residual_doppler = final_doppler;
  result.decision_residual = final_range_residual;
  result.final_residual = final_range_residual;
  result.latitude = final_llh[0] * radians_to_degrees;
  result.longitude = final_llh[1] * radians_to_degrees;
  result.height = height;
  if (!(std::abs(final_range_residual) < range_tol_m)) {
    return result;
  }
  result.converged = true;
  result.iterations = static_cast<int32_t>(attempts_evaluated);
  return result;
}

}  // namespace

std::vector<Tensor> rdr2geo_cpu(
    const Tensor& azimuth_index, const Tensor& range_index, const Tensor& height_m,
    const Tensor& orbit_times_s, const Tensor& orbit_positions_m,
    const Tensor& orbit_velocities_m_s, double sensing_offset_s,
    double azimuth_time_interval_s, double starting_slant_range_m,
    double range_spacing_m, double wavelength_m, int64_t max_iter,
    int64_t extra_iter, double range_tol_m, double doppler_tol_hz,
    bool right_looking) {
  validate_inputs(azimuth_index, range_index, height_m, orbit_times_s,
                  orbit_positions_m, orbit_velocities_m_s,
                  azimuth_time_interval_s, range_spacing_m, wavelength_m,
                  max_iter, extra_iter, range_tol_m, doppler_tol_hz);
  TORCH_CHECK(std::isfinite(sensing_offset_s) && std::isfinite(starting_slant_range_m),
              "radar timing and range origins must be finite");
  const int64_t count = azimuth_index.numel();
  auto options = azimuth_index.options();
  auto latitude = torch::full({count}, kNan, options);
  auto longitude = torch::full({count}, kNan, options);
  auto heights = height_m.clone();
  auto ranges = range_index.clone();
  auto azimuths = azimuth_index.clone();
  auto converged = torch::zeros({count}, options.dtype(torch::kBool));
  auto iterations = torch::full({count}, -1, options.dtype(torch::kInt32));
  auto decision_residual = torch::full({count}, kNan, options);
  auto final_residual = torch::full({count}, kNan, options);
  auto tolerance = torch::full({count}, range_tol_m, options);
  auto exhausted = torch::zeros({count}, options.dtype(torch::kBool));
  auto boundary_rechecked = torch::zeros({count}, options.dtype(torch::kBool));
  auto residual_range = torch::full({count}, kNan, options);
  auto residual_doppler = torch::full({count}, kNan, options);

  const auto* azimuths_in = azimuth_index.data_ptr<double>();
  const auto* ranges_in = range_index.data_ptr<double>();
  const auto* heights_in = height_m.data_ptr<double>();
  const auto* times = orbit_times_s.data_ptr<double>();
  const auto* positions = orbit_positions_m.data_ptr<double>();
  const auto* velocities = orbit_velocities_m_s.data_ptr<double>();
  auto* latitudes = latitude.data_ptr<double>();
  auto* longitudes = longitude.data_ptr<double>();
  auto* heights_out = heights.data_ptr<double>();
  auto* ranges_out = ranges.data_ptr<double>();
  auto* azimuths_out = azimuths.data_ptr<double>();
  auto* solved = converged.data_ptr<bool>();
  auto* iteration_values = iterations.data_ptr<int32_t>();
  auto* decisions = decision_residual.data_ptr<double>();
  auto* finals = final_residual.data_ptr<double>();
  auto* tolerances = tolerance.data_ptr<double>();
  auto* exhausted_values = exhausted.data_ptr<bool>();
  auto* range_residuals = residual_range.data_ptr<double>();
  auto* doppler_residuals = residual_doppler.data_ptr<double>();
  const int64_t budget = max_iter + extra_iter;
  const double orbit_start = times[0];
  const double orbit_end = times[orbit_times_s.numel() - 1];
  begin_telemetry(count, "rdr2geo_cpu");
  auto invalidate_lane = [&](int64_t point) {
    latitudes[point] = kNan;
    longitudes[point] = kNan;
    heights_out[point] = kNan;
    ranges_out[point] = kNan;
    azimuths_out[point] = kNan;
    tolerances[point] = kNan;
    solved[point] = false;
    iteration_values[point] = -1;
    decisions[point] = kNan;
    finals[point] = kNan;
    exhausted_values[point] = false;
    range_residuals[point] = kNan;
    doppler_residuals[point] = kNan;
  };

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t point = 0; point < count; ++point) {
    record_visit(point);
    const double azimuth = azimuths_in[point];
    const double range = ranges_in[point];
    const double height_seed = heights_in[point];
    if (!std::isfinite(azimuth) || !std::isfinite(range) || !std::isfinite(height_seed)) {
      invalidate_lane(point);
      continue;
    }
    double time_s = sensing_offset_s + azimuth * azimuth_time_interval_s;
    if (time_s < orbit_start || time_s > orbit_end) {
      invalidate_lane(point);
      continue;
    }
    const double target_range = starting_slant_range_m + range * range_spacing_m;
    const OrbitState state = interpolate_orbit(times, positions, velocities,
                                               orbit_times_s.numel(), time_s);
    const double velocity_norm = norm(state.velocity);
    const double satellite_norm = norm(state.position);
    if (!(velocity_norm > 0.0) || !(satellite_norm > 0.0) ||
        !(target_range > 0.0) || !std::isfinite(target_range)) {
      invalidate_lane(point);
      continue;
    }
    const Vec3 velocity_unit{state.velocity[0] / velocity_norm,
                             state.velocity[1] / velocity_norm,
                             state.velocity[2] / velocity_norm};
    const Vec3 normal{-state.position[0] / satellite_norm,
                      -state.position[1] / satellite_norm,
                      -state.position[2] / satellite_norm};
    Vec3 cross_track = cross(normal, state.velocity);
    const double cross_track_norm = norm(cross_track);
    if (!(cross_track_norm > 0.0) || !std::isfinite(cross_track_norm)) {
      invalidate_lane(point);
      continue;
    }
    cross_track = {cross_track[0] / cross_track_norm,
                   cross_track[1] / cross_track_norm,
                   cross_track[2] / cross_track_norm};
    Vec3 along_track = cross(cross_track, normal);
    const double along_track_norm = norm(along_track);
    if (!(along_track_norm > 0.0) || !std::isfinite(along_track_norm)) {
      invalidate_lane(point);
      continue;
    }
    along_track = {along_track[0] / along_track_norm,
                   along_track[1] / along_track_norm,
                   along_track[2] / along_track_norm};
    const double normal_dot_velocity = dot(normal, velocity_unit);
    const double velocity_dot_along = dot(velocity_unit, along_track);
    if (!std::isfinite(normal_dot_velocity) ||
        !(velocity_dot_along > 1.0e-12)) {
      invalidate_lane(point);
      continue;
    }
    const double minor = kWgs84SemiMajorAxisM *
                         std::sqrt(1.0 - kWgs84EccentricitySquared);
    const double eta = 1.0 / std::sqrt(
        (state.position[0] / kWgs84SemiMajorAxisM) *
            (state.position[0] / kWgs84SemiMajorAxisM) +
        (state.position[1] / kWgs84SemiMajorAxisM) *
            (state.position[1] / kWgs84SemiMajorAxisM) +
        (state.position[2] / minor) * (state.position[2] / minor));
    const double radius = eta * satellite_norm;
    const double ellipsoid_height = (1.0 - eta) * satellite_norm;
    double height = height_seed;
    double latitude_rad = kNan;
    double longitude_rad = kNan;
    double last_range_residual = kNan;
    double last_doppler = kNan;
    bool lane_solved = false;
    bool early_failure = false;
    int64_t attempts_evaluated = 0;
    for (int64_t iteration = 0; iteration < budget; ++iteration) {
      ++attempts_evaluated;
      const bool active = ellipsoid_height - height < target_range;
      if (!active) {
        early_failure = true;
        break;
      }
      const double semi_minor = radius + height;
      const double cos_theta = 0.5 *
          (satellite_norm / target_range + target_range / satellite_norm -
           (semi_minor / satellite_norm) * (semi_minor / target_range));
      const double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
      const double gamma = target_range * cos_theta;
      const double alpha = -gamma * normal_dot_velocity /
                           std::max(velocity_dot_along, 1.0e-12);
      const double beta_argument = (target_range * sin_theta) *
                                       (target_range * sin_theta) - alpha * alpha;
      if (!std::isfinite(beta_argument) || beta_argument < -1.0e-6) {
        early_failure = true;
        break;
      }
      const double beta = (right_looking ? 1.0 : -1.0) *
                          std::sqrt(std::max(0.0, beta_argument));
      const Vec3 target_xyz{
          state.position[0] + alpha * along_track[0] + beta * cross_track[0] +
              gamma * normal[0],
          state.position[1] + alpha * along_track[1] + beta * cross_track[1] +
              gamma * normal[1],
          state.position[2] + alpha * along_track[2] + beta * cross_track[2] +
              gamma * normal[2]};
      const Vec3 llh = ecef_to_llh_tcn(target_xyz);
      if (!std::isfinite(llh[0]) || !std::isfinite(llh[1])) {
        early_failure = true;
        break;
      }
      latitude_rad = llh[0];
      longitude_rad = llh[1];
      const double latitude_deg = latitude_rad * radians_to_degrees;
      const double longitude_deg = longitude_rad * radians_to_degrees;
      const Vec3 dem_xyz = llh_to_ecef(latitude_deg, longitude_deg, height_seed);
      const double slant_range = norm(Vec3{state.position[0] - dem_xyz[0],
                                           state.position[1] - dem_xyz[1],
                                           state.position[2] - dem_xyz[2]});
      last_range_residual = slant_range - target_range;
      const Vec3 look_vector{dem_xyz[0] - state.position[0],
                             dem_xyz[1] - state.position[1],
                             dem_xyz[2] - state.position[2]};
      const double look_norm = norm(look_vector);
      last_doppler = look_norm > 0.0
                         ? 2.0 * dot(state.velocity,
                                     Vec3{look_vector[0] / look_norm,
                                          look_vector[1] / look_norm,
                                          look_vector[2] / look_norm}) /
                               wavelength_m
                         : kNan;
      decisions[point] = last_range_residual;
      range_residuals[point] = last_range_residual;
      doppler_residuals[point] = last_doppler;
      const double next_height = norm(dem_xyz) - radius;
      if (!std::isfinite(next_height)) {
        early_failure = true;
        break;
      }
      if (std::abs(last_range_residual) < range_tol_m) {
        lane_solved = true;
        height = height_seed;
        break;
      }
      height = next_height;
    }
    if (!lane_solved) {
      if (!early_failure && attempts_evaluated >= budget) {
        exhausted_values[point] = true;
        iteration_values[point] = static_cast<int32_t>(budget);
      } else {
        exhausted_values[point] = false;
        iteration_values[point] = -1;
      }
      continue;
    }
    const double final_semi_minor = radius + height;
    const double final_cos_theta = 0.5 *
        (satellite_norm / target_range + target_range / satellite_norm -
         (final_semi_minor / satellite_norm) *
             (final_semi_minor / target_range));
    const double final_sin_theta =
        std::sqrt(std::max(0.0, 1.0 - final_cos_theta * final_cos_theta));
    const double final_gamma = target_range * final_cos_theta;
    const double final_alpha = -final_gamma * normal_dot_velocity /
                               std::max(velocity_dot_along, 1.0e-12);
    const double final_beta_argument =
        (target_range * final_sin_theta) * (target_range * final_sin_theta) -
        final_alpha * final_alpha;
    if (!std::isfinite(final_beta_argument) || final_beta_argument < -1.0e-6) {
      solved[point] = false;
      iteration_values[point] = -1;
      exhausted_values[point] = false;
      continue;
    }
    const double final_beta = (right_looking ? 1.0 : -1.0) *
                              std::sqrt(std::max(0.0, final_beta_argument));
    const Vec3 final_xyz{
        state.position[0] + final_alpha * along_track[0] +
            final_beta * cross_track[0] + final_gamma * normal[0],
        state.position[1] + final_alpha * along_track[1] +
            final_beta * cross_track[1] + final_gamma * normal[1],
        state.position[2] + final_alpha * along_track[2] +
            final_beta * cross_track[2] + final_gamma * normal[2]};
    const Vec3 final_llh = ecef_to_llh_tcn(final_xyz);
    if (!std::isfinite(final_llh[0]) || !std::isfinite(final_llh[1])) {
      solved[point] = false;
      iteration_values[point] = -1;
      exhausted_values[point] = false;
      continue;
    }
    const double final_latitude_deg = final_llh[0] * radians_to_degrees;
    const double final_longitude_deg = final_llh[1] * radians_to_degrees;
    const Vec3 final_look{final_xyz[0] - state.position[0],
                          final_xyz[1] - state.position[1],
                          final_xyz[2] - state.position[2]};
    const double final_slant_range = norm(final_look);
    if (!(final_slant_range > 0.0) || !std::isfinite(final_slant_range)) {
      solved[point] = false;
      iteration_values[point] = -1;
      exhausted_values[point] = false;
      continue;
    }
    const Vec3 final_unit{final_look[0] / final_slant_range,
                          final_look[1] / final_slant_range,
                          final_look[2] / final_slant_range};
    const double final_range_residual = final_slant_range - target_range;
    const double final_doppler =
        2.0 * dot(state.velocity, final_unit) / wavelength_m;
    range_residuals[point] = final_range_residual;
    doppler_residuals[point] = final_doppler;
    decisions[point] = final_range_residual;
    finals[point] = final_range_residual;
    if (!(std::abs(final_range_residual) < range_tol_m)) {
      solved[point] = false;
      iteration_values[point] = -1;
      exhausted_values[point] = false;
      continue;
    }
    latitudes[point] = final_latitude_deg;
    longitudes[point] = final_longitude_deg;
    heights_out[point] = height;
    solved[point] = true;
    iteration_values[point] = static_cast<int32_t>(attempts_evaluated);
    exhausted_values[point] = false;
  }
  return {latitude, longitude, heights, ranges, azimuths, converged, iterations,
          decision_residual, final_residual, tolerance, exhausted,
          boundary_rechecked, residual_range, residual_doppler};
}

std::vector<Tensor> rdr2geo_cpu_dem(
    const Tensor& azimuth_index, const Tensor& range_index,
    const Tensor& height_seed_m, const Tensor& orbit_times_s,
    const Tensor& orbit_positions_m, const Tensor& orbit_velocities_m_s,
    double sensing_offset_s, double azimuth_time_interval_s,
    double starting_slant_range_m, double range_spacing_m, double wavelength_m,
    int64_t max_iter, int64_t extra_iter, double range_tol_m,
    double doppler_tol_hz, bool right_looking, const Tensor& dem_samples,
    double dem_latitude_start_deg, double dem_longitude_start_deg,
    double dem_latitude_spacing_deg, double dem_longitude_spacing_deg,
    int64_t dem_iterations, double dem_height_tol_m) {
  validate_inputs(azimuth_index, range_index, height_seed_m, orbit_times_s,
                  orbit_positions_m, orbit_velocities_m_s,
                  azimuth_time_interval_s, range_spacing_m, wavelength_m,
                  max_iter, extra_iter, range_tol_m, doppler_tol_hz);
  TORCH_CHECK(std::isfinite(sensing_offset_s) &&
                  std::isfinite(starting_slant_range_m),
              "radar timing and range origins must be finite");
  TORCH_CHECK(!dem_samples.is_cuda() &&
                  dem_samples.scalar_type() == torch::kFloat64 &&
                  dem_samples.is_contiguous() && dem_samples.dim() == 2 &&
                  dem_samples.size(0) >= 6 && dem_samples.size(1) >= 6,
              "dem_samples must be a contiguous CPU float64 array with both "
              "dimensions at least 6");
  TORCH_CHECK(torch::isfinite(dem_samples).all().item<bool>(),
              "dem_samples must contain only finite values");
  TORCH_CHECK(dem_iterations > 0 && std::isfinite(dem_height_tol_m) &&
                  dem_height_tol_m > 0.0,
              "DEM fixed-point settings are invalid");
  TORCH_CHECK(std::isfinite(dem_latitude_start_deg) &&
                  std::isfinite(dem_longitude_start_deg) &&
                  std::isfinite(dem_latitude_spacing_deg) &&
                  std::isfinite(dem_longitude_spacing_deg) &&
                  dem_latitude_spacing_deg != 0.0 &&
                  dem_longitude_spacing_deg != 0.0,
              "DEM origin and spacing must be finite and non-zero");
  const int64_t count = height_seed_m.numel();
  auto options = height_seed_m.options();
  auto latitude = torch::full({count}, kNan, options);
  auto longitude = torch::full({count}, kNan, options);
  auto heights = height_seed_m.clone();
  auto ranges = range_index.clone();
  auto azimuths = azimuth_index.clone();
  auto converged = torch::zeros({count}, options.dtype(torch::kBool));
  auto iterations = torch::full({count}, -1, options.dtype(torch::kInt32));
  auto decision_residual = torch::full({count}, kNan, options);
  auto final_residual = torch::full({count}, kNan, options);
  auto tolerance = torch::full({count}, range_tol_m, options);
  auto exhausted = torch::zeros({count}, options.dtype(torch::kBool));
  auto boundary_rechecked = torch::zeros({count}, options.dtype(torch::kBool));
  auto residual_range = torch::full({count}, kNan, options);
  auto residual_doppler = torch::full({count}, kNan, options);

  const auto* azimuths_in = azimuth_index.data_ptr<double>();
  const auto* ranges_in = range_index.data_ptr<double>();
  const auto* heights_in = height_seed_m.data_ptr<double>();
  const auto* times = orbit_times_s.data_ptr<double>();
  const auto* positions = orbit_positions_m.data_ptr<double>();
  const auto* velocities = orbit_velocities_m_s.data_ptr<double>();
  auto* latitudes = latitude.data_ptr<double>();
  auto* longitudes = longitude.data_ptr<double>();
  auto* heights_out = heights.data_ptr<double>();
  auto* ranges_out = ranges.data_ptr<double>();
  auto* azimuths_out = azimuths.data_ptr<double>();
  auto* solved = converged.data_ptr<bool>();
  auto* iteration_values = iterations.data_ptr<int32_t>();
  auto* decisions = decision_residual.data_ptr<double>();
  auto* finals = final_residual.data_ptr<double>();
  auto* tolerances = tolerance.data_ptr<double>();
  auto* exhausted_values = exhausted.data_ptr<bool>();
  auto* range_residuals = residual_range.data_ptr<double>();
  auto* doppler_residuals = residual_doppler.data_ptr<double>();
  const int64_t budget = max_iter + extra_iter;
  const double dem_min = dem_samples.min().item<double>();
  const double dem_max = dem_samples.max().item<double>();
  const DemView dem_view{dem_samples.data_ptr<double>(),
                         static_cast<int>(dem_samples.size(0)),
                         static_cast<int>(dem_samples.size(1))};
  (void)dem_iterations;
  begin_telemetry(count, "rdr2geo_cpu_dem");
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t point = 0; point < count; ++point) {
    record_visit(point);
    const Rdr2GeoContext context = prepare_rdr2geo_context(
        azimuths_in[point], ranges_in[point], times, positions, velocities,
        orbit_times_s.numel(), sensing_offset_s, azimuth_time_interval_s,
        starting_slant_range_m, range_spacing_m);
    const Rdr2GeoPointResult final_result = solve_rdr2geo_point(
        azimuths_in[point], ranges_in[point], heights_in[point], times,
        positions, velocities, orbit_times_s.numel(), sensing_offset_s,
        azimuth_time_interval_s, starting_slant_range_m, range_spacing_m,
        wavelength_m, budget, max_iter, range_tol_m, right_looking,
        dem_height_tol_m,
        &context, &dem_view, dem_latitude_start_deg,
        dem_longitude_start_deg, dem_latitude_spacing_deg,
        dem_longitude_spacing_deg, dem_min, dem_max);
    if (!final_result.initialized) {
      latitudes[point] = kNan;
      longitudes[point] = kNan;
      heights_out[point] = kNan;
      ranges_out[point] = kNan;
      azimuths_out[point] = kNan;
      tolerances[point] = kNan;
      solved[point] = false;
      iteration_values[point] = -1;
      decisions[point] = kNan;
      finals[point] = kNan;
      exhausted_values[point] = false;
      range_residuals[point] = kNan;
      doppler_residuals[point] = kNan;
      continue;
    }
    const bool final_coordinates_valid =
        std::isfinite(final_result.latitude) &&
        std::isfinite(final_result.longitude) &&
        std::isfinite(final_result.final_height);
    latitudes[point] = final_coordinates_valid ? final_result.latitude : kNan;
    longitudes[point] = final_coordinates_valid ? final_result.longitude : kNan;
    heights_out[point] = final_coordinates_valid ? final_result.final_height : kNan;
    const double final_dem_height = final_coordinates_valid
                                        ? sample_dem_six(
                                              dem_view, final_result.latitude,
                                              final_result.longitude,
                                              dem_latitude_start_deg,
                                              dem_longitude_start_deg,
                                              dem_latitude_spacing_deg,
                                              dem_longitude_spacing_deg)
                                        : kNan;
    const bool final_dem_valid = std::isfinite(final_dem_height);
    const bool final_dem_converged =
        final_dem_valid && std::isfinite(final_result.final_height) &&
        std::abs(final_result.final_height - final_dem_height) <=
            dem_height_tol_m;
    solved[point] = final_result.converged && final_dem_converged;
    iteration_values[point] = final_result.iterations;
    decisions[point] = final_result.decision_residual;
    finals[point] = final_result.final_residual;
    exhausted_values[point] = final_result.exhausted;
    range_residuals[point] = final_result.residual_range;
    doppler_residuals[point] = final_result.residual_doppler;
  }
  return {latitude, longitude, heights, ranges, azimuths, converged, iterations,
          decision_residual, final_residual, tolerance, exhausted,
          boundary_rechecked, residual_range, residual_doppler};
}

}  // namespace faninsar_native_v2
