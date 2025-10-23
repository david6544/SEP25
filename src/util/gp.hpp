#pragma once

#include "io.hpp"

#include <cstddef>
#include <iterator>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

#include <Eigen/Dense>

namespace skx::algo::gp { // gaussian process

	namespace kernel {
		template<std::ranges::random_access_range RangeLengthScaleContainer>
			requires std::convertible_to<std::ranges::range_value_t<RangeLengthScaleContainer>, double>
		struct RbfArdParams {
			RangeLengthScaleContainer length_scale_invsq;
			double signal_variance_sq;
		};

		template <typename C, std::ranges::random_access_range RangePointA, std::ranges::random_access_range RangePointB>
		double rbf_ard(const RbfArdParams<C>& params, const RangePointA& point_a, const RangePointB& point_b) {
			double sum = 0.0;
			for (size_t i = 0; i < params.length_scale_invsq.size(); i++) {
				std::ranges::range_value_t<RangePointA> xa{};
				std::ranges::range_value_t<RangePointB> xb{};
				if (i < point_a.size()) {
					xa = point_a[i];
				}
				if (i < point_b.size()) {
					xb = point_b[i];
				}
				sum += std::pow<double>(static_cast<double>(xa - xb), 2.0) * static_cast<double>(params.length_scale_invsq[i]);
			}
			double ker = params.signal_variance_sq * std::exp(-0.5 * sum);
			return ker;
		}

		template <std::ranges::random_access_range RangeParams, std::ranges::random_access_range RangePointA, std::ranges::random_access_range RangePointB>
		double ker_rbf_ard(const RangeParams& params, const RangePointA& point_a, const RangePointB& point_b) {
			using ParamValT = std::ranges::range_value_t<RangeParams>;
			const ParamValT& sigma_f_sq = params[0];
			const auto length_scales_invsq = std::ranges::subrange(params.begin() + 1, params.end());

			double sum = 0.0;
			for (size_t i = 0; i < length_scales_invsq.size(); i++) {
				std::ranges::range_value_t<RangePointA> xa{};
				std::ranges::range_value_t<RangePointB> xb{};
				if (i < point_a.size()) {
					xa = point_a[i];
				}
				if (i < point_b.size()) {
					xb = point_b[i];
				}
				sum += std::pow<double>(static_cast<double>(xa - xb), 2.0) * static_cast<double>(length_scales_invsq[i]);
			}
			double ker = sigma_f_sq * std::exp(-0.5 * sum);
			return ker;
		}

		void ker_rbf_ard_grad(
			Eigen::MatrixXd& grad_out,
			const Eigen::MatrixXd& sample_points,
			const Eigen::VectorXd& params,
			size_t param_index)
		{
			using Eigen::Index;

			const Index nsamples = sample_points.rows();
			const Index ndims = sample_points.cols();

			grad_out.setZero(nsamples, nsamples);

			// There are (ndims + 1) parameters: [sigma_f_sq, q_1, ..., q_ndims]
			if (param_index > static_cast<size_t>(ndims)) {
				// size mismatch; nothing to do
				return;
			}

			const double sigma_f_sq = params(0);
			// q vector (length_scales_invsq), linear space
			Eigen::VectorXd q = params.segment(1, ndims);

			for (Index i = 0; i < nsamples; ++i) {
				for (Index j = i; j < nsamples; ++j) {
					// E_ij = sum_d q_d * (Δx_d)^2
					double E = 0.0;
					for (Index d = 0; d < ndims; ++d) {
						const double diff = sample_points(i, d) - sample_points(j, d);
						E += q(d) * diff * diff;
					}

					const double exp_term = std::exp(-0.5 * E);
					const double Kij = sigma_f_sq * exp_term;

					double deriv = 0.0;
					if (param_index == 0) {
						// dK/d(sigma_f_sq) = exp(-0.5 * E_ij)
						deriv = exp_term;
					}
					else {
						// dK/d(q_m) = -0.5 * (Δx_m)^2 * K_ij
						const Index m = static_cast<Index>(param_index - 1);
						const double diff_m = sample_points(i, m) - sample_points(j, m);
						deriv = -0.5 * (diff_m * diff_m) * Kij;
					}

					grad_out(i, j) = deriv;
					grad_out(j, i) = deriv; // symmetry
				}
			}
		}

		// PERIODIC-ARD kernel with independent per-dimension periods.
// Linear-space params layout: [ sigma_f_sq, q_1, ..., q_D, p_1, ..., p_D ]
// where q_d = l_d^{-2} and p_d > 0 is the period for dimension d.
// K_ij = sigma_f_sq * exp( - sum_d 2*q_d * sin^2( pi * (x_i,d - x_j,d) / p_d ) )
		template <std::ranges::random_access_range RangeParams,
			std::ranges::random_access_range RangePointA,
			std::ranges::random_access_range RangePointB>
		double ker_periodic_ard(const RangeParams& params, const RangePointA& point_a, const RangePointB& point_b) {
			using ParamValT = std::ranges::range_value_t<RangeParams>;
			const ParamValT& sigma_f_sq = params[0];

			const auto P = static_cast<size_t>(std::ranges::distance(params.begin(), params.end()));
			const size_t D = (P > 1) ? (P - 1) / 2 : 0; // expect P == 1 + 2D

			const auto q_begin = params.begin() + 1;
			const auto q_end = q_begin + D;
			const auto p_begin = q_end;
			const auto p_end = params.end();

			constexpr double PI = 3.141592653589793238462643383279502884;

			double E = 0.0;
			for (size_t d = 0; d < D; ++d) {
				std::ranges::range_value_t<RangePointA> xa{};
				std::ranges::range_value_t<RangePointB> xb{};
				if (d < point_a.size()) xa = point_a[d];
				if (d < point_b.size()) xb = point_b[d];

				const double qd = static_cast<double>(*(q_begin + static_cast<std::ptrdiff_t>(d)));
				const double pd = static_cast<double>(*(p_begin + static_cast<std::ptrdiff_t>(d)));

				const double diff = static_cast<double>(xa) - static_cast<double>(xb);
				const double arg = PI * diff / pd;             // π * Δx / p_d
				const double s = std::sin(arg);
				E += 2.0 * qd * s * s;
			}

			return static_cast<double>(sigma_f_sq) * std::exp(-E);
		}

		// Gradient of PERIODIC-ARD kernel w.r.t. linear-space params:
		// params = [ sigma_f_sq, q_1..q_D, p_1..p_D ]
		// dK/d(sigma_f_sq) = exp(-E)
		// dK/d(q_m)        = - (2 * sin^2(π Δx_m / p_m)) * K
		// dK/d(p_m)        = + 2 * q_m * (π Δx_m / p_m^2) * sin(2π Δx_m / p_m) * K
		inline void ker_periodic_ard_grad(
			Eigen::MatrixXd& grad_out,
			const Eigen::MatrixXd& sample_points,
			const Eigen::VectorXd& params,
			size_t param_index)
		{
			using Eigen::Index;

			const Index nsamples = sample_points.rows();
			const Index ndims = sample_points.cols();

			grad_out.setZero(nsamples, nsamples);

			// Expect params.size() == 1 + 2*ndims
			if (params.size() != (1 + 2 * ndims)) {
				return; // size mismatch
			}

			if (param_index >= static_cast<size_t>(1 + 2 * ndims)) {
				return; // out of range
			}

			const double sigma_f_sq = params(0);
			const Eigen::VectorXd q = params.segment(1, ndims);
			const Eigen::VectorXd p = params.segment(1 + ndims, ndims);

			constexpr double PI = 3.141592653589793238462643383279502884;
			const bool is_sigma = (param_index == 0);
			const bool is_q = (param_index > 0 && param_index <= static_cast<size_t>(ndims));
			const bool is_p = (param_index > static_cast<size_t>(ndims) && param_index < static_cast<size_t>(1 + 2 * ndims));

			const Index m = is_q ? static_cast<Index>(param_index - 1)
				: (is_p ? static_cast<Index>(param_index - 1 - ndims) : 0);

			for (Index i = 0; i < nsamples; ++i) {
				for (Index j = i; j < nsamples; ++j) {
					// Compute E_ij and K_ij
					double E = 0.0;
					for (Index d = 0; d < ndims; ++d) {
						const double pd = p(d);
						const double diff = sample_points(i, d) - sample_points(j, d);
						const double arg = PI * diff / pd;
						const double s = std::sin(arg);
						E += 2.0 * q(d) * s * s;
					}
					const double exp_term = std::exp(-E);
					const double Kij = sigma_f_sq * exp_term;

					double deriv = 0.0;
					if (is_sigma) {
						// dK/d(sigma_f_sq) = exp(-E)
						deriv = exp_term;
					}
					else if (is_q) {
						// dK/d(q_m) = - (2 * sin^2(π Δx_m / p_m)) * K
						const double diff_m = sample_points(i, m) - sample_points(j, m);
						const double arg_m = PI * diff_m / p(m);
						const double s_m = std::sin(arg_m);
						deriv = -(2.0 * s_m * s_m) * Kij;
					}
					else if (is_p) {
						// dK/d(p_m) = + 2 * q_m * (π Δx_m / p_m^2) * sin(2π Δx_m / p_m) * K
						const double diff_m = sample_points(i, m) - sample_points(j, m);
						const double pm = p(m);
						const double inv_pm = 1.0 / pm;
						const double arg_m = PI * diff_m * inv_pm;
						const double sin2 = std::sin(2.0 * arg_m);
						deriv = 2.0 * q(m) * (PI * diff_m) * (inv_pm * inv_pm) * sin2 * Kij;
					}

					grad_out(i, j) = deriv;
					grad_out(j, i) = deriv; // symmetry
				}
			}
		}

		// ADDITIVE KERNEL: RBF-ARD + PERIODIC-ARD
// params_sum = [ RBF_block | PERIODIC_block ]
//   RBF_block      = [ sigma_f_sq_rbf, q_rbf_1..q_rbf_D ]                  => size_rbf = 1 + D
//   PERIODIC_block = [ sigma_f_sq_per, q_per_1..q_per_D, p_1..p_D ]        => size_per = 1 + 2D
// total size = 2 + 3D
		template <std::ranges::random_access_range RangeParams,
			std::ranges::random_access_range RangePointA,
			std::ranges::random_access_range RangePointB>
		double ker_sum_rbf_plus_periodic_ard(const RangeParams& params, const RangePointA& point_a, const RangePointB& point_b) {
			const size_t D = point_a.size();

			const size_t size_rbf = 1 + D;
			const size_t size_per = 1 + 2 * D;
			const size_t total = size_rbf + size_per;

			if (static_cast<size_t>(std::ranges::distance(params.begin(), params.end())) != total) {
				// Size mismatch; return 0 as a safe default (or assert in your environment)
				return 0.0;
			}

			// First block: RBF-ARD
			const double k_rbf = ker_rbf_ard(
				std::ranges::subrange(params.begin(), params.begin() + static_cast<std::ptrdiff_t>(size_rbf)),
				point_a, point_b
			);

			// Second block: PERIODIC-ARD (with per-dim periods)
			const double k_per = ker_periodic_ard(
				std::ranges::subrange(params.begin() + static_cast<std::ptrdiff_t>(size_rbf),
					params.begin() + static_cast<std::ptrdiff_t>(total)),
				point_a, point_b
			);

			return k_rbf + k_per;
		}

		// Gradient of the additive kernel. By linearity and independence of parameter blocks,
		// only one block contributes for any given param_index.
		// We REUSE the existing gradient functions to compute the block gradient.
		inline void ker_sum_rbf_plus_periodic_ard_grad(
			Eigen::MatrixXd& grad_out,
			const Eigen::MatrixXd& sample_points,
			const Eigen::VectorXd& params,
			size_t param_index)
		{
			using Eigen::Index;

			const Index nsamples = sample_points.rows();
			const Index ndims = sample_points.cols();

			grad_out.setZero(nsamples, nsamples);

			const size_t size_rbf = static_cast<size_t>(1 + ndims);
			const size_t size_per = static_cast<size_t>(1 + 2 * ndims);
			const size_t total = size_rbf + size_per;

			if (params.size() != static_cast<Eigen::Index>(total)) {
				return; // size mismatch
			}
			if (param_index >= total) {
				return; // out of range
			}

			if (param_index < size_rbf) {
				// RBF block: pass only its parameters and the same param_index
				Eigen::VectorXd params_rbf(static_cast<Eigen::Index>(size_rbf));
				params_rbf = params.segment(0, static_cast<Eigen::Index>(size_rbf));
				ker_rbf_ard_grad(grad_out, sample_points, params_rbf, param_index);
			}
			else {
				// PERIODIC block: pass only its parameters and local index
				const size_t local_idx = param_index - size_rbf;
				Eigen::VectorXd params_per(static_cast<Eigen::Index>(size_per));
				params_per = params.segment(static_cast<Eigen::Index>(size_rbf),
					static_cast<Eigen::Index>(size_per));
				ker_periodic_ard_grad(grad_out, sample_points, params_per, local_idx);
			}
		}

		// Default initializers for kernel parameter vectors (linear space).
// Conventions (match your kernels):
// - RBF-ARD params:                [ sigma_f_sq, q_1, ..., q_D ]                 (size = 1 + D)
// - Periodic-ARD params:           [ sigma_f_sq, q_1, ..., q_D, p_1, ..., p_D ]  (size = 1 + 2D)
// - Sum (RBF + Periodic) params:   [ RBF_block | PERIODIC_block ]                (size = (1 + D) + (1 + 2D) = 2 + 3D)
//
// These functions return the initialized parameter vectors by value.

// -------------------------
// Simple ndims-only defaults
// -------------------------

// RBF-ARD: params = [ sigma_f_sq, q_1..q_D ]
		inline Eigen::VectorXd ker_rbf_ard_init(Eigen::Index ndims)
		{
			const Eigen::Index P = 1 + ndims;
			Eigen::VectorXd params(P);
			// sigma_f_sq
			params(0) = 1.0;
			// q_d = l_d^{-2}
			if (ndims > 0) {
				params.segment(1, ndims).setOnes();
			}
			return params;
		}

		// Periodic-ARD: params = [ sigma_f_sq, q_1..q_D, p_1..p_D ]
		inline Eigen::VectorXd ker_periodic_ard_init(Eigen::Index ndims)
		{
			const Eigen::Index P = 1 + 2 * ndims;
			Eigen::VectorXd params(P);
			// sigma_f_sq
			params(0) = 1.0;
			// q_d block
			if (ndims > 0) {
				params.segment(1, ndims).setOnes();
				// p_d block (unit periods by default)
				params.segment(1 + ndims, ndims).setOnes();
			}
			return params;
		}

		// Sum kernel (RBF-ARD + Periodic-ARD):
		// params_sum = [ sigma_f_sq_rbf, q_rbf_1..q_rbf_D, sigma_f_sq_per, q_per_1..q_per_D, p_1..p_D ]
		inline Eigen::VectorXd ker_sum_rbf_plus_periodic_ard_init(Eigen::Index ndims)
		{
			const Eigen::VectorXd rbf = ker_rbf_ard_init(ndims);
			const Eigen::VectorXd per = ker_periodic_ard_init(ndims);

			Eigen::VectorXd params(rbf.size() + per.size());
			params << rbf, per;
			return params;
		}

		// ------------------------------------
		// Data-driven initializers (recommended)
		// ------------------------------------

		// Helper to compute variance safely
		inline double variance_or(const Eigen::VectorXd& y, double fallback)
		{
			if (y.size() <= 1) return fallback;
			const double mean = y.mean();
			const Eigen::ArrayXd z = y.array() - mean;
			const double var = (z.square().sum()) / static_cast<double>(y.size() - 1);
			return std::max(var, fallback);
		}

		// RBF-ARD from data.
		// Heuristic: l_d = frac_of_range * (max(X_d) - min(X_d)), q_d = 1 / l_d^2.
		// sigma_f_sq = var(y) if provided, else 1.
		// Clamps:
		//   min_range -> to avoid zero lengths on constant dims,
		//   l_min/l_max -> keep length scales within practical bounds.
		inline Eigen::VectorXd ker_rbf_ard_init_from_data(
			const Eigen::MatrixXd& X,
			const Eigen::VectorXd* y_opt = nullptr,
			double frac_of_range = 0.25,
			double min_range = 1e-6,
			double l_min = 1e-3,
			double l_max = 1e+3)
		{
			const Eigen::Index D = X.cols();
			Eigen::VectorXd params(1 + D);

			// sigma_f_sq
			const double sigma_f_sq = (y_opt ? variance_or(*y_opt, 1.0) : 1.0);
			params(0) = sigma_f_sq;

			// q_d = 1 / l_d^2 with l_d based on per-dim range
			for (Eigen::Index d = 0; d < D; ++d) {
				const double xmin = X.col(d).minCoeff();
				const double xmax = X.col(d).maxCoeff();
				const double range = std::max(xmax - xmin, min_range);
				double l = std::clamp(frac_of_range * range, l_min, l_max);
				params(1 + d) = 1.0 / (l * l); // q_d
			}
			return params;
		}

		// Periodic-ARD from data.
		// Model: K = sigma_f_sq * exp( - Σ_d 2 q_d sin^2(π Δx_d / p_d) )
		// Heuristic:
		//   p_d ≈ range(X_d) (or 1 for degenerate dims)
		//   Choose an effective local length l_eff,d = frac_of_range * range(X_d)
		//   Small-angle approx: 2 q_d (π / p_d)^2 ≈ 1 / l_eff,d^2  =>  q_d = p_d^2 / (2 π^2 l_eff,d^2)
		inline Eigen::VectorXd ker_periodic_ard_init_from_data(
			const Eigen::MatrixXd& X,
			const Eigen::VectorXd* y_opt = nullptr,
			double frac_of_range = 0.25,
			double min_range = 1e-6,
			double p_min = 1e-3,
			double p_max = 1e+6,
			double l_eff_min = 1e-3,
			double l_eff_max = 1e+3)
		{
			const Eigen::Index D = X.cols();
			Eigen::VectorXd params(1 + 2 * D);

			const double sigma_f_sq = (y_opt ? variance_or(*y_opt, 1.0) : 1.0);
			params(0) = sigma_f_sq;

			constexpr double PI = 3.141592653589793238462643383279502884;

			for (Eigen::Index d = 0; d < D; ++d) {
				const double xmin = X.col(d).minCoeff();
				const double xmax = X.col(d).maxCoeff();
				const double range = std::max(xmax - xmin, min_range);

				// Period guess
				const double p = std::clamp(range, p_min, p_max);
				params(1 + d) = 0.0; // placeholder; filled after q computation
				params(1 + D + d) = p;

				// Effective local length and q via small-angle approximation
				const double l_eff = std::clamp(frac_of_range * range, l_eff_min, l_eff_max);
				const double denom = 2.0 * PI * PI * l_eff * l_eff; // 2 π^2 l_eff^2
				const double q = (p * p) / std::max(denom, 1e-30);
				params(1 + d) = q;
			}
			return params;
		}

		// Additive (RBF + Periodic) from data: concatenates the component initializations.
		inline Eigen::VectorXd ker_sum_rbf_plus_periodic_ard_init_from_data(
			const Eigen::MatrixXd& X,
			const Eigen::VectorXd* y_opt = nullptr,
			double frac_of_range_rbf = 0.25,
			double frac_of_range_per = 0.25)
		{
			const Eigen::VectorXd rbf = ker_rbf_ard_init_from_data(X, y_opt, frac_of_range_rbf);
			const Eigen::VectorXd per = ker_periodic_ard_init_from_data(X, y_opt, frac_of_range_per);

			Eigen::VectorXd params(rbf.size() + per.size());
			params << rbf, per;
			return params;
		}
		
	}


	struct GaussianProcess;
	using GP = GaussianProcess;

	//template <typename KernelFuncT>
	//requires std::is_invocable_r_v<double, KernelFuncT, const Eigen::VectorXd&, const Eigen::RowVectorXd&, const Eigen::RowVectorXd&>
	struct GaussianProcess {
		size_t ndims = 1;
		size_t nsamples = 0;
		size_t npredict = 0;
		Eigen::MatrixX<double> X_sample;
		Eigen::MatrixX<double> Y_sample;
		Eigen::MatrixX<double> Y_sample_prior_mue;
		Eigen::MatrixX<double> X_predict;
		Eigen::MatrixX<double> Y_predict_prior_mue;
		Eigen::MatrixX<double> Y_predict_posterior_mue;
		Eigen::MatrixX<double> Y_predict_posterior_covar;
		Eigen::MatrixX<double> Kxx;
		Eigen::MatrixX<double> Ksx;
		Eigen::MatrixX<double> Kss;
		Eigen::VectorX<double> kernel_params;
		std::function<double(const Eigen::VectorXd&, const Eigen::RowVectorXd&, const Eigen::RowVectorXd&)> kernel_func;
		//std::function<void(Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::VectorXd&, size_t)> kernel_grad_func;
		gp::kernel::RbfArdParams<Eigen::VectorX<double>> rbf_params;
		double var_noise = 1.0e-12;
		double jitter = 1.0e-8;
		Eigen::LLT<Eigen::MatrixX<double>> llt;
		Eigen::MatrixX<double> a;
		Eigen::MatrixX<double> b;
		Eigen::VectorX<double> Y_predict_score;
		Eigen::RowVectorX<double> col_dots;
		Eigen::VectorX<double> norms_sq;
		size_t best_score_idx = 0;
		double global_prior_sample_mean = 0.5;
	};

	inline size_t gp_nsamples(const GP& gp) {
		return gp.X_sample.rows();
	}

	inline size_t gp_npredict(const GP& gp) {
		return gp.X_predict.rows();
	}

	inline size_t gp_ndims(const GP& gp) {
		return gp.ndims;
	}

	inline Eigen::RowVectorXd gp_point_argmax_reduce_IMSE(const GP& gp) {
		return gp.X_predict.row(gp.best_score_idx);
	}

	inline double gp_get_sample_mean(const gp::GP& gp) {
		if (0 >= gp.Y_sample.rows()) {
			return 0.5;
		}
		return gp.Y_sample.mean();
	}

	GP gp_create(size_t ndims) {
		GP gp;
		gp.ndims = ndims;

		gp.rbf_params = gp::kernel::RbfArdParams<Eigen::VectorX<double>>();
		gp.rbf_params.length_scale_invsq.resize(ndims);
		gp.rbf_params.length_scale_invsq.setConstant(0.25);
		gp.rbf_params.signal_variance_sq = 0.0625;

		gp.kernel_func = gp::kernel::ker_rbf_ard<const Eigen::VectorXd&, const Eigen::RowVectorXd&, const Eigen::RowVectorXd&>;
		//gp.kernel_grad_func = gp::kernel::ker_rbf_ard_grad;
		gp.kernel_params = Eigen::VectorXd(1 + ndims);
		gp.kernel_params.setConstant(1.0);

		return gp;
	}
	
	void gp_add_predict_points(
		GP& gp,
		const Eigen::MatrixXd& new_points
	) {
		using Eigen::MatrixXd;
		using Eigen::VectorXd;
		using std::size_t;

		const size_t new_pts = new_points.rows();
		const size_t ndims = gp.ndims;

		if (new_pts == 0) return;
		if (new_points.cols() != ndims) {
			skx::io::log(skx::io::getDefaultErrorLoggerRef(),
				std::string("gp_add_predict_points(): dimension mismatch\n"));
			return;
		}

		const size_t old_npredict = gp.npredict;
		const size_t npredict_new = old_npredict + new_pts;
		const size_t nsamples = gp.nsamples; // current number of samples

		gp.X_predict.conservativeResize(npredict_new, gp_ndims(gp));
		gp.X_predict.bottomRows(new_points.rows()) = new_points;

		gp.npredict = npredict_new;

		gp.Y_predict_prior_mue.conservativeResize(npredict_new, 1);
		gp.Y_predict_prior_mue.bottomRows(new_points.rows()).setConstant(gp_get_sample_mean(gp));

		gp.Ksx.conservativeResize(npredict_new, gp_nsamples(gp));
		gp.Kss.conservativeResize(npredict_new, npredict_new);

		for (size_t i = old_npredict; i < npredict_new; i++) {
			for (size_t j = 0; j < gp_nsamples(gp); j++) {
				double kval = gp.kernel_func(
					gp.kernel_params,
					gp.X_predict.row(i),
					gp.X_sample.row(j)
				);
				gp.Ksx(i, j) = kval;
			}
		}

		for (size_t i = 0; i < npredict_new; ++i) {
			size_t jstart = std::max(i, old_npredict);
			for (size_t j = jstart; j < npredict_new; ++j) {
				double kval = gp.kernel_func(
					gp.kernel_params,
					gp.X_predict.row(i),
					gp.X_predict.row(j)
				);
				gp.Kss(i, j) = kval;
				gp.Kss(j, i) = kval;
			}
		}

		// --- Recompute posterior-related quantities using existing LLT / a ---
		// b = Kxx^{-1} * Ksx^T  (uses current LLT that factors the Kxx of samples)
		if (0 < gp_nsamples(gp)) {
			// Ksx.transpose() shape: (nsamples x npredict_new) => matches LLT rows
			gp.b = gp.llt.solve(gp.Ksx.transpose());
		}
		else {
			gp.b.resize(0, 0);
		}

		// posterior mean & covariance
		gp.Y_predict_posterior_mue = gp.Y_predict_prior_mue + gp.Ksx * gp.a;
		gp.Y_predict_posterior_covar = gp.Kss - gp.Ksx * gp.b;

		gp.Y_predict_score = gp.Kss.diagonal();
		gp.Y_predict_score.array() += gp.var_noise;
		gp.col_dots = (gp.Ksx.transpose().array() * gp.b.array()).colwise().sum();
		gp.Y_predict_score.array() += -gp.col_dots.transpose().array();

		gp.norms_sq.resize(npredict_new);
		for (size_t i = 0; i < npredict_new; ++i)
			gp.norms_sq(i) = gp.Y_predict_posterior_covar.col(i).squaredNorm();

		gp.Y_predict_score = gp.norms_sq.array() / gp.Y_predict_score.array();
		gp.Y_predict_score.maxCoeff(&gp.best_score_idx);
	}

	void gp_clear_predict_points(GP& gp) {
		gp.npredict = 0;
		gp.X_predict.resize(0, gp.ndims);
		gp.Y_predict_prior_mue.resize(0, 0);
		gp.Y_predict_posterior_mue.resize(0, 0);
		gp.Y_predict_posterior_covar.resize(0, 0);
		gp.Ksx.resize(0, 0);
		gp.Kss.resize(0, 0);
		gp.Y_predict_score.resize(0);
		gp.norms_sq.resize(0);
		gp.col_dots.resize(0);
		gp.best_score_idx = 0;
	}

	void gp_compute_posterior(GP& gp)
	{
		using Eigen::MatrixXd;
		using Eigen::VectorXd;

		if (gp.nsamples == 0 || gp.npredict == 0)
			return;

		// b = L^{-1} (L^{-T} Ksx^T) = Kxx^{-1} Ksx^T
		gp.b = gp.llt.solve(gp.Ksx.transpose());

		// Posterior mean: m(X_*) + Ksx * a
		gp.Y_predict_posterior_mue =
			gp.Y_predict_prior_mue + gp.Ksx * gp.a;

		// Posterior covariance: Kss - Ksx * b
		gp.Y_predict_posterior_covar =
			gp.Kss - gp.Ksx * gp.b;

		// Optional: compute variance and “scores” (if you’re doing acquisition)
		gp.Y_predict_score = gp.Y_predict_posterior_covar.diagonal();

		// Add noise variance if necessary
		gp.Y_predict_score.array() += gp.var_noise;

		// Keep track of best score (if used for active learning)
		gp.Y_predict_score.maxCoeff(&gp.best_score_idx);
	}

	void gp_query_posterior(const GP& gp, size_t idx,
		double& mue_out, double& var_out)
	{
		if (idx >= gp.npredict) {
			skx::io::log(skx::io::getDefaultErrorLoggerRef(),
				std::string("gp_query_posterior(): index out of range\n"));
			mue_out = var_out = std::numeric_limits<double>::quiet_NaN();
			return;
		}

		mue_out = gp.Y_predict_posterior_mue(idx);
		var_out = gp.Y_predict_posterior_covar(idx, idx);
	}

	Eigen::VectorXd gp_get_best_posterior_score_point(gp::GP& gp) {
		if (gp.npredict == 0) {
			skx::io::log(skx::io::getDefaultErrorLoggerRef(),
				std::string("No prediction points set in gp_get_best_posterior_score_point()\n"));
			return Eigen::VectorXd();
		}

		// Recompute the scoring if not up-to-date
		gp.Y_predict_score = gp.Kss.diagonal();
		gp.Y_predict_score.array() += gp.var_noise;

		if (gp.Ksx.size() && gp.b.size()) {
			gp.col_dots = (gp.Ksx.transpose().array() * gp.b.array()).colwise().sum();
			gp.Y_predict_score.array() -= gp.col_dots.transpose().array();
		}

		gp.norms_sq.resize(gp.npredict);
		for (size_t i = 0; i < gp.npredict; ++i)
			gp.norms_sq(i) = gp.Y_predict_posterior_covar.col(i).squaredNorm();

		gp.Y_predict_score = gp.norms_sq.array() / gp.Y_predict_score.array();

		// Find the index of the maximum score
		gp.Y_predict_score.maxCoeff(&gp.best_score_idx);
		size_t idx = gp.best_score_idx;

		// Return the coordinates of the next best sample point
		return gp.X_predict.row(idx);
	}

	double gp_reset_global_prior_mean(gp::GP& gp) {
		using Eigen::MatrixXd;

		if (gp.nsamples == 0) {
			skx::io::log(skx::io::getDefaultErrorLoggerRef(),
				std::string("No samples in GP to compute global prior.\n"));
			return gp_get_sample_mean(gp);
		}

		// Compute new global prior from observed targets
		gp.global_prior_sample_mean = gp.Y_sample.mean();
		const double global_prior = gp.global_prior_sample_mean;

		// Update priors for training outputs
		//gp.Y_sample_prior_mue.setConstant(global_prior);

		// Update priors for prediction outputs (if any)
		//if (gp.npredict > 0 && gp.Y_predict_prior_mue.size() > 0) {
		//	gp.Y_predict_prior_mue.setConstant(global_prior);
		//}

		// Recompute a with updated training prior mean
		// (Kxx + (sigma^2 + jitter) * I) * a = (Y_sample - Y_sample_prior_mue)
		//gp.a = gp.llt.solve(gp.Y_sample - gp.Y_sample_prior_mue);

		//if (gp.npredict > 0) {
		//	// Recompute b using the existing LLT factorization
		//	gp.b = gp.llt.solve(gp.Ksx.transpose());

		//	// Posterior mean and covariance with updated priors
		//	gp.Y_predict_posterior_mue = gp.Y_predict_prior_mue + gp.Ksx * gp.a;
		//	gp.Y_predict_posterior_covar = gp.Kss - gp.Ksx * gp.b;

		//	// Optionally recompute your acquisition score
		//	gp.Y_predict_score = gp.Kss.diagonal();
		//	gp.Y_predict_score.array() += gp.var_noise;

		//	gp.col_dots = (gp.Ksx.transpose().array() * gp.b.array()).colwise().sum();
		//	gp.Y_predict_score.array() -= gp.col_dots.transpose().array();

		//	gp.norms_sq.resize(gp.npredict);
		//	for (size_t i = 0; i < gp.npredict; ++i) {
		//		gp.norms_sq(i) = gp.Y_predict_posterior_covar.col(i).squaredNorm();
		//	}
		//	gp.Y_predict_score = gp.norms_sq.array() / gp.Y_predict_score.array();
		//	gp.Y_predict_score.maxCoeff(&gp.best_score_idx);
		//}

		// // Global prior mean reset to: (converted logs only; original cout left commented)
		return global_prior;
	}
	
	inline double gp_calc_expectation(const gp::GP& gp, const Eigen::RowVectorXd& point) {
		const double m_prior = gp_get_sample_mean(gp);

		if (0 == gp_nsamples(gp)) {
			return m_prior;
		}

		Eigen::VectorXd k(gp_nsamples(gp));
		for (size_t i = 0; i < gp_nsamples(gp); i++) {
			k(i) = gp.kernel_func(gp.kernel_params, point, gp.X_sample.row(i));
		}

		const Eigen::VectorXd a_vec = gp.a.col(0);

		return m_prior + k.dot(a_vec);
	}

	void gp_on_update_kernel(GP& gp) {

		// update Kxx
		if (0 < gp_nsamples(gp)) {
			for (size_t i = 0; i < gp_nsamples(gp); i++) {
				for (size_t j = i; j < gp_nsamples(gp); j++) {
					double kval = gp.kernel_func(
						gp.kernel_params,
						gp.X_sample.row(i),
						gp.X_sample.row(j)
					);
					gp.Kxx(i, j) = kval;
					gp.Kxx(j, i) = kval;
				}
			}
		}

		// update Kss
		if (0 < gp_npredict(gp)) {
			for (size_t i = 0; i < gp_npredict(gp); i++) {
				for (size_t j = i; j < gp_npredict(gp); j++) {
					double kval = gp.kernel_func(
						gp.kernel_params,
						gp.X_predict.row(i),
						gp.X_predict.row(j)
					);
					gp.Kss(i, j) = kval;
					gp.Kss(j, i) = kval;
				}
			}
		}

		// update Ksx
		if (0 < gp_npredict(gp) && 0 < gp_nsamples(gp)) {
			for (size_t i = 0; i < gp_npredict(gp); i++) {
				for (size_t j = 0; j < gp_nsamples(gp); j++) {	
					double kval = gp.kernel_func(
						gp.kernel_params,
						gp.X_predict.row(i),
						gp.X_sample.row(j)
					);
					gp.Ksx(i, j) = kval;
				}
			}
		}

		// ---- Recompute LLT (on updated Kxx) ----
		double jitter = gp.jitter;
		constexpr const size_t llt_ix_max = 8;
		size_t llt_ix = 0;
		do {
			gp.llt.compute(gp.Kxx + (gp.var_noise + jitter) *
				Eigen::MatrixXd::Identity(gp_nsamples(gp), gp_nsamples(gp)));
			jitter *= 1.54321;
			llt_ix++;
		} while (gp.llt.info() != Eigen::Success && llt_ix < llt_ix_max);
		if (llt_ix >= llt_ix_max && gp.llt.info() != Eigen::Success) {
			skx::io::log(skx::io::getDefaultErrorLoggerRef(),
				std::string("llt failed during gp_on_update_kernel()! Try increasing jitter. Too many iterations reached.\n"));
		}

		if (0 < gp_npredict(gp)) {

			// Now Ksx.transpose() has shape (nsamples x npredict) matching LLT's rows
			gp.b = gp.llt.solve(gp.Ksx.transpose()); // OK: dimensions match

			// Posterior mean and covariance
			gp.Y_predict_posterior_mue = gp.Y_predict_prior_mue + gp.Ksx * gp.a;
			gp.Y_predict_posterior_covar = gp.Kss - gp.Ksx * gp.b;

			// Update score and best index
			gp.Y_predict_score = gp.Kss.diagonal();
			gp.Y_predict_score.array() += gp.var_noise;

			gp.col_dots = (gp.Ksx.transpose().array() * gp.b.array()).colwise().sum();
			gp.Y_predict_score.array() += -gp.col_dots.transpose().array();

			gp.norms_sq.resize(gp.npredict);
			for (size_t i = 0; i < gp.npredict; ++i) {
				gp.norms_sq(i) = gp.Y_predict_posterior_covar.col(i).squaredNorm();
			}
			gp.Y_predict_score = gp.norms_sq.array() / gp.Y_predict_score.array();
			gp.Y_predict_score.maxCoeff(&gp.best_score_idx);
		}
	}

	void gp_add_sample(GP& gp, Eigen::RowVectorXd point, double value) {
		using Eigen::MatrixXd;

		double expectation = gp_get_sample_mean(gp);

		const size_t old_nsamples = gp_nsamples(gp);
		const size_t new_nsamples = old_nsamples + 1;
		const size_t ndims = gp_ndims(gp);

		gp.X_sample.conservativeResize(new_nsamples, gp_ndims(gp));
		gp.X_sample.row(new_nsamples - 1) = point;

		gp.Y_sample.conservativeResize(new_nsamples, 1);
		gp.Y_sample(new_nsamples - 1, 0) = value;

		gp.Y_sample_prior_mue.conservativeResize(new_nsamples, 1);
		//gp.Y_sample_prior_mue(new_nsamples - 1, 0) = expectation;
		gp.Y_sample_prior_mue.setConstant(gp_get_sample_mean(gp));

		if (0 < gp_npredict(gp)) {
			gp.Y_predict_prior_mue.setConstant(gp_get_sample_mean(gp));
		}

		gp.Kxx.conservativeResize(new_nsamples, new_nsamples);

		for (size_t i = 0; i < new_nsamples; i++) {
			double kval = gp.kernel_func(
				gp.kernel_params,
				gp.X_sample.row(i),
				gp.X_sample.row(new_nsamples - 1)
			);
			gp.Kxx(i, new_nsamples - 1) = kval;
			gp.Kxx(new_nsamples - 1, i) = kval;
		}

		gp.Ksx.conservativeResize(gp_npredict(gp), new_nsamples);
		if (0 < gp_npredict(gp)) {
			for (size_t i = 0; i < gp_npredict(gp); i++) {
				double kval = gp.kernel_func(
					gp.kernel_params,
					gp.X_predict.row(i),
					gp.X_sample.row(new_nsamples - 1)
				);
				gp.Ksx(i, new_nsamples - 1) = kval;
			}
		}

		gp.nsamples = new_nsamples;

		// ---- Recompute LLT (on updated Kxx) ----
		double jitter = gp.jitter;
		constexpr const size_t llt_ix_max = 8;
		size_t llt_ix = 0;
		do {
			gp.llt.compute(gp.Kxx + (gp.var_noise + jitter) *
				MatrixXd::Identity(new_nsamples, new_nsamples));
			jitter *= 2.54829482948;
			llt_ix++;
		} while (gp.llt.info() != Eigen::Success && llt_ix < llt_ix_max);
		if (llt_ix >= llt_ix_max && gp.llt.info() != Eigen::Success) {
			skx::io::log(skx::io::getDefaultErrorLoggerRef(),
				std::string("llt failed during gp_add_sample()! Try increasing jitter. Too many iterations reached.\n"));
		}

		// ---- Recompute 'a' (always depends on samples) ----
		gp.a = gp.llt.solve(gp.Y_sample - gp.Y_sample_prior_mue);

		// ---- If we have prediction points, compute b and posterior
		if (0 < gp_npredict(gp)) {

			// Now Ksx.transpose() has shape (nsamples x npredict) matching LLT's rows
			gp.b = gp.llt.solve(gp.Ksx.transpose()); // OK: dimensions match

			// Posterior mean and covariance
			gp.Y_predict_posterior_mue = gp.Y_predict_prior_mue + gp.Ksx * gp.a;
			gp.Y_predict_posterior_covar = gp.Kss - gp.Ksx * gp.b;

			// Update score and best index
			gp.Y_predict_score = gp.Kss.diagonal();
			gp.Y_predict_score.array() += gp.var_noise;

			gp.col_dots = (gp.Ksx.transpose().array() * gp.b.array()).colwise().sum();
			gp.Y_predict_score.array() += -gp.col_dots.transpose().array();

			gp.norms_sq.resize(gp.npredict);
			for (size_t i = 0; i < gp.npredict; ++i) {
				gp.norms_sq(i) = gp.Y_predict_posterior_covar.col(i).squaredNorm();
			}
			gp.Y_predict_score = gp.norms_sq.array() / gp.Y_predict_score.array();
			gp.Y_predict_score.maxCoeff(&gp.best_score_idx);
		}

		//if (0){
		//	std::ostringstream oss;
		//	oss << "[gp_add_sample]" << " point[ ";
		//	for (size_t d = 0; d < point.rows(); d++) {
		//		oss << point(d) << " ";
		//	}
		//	oss << "], value[ " << value
		//		<< " ], pre-expectation[ " << expectation
		//		<< " ], post-expectation[ " << gp_calc_expectation(gp, point)
		//		<< " ]\n";
		//	skx::io::log(skx::io::getDefaultLoggerRef(), oss.str());
		//}

	}

	// Enumerate all integer lattice points within bounds into a matrix (P x D).
	inline Eigen::MatrixXd enumerate_all_integer_points(const Eigen::MatrixXd& bounds) {
		const int D = bounds.rows();
		// Build per-dimension integer levels
		std::vector<std::vector<long>> levels(D);
		std::size_t total = 1;
		for (int d = 0; d < D; ++d) {
			const long lo = static_cast<long>(std::ceil(bounds(d, 0)));
			const long hi = static_cast<long>(std::floor(bounds(d, 1)));
			if (hi < lo) {
				return Eigen::MatrixXd(0, D);
			}
			const std::size_t Ld = static_cast<std::size_t>(hi - lo + 1);
			levels[d].reserve(Ld);
			for (long v = lo; v <= hi; ++v) levels[d].push_back(v);

			if (Ld != 0 && total > std::numeric_limits<std::size_t>::max() / Ld) {
				// Too many to enumerate safely.
				return Eigen::MatrixXd(0, D);
			}
			total *= Ld;
		}

		const int P = static_cast<int>(total);
		Eigen::MatrixXd X(P, D);

		// Mixed-radix enumeration
		std::vector<std::size_t> idx(D, 0);
		for (int i = 0; i < P; ++i) {
			for (int d = 0; d < D; ++d) {
				X(i, d) = static_cast<double>(levels[d][idx[d]]);
			}
			// Increment counters
			for (int d = D - 1; d >= 0; --d) {
				idx[d]++;
				if (idx[d] < levels[d].size()) break;
				idx[d] = 0;
			}
		}
		return X;
	}
}