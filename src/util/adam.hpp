
#pragma once


#include <cstddef>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <limits>
#include <functional>
#include <vector>

#include <Eigen/Dense>

namespace skx::algo::optim {

	using KernelFuncT = std::function<double(const Eigen::VectorXd&, const Eigen::RowVectorXd&, const Eigen::RowVectorXd&)>;
	using KernelGradFuncT = std::function<void(Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::VectorXd&, size_t)>;

	// chat rewrote this for me to support kernel functions in both log and linear space
	void calc_log_marginal_likelihood_and_grad(
		double& lml_out,
		Eigen::VectorXd& lml_grad_out,
		Eigen::VectorXd& params,
		bool params_is_logspace,
		const Eigen::MatrixXd& sample_points,
		const Eigen::VectorXd& sample_values,
		double sample_noise_var,
		KernelFuncT kernel_func,
		bool kernel_func_is_logspace,
		KernelGradFuncT kernel_grad_func,
		bool kernel_grad_func_is_logspace
	) {
		using Eigen::MatrixXd;
		using Eigen::VectorXd;
		using Eigen::Index;

		const Index n = sample_points.rows();
		if (n == 0) {
			lml_out = -std::numeric_limits<double>::infinity();
			lml_grad_out.setZero(params.size());
			return;
		}

		// Prepare parameter vectors in both spaces without mutating 'params'
		// params_linear: the positive (linear) parameterization expected by linear-space kernels
		// params_log:    the log-parameterization expected by log-space kernels
		const double eps_pos = 1e-50; // guard for log/reciprocal
		VectorXd params_linear = params_is_logspace ? params.array().exp().matrix() : params;
		VectorXd params_log;
		if (params_is_logspace) {
			params_log = params;
		}
		else {
			params_log.resize(params.size());
			for (Index i = 0; i < params.size(); ++i) {
				const double v = params(i);
				// If a linear parameter is non-positive, clamp for log; underlying kernel may still be invalid,
				// but this avoids NaNs here.
				params_log(i) = std::log(std::max(std::abs(v), eps_pos));
			}
		}

		// Choose which parameterization to pass into the kernel function
		const VectorXd& params_for_kernel = kernel_func_is_logspace ? params_log : params_linear;

		// Build K (symmetric)
		MatrixXd Kxx(n, n);
		for (Index i = 0; i < n; ++i) {
			for (Index j = i; j < n; ++j) {
				const double kij = kernel_func(params_for_kernel,
					sample_points.row(i),
					sample_points.row(j));
				Kxx(i, j) = kij;
				Kxx(j, i) = kij;
			}
		}

		// Add noise variance (and a small jitter for stability)
		const double jitter = 1e-8;
		Kxx.diagonal().array() += (sample_noise_var + jitter);

		// Cholesky factorization
		Eigen::LLT<MatrixXd> llt(Kxx);
		if (llt.info() != Eigen::Success) {
			lml_out = -std::numeric_limits<double>::infinity();
			lml_grad_out.setZero(params.size());
			return;
		}
		const MatrixXd& L = llt.matrixL();

		// alpha = Ky^{-1} y via triangular solves
		VectorXd alpha = L.transpose().template triangularView<Eigen::Upper>()
			.solve(L.template triangularView<Eigen::Lower>().solve(sample_values));

		// log |Ky| = 2 * sum(log diag(L))
		const double logdet = 2.0 * L.diagonal().array().log().sum();

		// Log marginal likelihood
		lml_out = -0.5 * sample_values.dot(alpha)
			- 0.5 * logdet
			- 0.5 * static_cast<double>(n) * std::log(2.0 * M_PI);

		// Ky^{-1} via solving against identity
		MatrixXd Ky_inv = llt.solve(MatrixXd::Identity(n, n));

		// Q = alpha alpha^T - Ky^{-1}
		MatrixXd Q = alpha * alpha.transpose() - Ky_inv;

		// Gradient
		lml_grad_out.setZero(params.size());

		// Decide param space for gradient kernel and chain-rule scaling
		const VectorXd& params_for_grad = kernel_grad_func_is_logspace ? params_log : params_linear;

		MatrixXd dK(n, n);
		for (Index i = 0; i < params.size(); ++i) {
			// dK/dθ_i in the kernel_grad_func's parameterization
			dK.setZero(n, n);
			kernel_grad_func(dK, sample_points, params_for_grad, static_cast<size_t>(i));

			// Base gradient component: dL/dθ_kernel = 0.5 * trace(Q * dK)
			double dL_dtheta_kernel = 0.5 * (Q.array() * dK.array()).sum();

			// Chain rule to convert to this function's 'params' parameterization
			// Cases:
			//  - If both spaces match: no scaling.
			//  - If params is linear and kernel_grad is log-space:
			//        dL/d(theta_linear) = dL/d(log theta) * d(log theta)/d(theta) = (1/theta_linear) * dL/d(log theta)
			//  - If params is log-space and kernel_grad is linear-space:
			//        dL/d(log theta) = dL/d(theta_linear) * d(theta)/d(log theta) = theta_linear * dL/d(theta)
			double chain = 1.0;
			if (params_is_logspace != kernel_grad_func_is_logspace) {
				const double theta_linear = params_linear(i);
				if (!params_is_logspace && kernel_grad_func_is_logspace) {
					// linear params, log-space gradient provided
					chain = 1.0 / std::max(std::abs(theta_linear), eps_pos);
				}
				else if (params_is_logspace && !kernel_grad_func_is_logspace) {
					// log-space params, linear-space gradient provided
					chain = theta_linear;
				}
			}

			lml_grad_out(static_cast<Eigen::Index>(i)) = dL_dtheta_kernel * chain;
		}
	}

	struct AdamConfig {
		size_t t_max = 32;
		double alpha = 0.001; // this is the learning rate (lr)
		double beta1 = 0.9;
		double beta2 = 0.999;
		double epsilon = 1e-8;
		double alpha_dropoff = 0.90123;
		size_t nmultistart = 8;
		bool params_is_logspace = false;
		bool kernel_grad_func_is_logspace = false;
		bool kernel_func_is_logspace = false;
	};

	// I just winged it with the multi start
	void optimise_params_adam(
		Eigen::VectorXd& params, 
		const Eigen::MatrixXd& sample_points, 
		const Eigen::VectorXd& sample_values, 
		double sample_noise_var, 
		KernelFuncT kernel_func,
		KernelGradFuncT kernel_grad_func,
		const AdamConfig& config
	) {
		const size_t& t_max = config.t_max;
		const double& alpha = config.alpha;
		const double& beta1 = config.beta1;
		const double& beta2 = config.beta2;
		const double& epsilon = config.epsilon;
		const size_t& nmultistart = config.nmultistart;

		std::mt19937 rng(123);
		std::normal_distribution<double> normal_z(0.0, 1.0);

		Eigen::VectorXd params_logspace = params;
		if (false == config.params_is_logspace) {
			for (size_t i = 0; i < params.size(); i++) {
				const double v = params(i);
				params_logspace(i) = std::log(std::max(std::abs(v), 1e-48));
			}
		}

		std::vector<Eigen::VectorXd> thetas;
		thetas.push_back(params_logspace);
		for (size_t i = 0; i < nmultistart - 1; i++) {
			thetas.push_back(params_logspace);
			double sc = 0.10000102313;
			if (0 == i) {
				sc = 1.0000001239124;
			}
			if (1 == i) {
				for (size_t d = 0; d < params.size(); d++) {
					thetas[thetas.size() - 1](d) += normal_z(rng);
				}
				continue;
			}
			for (size_t d = 0; d < params.size(); d++) {
				thetas[thetas.size() - 1](d) += sc * normal_z(rng) * thetas[thetas.size() - 1](d);
			}
		}

		double lml_logspace_best = -std::numeric_limits<double>::infinity();
		Eigen::VectorXd params_logspace_best(params);

		for (size_t multistart_ix = 0; multistart_ix < nmultistart; multistart_ix++) {
			Eigen::VectorXd params_logspace_grad = Eigen::VectorXd::Zero(params.size());
			params_logspace = thetas[multistart_ix];

			Eigen::VectorXd m = Eigen::VectorXd::Zero(params.size());
			Eigen::VectorXd v = Eigen::VectorXd::Zero(params.size());
			Eigen::VectorXd m_hat = Eigen::VectorXd::Zero(params.size());
			Eigen::VectorXd v_hat = Eigen::VectorXd::Zero(params.size());

			double alpha_local = alpha;

			for (size_t t = 1; t < t_max; t++) {
				double lml_logspace = -std::numeric_limits<double>::infinity();
				calc_log_marginal_likelihood_and_grad(
					lml_logspace, 
					params_logspace_grad, 
					params_logspace, 
					true,
					sample_points, 
					sample_values, 
					sample_noise_var, 
					kernel_func, 
					config.kernel_func_is_logspace,
					kernel_grad_func,
					config.kernel_grad_func_is_logspace
				);

				if (lml_logspace > lml_logspace_best && std::isfinite(lml_logspace)) {
					lml_logspace_best = lml_logspace;
					params_logspace_best = params_logspace;
				}

				m = beta1 * m + (1.0 - beta1) * params_logspace_grad;
				v = beta2 * v + (1.0 - beta2) * params_logspace_grad.array().square().matrix();

				m_hat = m / (1.0 - std::pow(beta1, t));
				v_hat = v / (1.0 - std::pow(beta2, t));

				params_logspace.array() += alpha_local * m_hat.array() / (v_hat.array().sqrt() + epsilon);
				alpha_local *= config.alpha_dropoff;
			}

		}

		params_logspace = params_logspace_best;
		if (false == config.params_is_logspace) {
			params = params_logspace.array().exp().matrix();
		}
	}

}; // namespace skx::algo::optim
