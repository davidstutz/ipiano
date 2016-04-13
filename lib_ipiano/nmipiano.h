/**
 * Implementation of nmiPiano as described in [1] and [2]:
 * 
 * [1] P. Ochs, Y. Chen, T. Brox, T. Pock.
 *     iPiano: Inertial Proximal Algorithm for Nonconvex Optimization
 *     SIAM Journal of Imaging Sciences, colume 7, number 2, 2014.
 * [2] D. Stutz.
 *     Seminar paper "iPiano: Inertial Proximal Algorithm for Non-Convex Optimization"
 *     https://github.com/davidstutz/seminar-ipiano
 * 
 * The code is published under the BSD 3-Clause:
 * 
 * Copyright (c) 2016, David Stutz
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef NMIPIANO_H
#define	NMIPIANO_H

#include <random>
#include <fstream>
#include <functional>
#include <Eigen/Dense>
#include <glog/logging.h>

/** \brief Implementation of nmiPiano (algorithm 4) as proposed in [1]:
 *  [1] P. Ochs, Y. Chen, T. Brox, T. Pock.
 *      iPiano: Inertial Proximal Algorithm for Nonconvex Optimization.
 *      SIAM J. Imaging Sciences, vol. 7, no. 2, 2014.
 * \author David Stutz
 */
class nmiPiano
{
public:
    
    /** \brief Options of algorithm. */
    struct Options
    {
        /** \brief Initial iterate. */
        Eigen::MatrixXf x_0;
        /** \brief Maximum number of iterations. */
        unsigned int max_iter;
        /** \brief Fixed beta in [0, 1). */
        float beta = 0.5f;
        /** \brief Fixed eta for backtracking the local lipschitz constant. */
        float eta = 1.05f;
        /** \brief Initialization of loca Lipschitz. */
        float L_0m1 = 1.f;
        /** \brief Whether to bound estimated Lipschitz constant below by the given L_n. */
        bool BOUND_L_N = false;
        /** \brief Termination criterion; stop if Delta_n smaller than epsilon. */
        float epsilon = 0;
    };
    
    /** \brief Structure representing an iteration, passed as is to a callback function
     * to be able to monitor process. */
    struct Iteration
    {
        /** \brief Current iterate. */
        Eigen::MatrixXf x_n;
        /** \brief Last iterate. */
        Eigen::MatrixXf x_nm1;
        /** \brief Update: x_np1 = x_n + Delta_x_n. */
        Eigen::MatrixXf Delta_x_n;
        /** \brief Difference between two iterates. */
        float Delta_n;
        /** \brief Smooth objective function at current iterate. */
        float f_x_n;
        /** \brief Derivative of smooth objective at current iterate. */
        Eigen::MatrixXf df_x_n;
        /** Nonsmooth objective function at current iterate. */
        float g_x_n;
        /** \brief L_n of current iterate (within iterations, this is also used to estimate L_np1 via backtracking). */
        float L_n;
        /** \brief alpha_n of current iterate (within iterations, this is also used to estimate alpha_np1). */
        float alpha_n;
        /** \brief Current iteration. */
        int n;
    };
    
    /** \brief Silent callback to use with nmiPiano.
     * \param[in] iteration
     */
    static void silent_callback(const Iteration &iteration);
    
    /** \brief Default callback to use with nmiPiano.
     * \param[in] iteration
     */
    static void default_callback(const Iteration &iteration, int n = 10);
    
    /** \brief Brief callback to use with nmiPiano.
     * \param[in] iteration
     */
    static void brief_callback(const Iteration &iteration, int n = 10);
    
    /** \brief File callback, used to write all information to the specified file.
     * \param[in] iteration
     * \param[in] file
     */
    static void file_callback(const Iteration &iteration, const std::string &file);
    
    /** \brief File callback for plotting, writes CSV output of format: iteration,f,g,alpha_n,L_n,Delta_n.
     * \param[in] iteration
     * \param[in] file
     */
    static void plot_callback(const Iteration &iteration, const std::string &file);
    
    /** \brief Constructor.
     * \param[in] f
     * \param[in] df
     * \param[in] prox_g
     * \param[in] callback
     */
    nmiPiano(const std::function<float(const Eigen::MatrixXf&)> f, 
            const std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&)> df,
            std::function<float(const Eigen::MatrixXf&)> g,
            const std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&, float alpha)> prox_g,
            Options options,
            const std::function<void(const Iteration&)> callback = [](const Iteration &iteration) -> void { /* Nothing ... */ });
            
    /** \brief Destructor. */
    ~nmiPiano();
    
    /** \brief Optimize the given objective using the nmiPiano algorithm. 
     * \param[out] x_star
     * \param[out] f_x_star
     */
    void optimize(Eigen::MatrixXf &x_star, float &f_x_star);
    
protected:
    
    /** \brief Initialize the iteration structure (i.e. set iteration 0): 
     * \param[in] iteration
     */
    void initialize(Iteration &iteration);
    
    float estimateL(const Iteration &iteration);
    
    /** \brief Do an iteration, i.e. compute x_np1.
     * \param[in] iteration
     * \param[in] beta
     * \param[out] x_np1
     */
    void iterate(Iteration &iteration, float beta, Eigen::MatrixXf &x_np1);
    
    /** \brief Check the lipschitz condition for backtracking.
     * \param[in] iteration
     * \param[in] x_np1
     * \return
     */
    bool checkLipschitz(const Iteration& iteration, const Eigen::MatrixXf &x_np1);
    
    /** \brief The smooth part of the objective function. */
    std::function<float(const Eigen::MatrixXf&)> f_;
    /** \brief Gradient of smooth part of the objective function (i.e. derivative). */
    std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&)> df_;
    /** \brief Convex, nonsmooth part of objective. */
    std::function<float(const Eigen::MatrixXf&)> g_;
    /** \brief Proximal map for g, i.e. (I + alpha_n*g)^(-1). */
    std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&, float)> prox_g_;
    /** \brief Callback, can e.g. be sued to monitor process using the provided information in the Iteration structure. */
    std::function<void(const Iteration&)> callback_;
    /** \brief Options.*/
    Options options_;
};

////////////////////////////////////////////////////////////////////////////////
// nmiPiano::silent_callback
////////////////////////////////////////////////////////////////////////////////

void nmiPiano::silent_callback(const nmiPiano::Iteration &iteration)
{
    
}

////////////////////////////////////////////////////////////////////////////////
// nmiPiano::default_callback
////////////////////////////////////////////////////////////////////////////////

void nmiPiano::default_callback(const nmiPiano::Iteration &iteration, int n)
{
    if (iteration.n%n == 0)
    {
        std::cout << "[" << iteration.n << "] " 
                << iteration.f_x_n + iteration.g_x_n
                << " (Delta_n = " << iteration.Delta_n << "; L_n = " << iteration.L_n 
                << "; alpha_n = " << iteration.alpha_n << ")"
                << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
// nmiPiano::brief_callback
////////////////////////////////////////////////////////////////////////////////

void nmiPiano::brief_callback(const nmiPiano::Iteration &iteration, int n)
{
    if (iteration.n%n == 0)
    {
        std::cout << "[" << iteration.n << "] " 
                << iteration.f_x_n + iteration.g_x_n
                << " (Delta_n = " << iteration.Delta_n << ")"
                << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
// nmiPiano::file_callback
////////////////////////////////////////////////////////////////////////////////

void nmiPiano::file_callback(const nmiPiano::Iteration &iteration, const std::string &file)
{
    std::ofstream out;
    out.open(file, std::ios_base::app);
    out << "iteration = " << iteration.n << "; f_x_n = " << iteration.f_x_n 
            << "; g_x_n = " << iteration.g_x_n << "; alpha_n = " << iteration.alpha_n
            << "; L_n = " << iteration.L_n
            << "; |df_x_n| = " << std::sqrt(iteration.df_x_n.squaredNorm()) 
            << "; Delta_n = " << iteration.Delta_n << std::endl;
    out.close();
}

////////////////////////////////////////////////////////////////////////////////
// nmiPiano::plot_callback
////////////////////////////////////////////////////////////////////////////////

void nmiPiano::plot_callback(const nmiPiano::Iteration &iteration, const std::string &file)
{
    std::ofstream out;
    out.open(file, std::ios_base::app);
    out << iteration.n << "," << iteration.f_x_n + iteration.g_x_n 
            << "," << iteration.f_x_n  << "," << iteration.g_x_n 
            << "," << iteration.alpha_n <<  "," << iteration.L_n 
            << "," << iteration.Delta_n << std::endl;
    out.close();
}

////////////////////////////////////////////////////////////////////////////////
// nmiPiano::nmiPiano
////////////////////////////////////////////////////////////////////////////////

nmiPiano::nmiPiano(const std::function<float(const Eigen::MatrixXf&)> f, 
        const std::function<void(const Eigen::MatrixXf&,Eigen::MatrixXf&)> df, 
        std::function<float(const Eigen::MatrixXf &)> g,
        const std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&, float alpha)> prox_g,
        nmiPiano::Options options,
        const std::function<void(const nmiPiano::Iteration&)> callback)
{
    f_ = f;
    df_ = df;
    g_ = g;
    prox_g_ = prox_g;
    callback_ = callback;
    options_ = options;
}

////////////////////////////////////////////////////////////////////////////////
// nmiPiano::~nmiPiano
////////////////////////////////////////////////////////////////////////////////

nmiPiano::~nmiPiano()
{
    // ...
}

////////////////////////////////////////////////////////////////////////////////
// nmiPiano::optimize
////////////////////////////////////////////////////////////////////////////////

void nmiPiano::optimize(Eigen::MatrixXf &x_star, float &f_x_star)
{
    nmiPiano::Iteration iteration;
    initialize(iteration);
    
    // Used for backtracking; referring to the next iterate x_np1:
    Eigen::MatrixXf x_np1;
    
    for (unsigned int t = 0; t <= options_.max_iter; t++)
    {
        callback_(iteration);
        
        // Backtrack for the lipschitz constant L_n:
        float L_nm1 = estimateL(iteration);
        
        // Fall back to last L_n in worst case.
        if (std::isinf(L_nm1) || std::isnan(L_nm1) || options_.BOUND_L_N)
        {
            LOG_IF(INFO, !options_.BOUND_L_N) << "Could not get starting value for local Lipschitz constant, using L_nm1 instead (L_n = " << L_nm1 << ")";
            L_nm1 = iteration.L_n;
        }
        
        bool condition = false;
        
        int l = 0; // Backtracking steps.
        do
        {
            iteration.L_n = std::pow(options_.eta,  l)*L_nm1;
            
            LOG_IF(FATAL, std::isinf(iteration.L_n) || std::isnan(iteration.L_n)) 
                    << "Could not find the local Lipschitz constant - L_n = " 
                    << iteration.L_n << std::endl;
            LOG_IF(INFO, l > 0 && l%1000 == 0) 
                    << "Having a hard time finding the local Lipschitz constant (L_n = " << iteration.L_n 
                    << "; L_nm1 = " << L_nm1 << "; eta = " << options_.eta << "; l = " << l << ")" << std::endl;
            
            iteration.alpha_n = 2*(1 - options_.beta)/iteration.L_n;
            LOG_IF(FATAL, iteration.alpha_n < 0)
                    << "Cannot choose alpha_n - it does not exist: alpha_n = " 
                    << iteration.alpha_n << " (L_n = " << iteration.L_n << ")!";
            
            
            // We fixes L_n and alpha_n, so we can try an iteration to check the
            // Lipschitz condition.
            iterate(iteration, options_.beta, x_np1);
            condition = checkLipschitz(iteration, x_np1);
            
            l++;
        }
        while (!condition); // Now alpha_n and L_n are set correctly.
        
        iteration.x_nm1 = iteration.x_n;
        iteration.x_n = x_np1;
        
        // For iteration 0, this is done in initialize():
        iteration.f_x_n = f_(iteration.x_n);
        iteration.g_x_n = g_(iteration.x_n);
        df_(iteration.x_n, iteration.df_x_n);
        
        LOG_IF(FATAL, iteration.x_n.rows() != iteration.df_x_n.rows() 
                || iteration.x_n.cols() != iteration.df_x_n.cols()) << "Output dimensions of df invalid.";
        
        iteration.n++;
        
        // Termination criterion
        if (iteration.Delta_n < options_.epsilon && options_.epsilon > 0)
        {
            break;
        }
    }
    
    x_star = iteration.x_n;
    f_x_star = iteration.f_x_n;
}

////////////////////////////////////////////////////////////////////////////////
// nmiPiano::initialize
////////////////////////////////////////////////////////////////////////////////

void nmiPiano::initialize(nmiPiano::Iteration &iteration)
{
    iteration.n = 0;
    iteration.x_n = options_.x_0;
    iteration.x_nm1 = options_.x_0;
    
    iteration.g_x_n = g_(iteration.x_n);
    iteration.f_x_n = f_(iteration.x_n);
    df_(iteration.x_n, iteration.df_x_n);
    
    LOG_IF(FATAL, iteration.x_n.rows() != iteration.df_x_n.rows() 
            || iteration.x_n.cols() != iteration.df_x_n.cols()) << "Output dimensions of df invalid.";
    
    iteration.alpha_n = 0.1f; // Required for estimateL!
    
    // Estimate L_n.
    if (options_.L_0m1 > 0)
    {
        iteration.L_n = options_.L_0m1;
    }
    
    // Estimate L and take max.
    if (iteration.df_x_n.squaredNorm() > 0.001)
    {
        iteration.L_n = std::max(iteration.L_n, estimateL(iteration));
    }
    
    iteration.alpha_n = 2*(1 - options_.beta)/iteration.L_n;
}

float nmiPiano::estimateL(const nmiPiano::Iteration &iteration)
{
    Eigen::MatrixXf x_tilde = iteration.x_n - iteration.alpha_n*iteration.df_x_n;

    Eigen::MatrixXf df_x_tilde;
    df_(x_tilde, df_x_tilde);

    Eigen::MatrixXf delta_df_x = iteration.df_x_n - df_x_tilde;
    Eigen::MatrixXf delta_x = iteration.x_n - x_tilde;
    
    float L_n = std::sqrt(delta_df_x.squaredNorm())/std::sqrt(delta_x.squaredNorm ());
    if (options_.BOUND_L_N) {
        L_n = std::max(L_n, options_.L_0m1);
    }
    
    return L_n;
}

////////////////////////////////////////////////////////////////////////////////
// nmiPiano::iterate
////////////////////////////////////////////////////////////////////////////////

void nmiPiano::iterate(nmiPiano::Iteration& iteration, float beta, Eigen::MatrixXf& x_np1)
{
    // TODO: precompute x_n - x_nm1
    iteration.Delta_x_n = - iteration.alpha_n*iteration.df_x_n + beta*(iteration.x_n - iteration.x_nm1);
    
    prox_g_(iteration.x_n + iteration.Delta_x_n, x_np1, iteration.alpha_n);
    
    LOG_IF(FATAL, iteration.x_n.rows() != x_np1.rows() 
            || iteration.x_n.cols() != x_np1.cols()) << "Output dimensions of prox_g invalid.";
    
    Eigen::MatrixXf Delta_n = iteration.x_n - x_np1;
    iteration.Delta_n = std::sqrt(Delta_n.squaredNorm());
}

////////////////////////////////////////////////////////////////////////////////
// nmiPiano::checkLipschitz
////////////////////////////////////////////////////////////////////////////////

bool nmiPiano::checkLipschitz(const nmiPiano::Iteration& iteration, const Eigen::MatrixXf& x_np1)
{
    float f_x_np1 = f_(x_np1);
    
    Eigen::MatrixXf residual = x_np1 - iteration.x_n;
    Eigen::MatrixXf product = iteration.df_x_n.array()*residual.array();
    float threshold = iteration.f_x_n + product.squaredNorm() + iteration.L_n/2.f * residual.squaredNorm();
    
    return (f_x_np1 <= threshold);
}

#endif	/* NMIPIANO_H */

