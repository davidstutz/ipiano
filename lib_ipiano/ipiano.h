/**
 * Implementation of iPiano as described in [1] and [2]:
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
#ifndef IPIANO_H
#define	IPIANO_H

#include <random>
#include <fstream>
#include <functional>
#include <Eigen/Dense>
#include <glog/logging.h>

#include "nmipiano.h"

/** \brief Implementation of iPiano (algorithm 5) as proposed in [1]:
 *  [1] P. Ochs, Y. Chen, T. Brox, T. Pock.
 *      iPiano: Inertial Proximal Algorithm for Nonconvex Optimization.
 *      SIAM J. Imaging Sciences, vol. 7, no. 2, 2014.
 * \author David Stutz
 */
class iPiano : public nmiPiano
{
public:
    
    /** \brief Options of algorithm. */
    struct Options : public nmiPiano::Options
    {
        /** \brief beta_0m1 to initialize alpha_0m1, delta_0m1 and gamma_0m1 according to Equations (21) and (22). */
        float beta_0m1 = 0.5f;
        /** \brief Fixed c_1. */
        float c_1 = 1e-8;
        /** \brief Fixed c_2. */
        float c_2 = 1e-8;
        /** \brief Number of dicsrete steps for alpha_n and beta_n to try. */
        int steps = 10000;
    };
    
    /** \brief Structure representing an iteration, passed as is to a callback function
     * to be able to monitor process. */
    struct Iteration : public nmiPiano::Iteration
    {
        /** \brief beta_n of current iterate (within iterations, this is also used to estimate beta_np1). */
        float beta_n;
        /** \brief delta_n of current iterate (within iterations, this is also used to estimate delta_np1). */
        float delta_n;
        /** \brief gamma_n of current iterate (within iterations, this is also used to estimate gamma_np1). */
        float gamma_n;
    };
    
    /** \brief Silent callback to use with iPiano.
     * \param[in] iteration
     */
    static void silent_callback(const Iteration &iteration);
    
    /** \brief Default callback to use with iPiano.
     * \param[in] iteration
     */
    static void default_callback(const Iteration &iteration, int n = 10);
    
    /** \brief Brief callback to use with iPiano.
     * \param[in] iteration
     */
    static void brief_callback(const Iteration &iteration, int n = 10);
    
    /** \brief File callback, used to write all information to the specified file.
     * \param[in] iteration
     * \param[in] file
     */
    static void file_callback(const Iteration &iteration, const std::string &file);
    
    /** \brief File callback for plotting, writes CSV output of format: iteration,f,g,alpha_n,beta_n,L_n,Delta_n.
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
    iPiano(const std::function<float(const Eigen::MatrixXf&)> f, 
            const std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&)> df,
            std::function<float(const Eigen::MatrixXf&)> g,
            const std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&, float alpha)> prox_g,
            Options options,
            const std::function<void(const Iteration&)> callback = [](const Iteration &iteration) -> void { /* Nothing ... */ });
            
    /** \brief Destructor. */
    ~iPiano();
    
    /** \brief Optimize the given objective using the iPiano algorithm. 
     * \param[out] x_star
     * \param[out] f_x_star
     */
    void optimize(Eigen::MatrixXf &x_star, float &f_x_star);
    
protected:
    
    /** \brief Initialize the iteration structure (i.e. set iteration 0): 
     * \param[in] iteration
     */
    void initialize(Iteration &iteration);
    
    /** \brief Callback, note: overwrites nmiPiano::callback_! */
    std::function<void(const Iteration&)> callback_;
    /** \brief Options (note: overwrites nmiPiano::options_!). */
    Options options_;
};

////////////////////////////////////////////////////////////////////////////////
// iPiano::silent_callback
////////////////////////////////////////////////////////////////////////////////

void iPiano::silent_callback(const iPiano::Iteration &iteration)
{
    
}

////////////////////////////////////////////////////////////////////////////////
// iPiano::default_callback
////////////////////////////////////////////////////////////////////////////////

void iPiano::default_callback(const iPiano::Iteration &iteration, int n)
{
    if (iteration.n%n == 0)
    {
        std::cout << "[" << iteration.n << "] " 
                << iteration.f_x_n + iteration.g_x_n
                << " (Delta_n = " << iteration.Delta_n << "; L_n = " << iteration.L_n 
                << "; alpha_n = " << iteration.alpha_n << "; beta_n = " << iteration.beta_n << ")"
                << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
// iPiano::default_callback
////////////////////////////////////////////////////////////////////////////////

void iPiano::brief_callback(const iPiano::Iteration &iteration, int n)
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
// iPiano::file_callback
////////////////////////////////////////////////////////////////////////////////

void iPiano::file_callback(const iPiano::Iteration &iteration, const std::string &file)
{
    std::ofstream out;
    out.open(file, std::ios_base::app);
    out << "iteration = " << iteration.n << "; f_x_n = " << iteration.f_x_n 
            << "; g_x_n = " << iteration.g_x_n << "; alpha_n = " << iteration.alpha_n
            << "; beta_n = " << iteration.beta_n <<  "; L_n = " << iteration.L_n 
            << "; |df_x_n| = " << std::sqrt(iteration.df_x_n.squaredNorm()) 
            << "; Delta_n = " << iteration.Delta_n << std::endl;
    out.close();
}

////////////////////////////////////////////////////////////////////////////////
// iPiano::plot_callback
////////////////////////////////////////////////////////////////////////////////

void iPiano::plot_callback(const iPiano::Iteration &iteration, const std::string &file)
{
    std::ofstream out;
    out.open(file, std::ios_base::app);
    out << iteration.n << "," << iteration.f_x_n + iteration.g_x_n 
            << "," << iteration.f_x_n  << "," << iteration.g_x_n 
            << "," << iteration.alpha_n << "," << iteration.beta_n 
            <<  "," << iteration.L_n << "," << iteration.Delta_n << std::endl;
    out.close();
}

////////////////////////////////////////////////////////////////////////////////
// iPiano::iPiano
////////////////////////////////////////////////////////////////////////////////

iPiano::iPiano(const std::function<float(const Eigen::MatrixXf&)> f, 
        const std::function<void(const Eigen::MatrixXf&,Eigen::MatrixXf&)> df, 
        std::function<float(const Eigen::MatrixXf &)> g,
        const std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&, float alpha)> prox_g,
        iPiano::Options options,
        const std::function<void(const iPiano::Iteration&)> callback)
        : nmiPiano(f, df, g, prox_g, options)
{
    options_ = options;
    callback_ = callback;
}

////////////////////////////////////////////////////////////////////////////////
// iPiano::~iPiano
////////////////////////////////////////////////////////////////////////////////

iPiano::~iPiano()
{
    // ...
}

////////////////////////////////////////////////////////////////////////////////
// iPiano::optimize
////////////////////////////////////////////////////////////////////////////////

void iPiano::optimize(Eigen::MatrixXf &x_star, float &f_x_star)
{
    iPiano::Iteration iteration;
    initialize(iteration);
    
    // Used for backtracking; referring to the next iterate x_np1:
    Eigen::MatrixXf x_np1;
    
    for (unsigned int t = 0; t <= options_.max_iter; t++)
    {
        callback_(iteration);
        
        // Backtrack for the Lipschitz constant L_n and the updates
        // alpha_n and beta_n:
        float L_nm1 = estimateL(iteration);
        
        // Fall back to last L_n in worst case.
        if (std::isinf(L_nm1) || std::isnan(L_nm1) || options_.BOUND_L_N)
        {
            LOG_IF(INFO, !options_.BOUND_L_N) << "Could not get starting value for local Lipschitz constant, using L_nm1 (L_n = " << L_nm1 << ")";
            L_nm1 = iteration.L_n;
        }
        
        float delta_nm1 = iteration.delta_n;
        bool condition = false;
        
        int l = 0;
        do
        {
            iteration.L_n = std::pow(options_.eta, l)*L_nm1;
            
            LOG_IF(FATAL, std::isinf(iteration.L_n) || std::isnan(iteration.L_n)) 
                    << "Could not find the local Lipschitz constant - L_n = " 
                    << iteration.L_n << std::endl;
            LOG_IF(INFO, l > 0 && l%1000 == 0) 
                    << "Having a hard time finding the local Lipschitz constant (L_n = " << iteration.L_n 
                    << "; L_nm1 = " << L_nm1 << "; eta = " << options_.eta << "; l = " << l << ")" << std::endl;
            
            float b = (delta_nm1 + iteration.L_n/2)/(options_.c_2 + iteration.L_n/2);
            float beta = (b - 1)/(b - 1.f/2.f);
//            float beta = 0.999;
            
            for (iteration.beta_n = beta; iteration.beta_n >= 0; 
                    iteration.beta_n -= beta/options_.steps)
            {
                bool take = false; // Whether to take current beta_n.
//                float alpha = 2*(1 - iteration.beta_n)/(iteration.L_n/2.f + options_.c_2);
                float alpha = 2*(1 - iteration.beta_n)/(iteration.L_n);
                
                LOG_IF(FATAL, alpha < options_.c_1)
                        << "Cannot choose alpha_n - it does not exist: c_1 = " 
                        << options_.c_1  << "; alpha_n = " << alpha
                        << " (L_n = " << iteration.L_n << ")!";
                
                for (iteration.alpha_n = alpha; iteration.alpha_n > options_.c_1; 
                        iteration.alpha_n -= (alpha - options_.c_1)/options_.steps)
                {
                    // TODO save some computation here!
                    iteration.delta_n = 1/iteration.alpha_n - iteration.L_n/2 
                            - iteration.beta_n/(2*iteration.alpha_n);
                    iteration.gamma_n = 1/iteration.alpha_n - iteration.L_n/2 
                            - iteration.beta_n/iteration.alpha_n;
                    
                    if (iteration.delta_n < delta_nm1
                            && iteration.delta_n >= iteration.gamma_n 
                            && iteration.gamma_n >= options_.c_2)
                    {
                        // Good parameters, so take these!
                        take = true;
                        break;
                    }
                }
                
                if (take)
                {
                    break;
                }
            }
                
            // We fixes L_n and alpha_n, so we can try an iteration to check the
            // Lipschitz condition.
            iterate(iteration, iteration.beta_n, x_np1);
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
// iPiano::initialize
////////////////////////////////////////////////////////////////////////////////

void iPiano::initialize(iPiano::Iteration &iteration)
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
    
    // beta_0m1 is given by the options!
    iteration.beta_n = options_.beta_0m1;
    
    // Initialize alpha_0m1 to suffice alpha_0m1 <= 2*(1 - beta_0m)/L_0m1
    // and delta_0m1 >= gamma_0m1 > c_2 where delta_0m1 and gamma_0m1 depend upon alpha_0m1.
    
//    float alpha = 2*(1 - iteration.beta_n)/(iteration.L_n/2.f + options_.c_2);
    float alpha = 2*(1 - iteration.beta_n)/(iteration.L_n);
    
    LOG_IF(FATAL, alpha < options_.c_1)
            << "Cannot choose alpha_n - it does not exist: c_1 = " 
            << options_.c_1  << "; alpha_n = " << alpha 
            << " (L_n = " << iteration.L_n << ")!";
                
    for (iteration.alpha_n = alpha; iteration.alpha_n > options_.c_1; iteration.alpha_n -= (alpha - options_.c_1)/options_.steps)
    {
        // TODO save some computation here!
        iteration.delta_n = 1/iteration.alpha_n - iteration.L_n/2 
                - iteration.beta_n/(2*iteration.alpha_n);
        iteration.gamma_n = 1/iteration.alpha_n - iteration.L_n/2 
                - iteration.beta_n/iteration.alpha_n;

        if (iteration.delta_n >= iteration.gamma_n && iteration.gamma_n >= options_.c_2)
        {
            // Good parameters, so take these!
            break;
        }
    }
}

#endif	/* IPIANO_H */

