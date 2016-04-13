/**
 * Using iPiano, as described in [1] and [2], for denoising one-dimensional signals.
 * 
 * [1] P. Ochs, Y. Chen, T. Brox, T. Pock.
 *     iPiano: Inertial Proximal Algorithm for Nonconvex Optimization
 *     SIAM Journal of Imaging Sciences, colume 7, number 2, 2014.
 * [2] D. Stutz.
 *     Seminar paper "iPiano: Inertial Proximal Algorithm for Non-Convex Optimization"
 *     https://github.com/davidstutz/seminar-ipiano
 * 
 * Usage:
 * 
 * $ ./signal_denoising_cli/signal_denoising_cli --help
 * Allowed options:
 *   --seed arg (=1460573323) seed
 *   -h [ --help ]            produce help message
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
#include <functional>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include <glog/logging.h>
#include <opencv2/opencv.hpp> // Visualize 1D signals.
#include "nmipiano.h"
#include "ipiano.h"
#include "functionals.h"

const int M = 100; // Dimension of signal.
const float lambda = 0.5;
const float sigma = 0.05;
    
void sampleSignal(Eigen::MatrixXf &signal)
{
    signal = Eigen::MatrixXf::Zero(M, 1);
    
    for (int i = 0; i < M/2; i++)
    {
        signal(i) = 0;
    }
    
    for (int i = M/2; i < M; i++)
    {
        signal(i) = 1;
    }
}

void perturbSignal(Eigen::MatrixXf &signal)
{
    const float sigma = 0.05;
    std::random_device random;
    std::mt19937 generator(random());
    std::normal_distribution<float> gaussian(0, sigma);
    
    for (int i = 0; i < signal.rows(); i++)
    {
        signal(i) = signal(i) + gaussian(generator);
    }
}

void visualizeSignal(const Eigen::MatrixXf &signal, 
        cv::Mat &image)
{
    const int width = 100;
    const int height = 5;
    image.create(height*M, width, CV_8UC1);
    
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            image.at<unsigned char>(i, j) = (unsigned char) (signal(i/height, 0)*127 + 63);
        }
    }
}

int main(int argc, char** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("seed", boost::program_options::value<int> ()->default_value(time(0)), "seed")
        ("help,h", "produce help message");
    
    boost::program_options::positional_options_description positionals;
    
    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
    boost::program_options::notify(parameters);

    if (parameters.find("help") != parameters.end()) {
        std::cout << desc << std::endl;
        return 1;
    }
    
    std::srand (parameters["seed"].as<int>());
    
    Eigen::MatrixXf signal;
    sampleSignal(signal);
    
    Eigen::MatrixXf perturbed_signal = signal;
    perturbSignal(perturbed_signal);
    
    std::function<float(const Eigen::MatrixXf&)> bound_f 
            = std::bind(Functionals::Denoising::f_lorentzianPairwise, std::placeholders::_1, sigma, lambda);
    std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&)> bound_df 
            = std::bind(Functionals::Denoising::df_lorentzianPairwise, std::placeholders::_1, std::placeholders::_2, sigma, lambda);
    std::function<float(const Eigen::MatrixXf&)> bound_g
            = std::bind(Functionals::Denoising::g_absoluteUnary, std::placeholders::_1, perturbed_signal);
    std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&, float)> bound_prox_g
            = std::bind(Functionals::Denoising::prox_g_absoluteUnary, std::placeholders::_1, perturbed_signal, std::placeholders::_2, std::placeholders::_3);
    
    // Starting signal:
    Eigen::MatrixXf x_0 = Eigen::MatrixXf::Zero(M, 1);
    perturbSignal(x_0);
    
    nmiPiano::Options nmi_options;
    nmi_options.x_0 = x_0;
    nmi_options.max_iter = 1000;
    nmi_options.L_0m1 = 100.f;
    nmi_options.beta = 0.5;
    nmi_options.eta = 1.05;
    nmi_options.epsilon = 1e-8;
    
    std::function<void(const nmiPiano::Iteration &iteration)> nmi_bound_callback 
            = std::bind(nmiPiano::default_callback, std::placeholders::_1, 10);
    nmiPiano nmipiano(bound_f, bound_df, bound_g, bound_prox_g, nmi_options, 
            nmi_bound_callback);
    
    Eigen::MatrixXf nmi_x_star;
    float nmi_f_x_star;
    nmipiano.optimize(nmi_x_star, nmi_f_x_star);
    
    iPiano::Options i_options;
    i_options.x_0 = x_0;
    i_options.max_iter = 1000;
    i_options.beta_0m1 = 0.5;
    i_options.L_0m1 = 100.f;
    i_options.c_1 = 1e-12;
    i_options.c_2 = 1e-12;
    i_options.eta = 1.05;
    i_options.steps = 10000;
    i_options.epsilon = 1e-8;
    
    std::function<void(const iPiano::Iteration &iteration)> i_bound_callback 
            = std::bind(iPiano::default_callback, std::placeholders::_1, 10);
    iPiano ipiano(bound_f, bound_df, bound_g, bound_prox_g, i_options, 
            i_bound_callback);
    
    Eigen::MatrixXf i_x_star;
    float i_f_x_star;
    ipiano.optimize(i_x_star, i_f_x_star);
    
    std::cout << "x_star" << std::endl;
    for (int i = 0; i < M; i++)
    {
        std::cout << signal(i) << " " << perturbed_signal(i) << " " << nmi_x_star(i) << " " << i_x_star(i) << std::endl;
    }
    
    cv::Mat image_signal;
    visualizeSignal(signal, image_signal);
    cv::Mat image_perturbed_signal;
    visualizeSignal(perturbed_signal, image_perturbed_signal);
    cv::Mat image_nmipiano;
    visualizeSignal(nmi_x_star, image_nmipiano);
    cv::Mat image_ipiano;
    visualizeSignal(i_x_star, image_ipiano);
    
    cv::imshow("Signal", image_signal);
    cv::imshow("Input", image_perturbed_signal);
    cv::imshow("nmiPiano", image_nmipiano);
    cv::imshow("iPiano", image_ipiano);
    cv::waitKey(0);
}

