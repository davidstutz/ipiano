/**
 * Using iPiano, as described in [1] and [2], for denoising grayscale images.
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
 * $ ./image_denoising_cli/image_denoising_cli --help
 * Allowed options:
 *   --image arg                    image file
 *   --sigma arg (=0.0500000007)    noise level
 *   --lambda arg (=0.100000001)    balancing term between unary and pairwise term
 *   --iterations arg (=250)        number of iterations
 *   --eta arg (=1.04999995)        eta, i.e. coefficient to choose local 
 *                                  Lipschitz constant
 *   --beta arg (=0.5)              momentum parameter (for nmiPiano) or 
 *                                  initialization of momentum parameter (iPiano)
 *   --c1 arg (=9.99999996e-13)     c1 for iPiano
 *   --c2 arg (=9.99999996e-13)     c2 for iPiano
 *   --epsilon arg (=0.00100000005) epsilon for iPiano
 *   -h [ --help ]                  produce help message
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
#include <opencv2/opencv.hpp>
#include "nmipiano.h"
#include "ipiano.h"
#include "functionals.h"

int main(int argc, char** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("image", boost::program_options::value<std::string>(), "image file")
        ("sigma", boost::program_options::value<float>()->default_value(0.05), "noise level")
        ("lambda", boost::program_options::value<float>()->default_value(0.1), "balancing term between unary and pairwise term")
        ("iterations", boost::program_options::value<int>()->default_value(250), "number of iterations")
        ("eta", boost::program_options::value<float>()->default_value(1.05), "eta, i.e. coefficient to choose local Lipschitz constant")
        ("beta", boost::program_options::value<float>()->default_value(0.5), "momentum parameter (for nmiPiano) or initialization of momentum parameter (iPiano)")
        ("c1", boost::program_options::value<float>()->default_value(1e-12), "c1 for iPiano")
        ("c2", boost::program_options::value<float>()->default_value(1e-12), "c2 for iPiano")
        ("epsilon", boost::program_options::value<float>()->default_value(0.001), "epsilon for iPiano")
        ("help,h", "produce help message");
    
    boost::program_options::positional_options_description positionals;
    positionals.add("image", 1);
    
    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
    boost::program_options::notify(parameters);

    if (parameters.find("help") != parameters.end()) {
        std::cout << desc << std::endl;
        return 1;
    }
    
    boost::filesystem::path image_file(parameters["image"].as<std::string>());
    if (!boost::filesystem::is_regular_file(image_file))
    {
        std::cout << "Image does not exist." << std::endl;
        return 1;
    }
    
    float sigma = parameters["sigma"].as<float>();
    float lambda = parameters["lambda"].as<float>();
    int iterations = parameters["iterations"].as<int>();
    float eta = parameters["eta"].as<float>();
    float beta = parameters["beta"].as<float>();
    float c1 = parameters["c1"].as<float>();
    float c2 = parameters["c2"].as<float>();
    float epsilon = parameters["epsilon"].as<float>();
    
    cv::Mat image = cv::imread(image_file.string());

    // Ensure that we have a gray scale image.
    if (image.channels() > 1)
    {
        cv::cvtColor(image, image, CV_BGR2GRAY);
    }

    cv::Mat float_image;
    image.convertTo(float_image, CV_32FC1);
    float_image /= 255;

    Eigen::MatrixXf signal;
    Functionals::Util::convertOpenCVToEigen(float_image, signal);

    Eigen::MatrixXf perturbed_signal;
    Functionals::Denoising::Noise::addGaussianAdditive(signal, sigma,
            perturbed_signal);

    cv::Mat image_tilde(image.rows, image.cols, CV_8UC1);
    for (int i = 0; i < image_tilde.rows; i++)
    {
        for (int j = 0; j < image_tilde.cols; j++)
        {
            image_tilde.at<unsigned char> (i, j) = std::min(255.f, std::max(0.f, perturbed_signal(i, j)*255));
        }
    }
    
    cv::imwrite(image_file.stem().string() + "_tilde.png", image_tilde);          

    std::function<float(const Eigen::MatrixXf&)> bound_f 
            = std::bind(Functionals::Denoising::f_lorentzianPairwise, 
            std::placeholders::_1, sigma, lambda);
    std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&)> bound_df
            = std::bind(Functionals::Denoising::df_lorentzianPairwise, 
            std::placeholders::_1, std::placeholders::_2, sigma, lambda);
    std::function<float(const Eigen::MatrixXf&)> bound_g
            = std::bind(Functionals::Denoising::g_absoluteUnary, 
            std::placeholders::_1, perturbed_signal);
    std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&, float)> bound_prox_g
            = std::bind(Functionals::Denoising::prox_g_absoluteUnary, 
            std::placeholders::_1, perturbed_signal, std::placeholders::_2, 
            std::placeholders::_3);
    std::function<float(const Eigen::MatrixXf&)> bound_g_squared
            = std::bind(Functionals::Denoising::g_squaredUnary, 
            std::placeholders::_1, perturbed_signal);
    std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&, float)> bound_prox_g_squared
            = std::bind(Functionals::Denoising::prox_g_squaredUnary, 
            std::placeholders::_1, perturbed_signal, std::placeholders::_2, 
            std::placeholders::_3);

    ////////////////////////////////////////////////////////////////////
    // nmiPiano absolute
    ////////////////////////////////////////////////////////////////////

    nmiPiano::Options nmi_options;
    nmi_options.x_0 = perturbed_signal;
    nmi_options.max_iter = iterations;
    nmi_options.beta = beta;
    nmi_options.eta = eta;
    nmi_options.epsilon = epsilon;
    
    std::function<void(const typename nmiPiano::Iteration &iteration)> nmi_bound_callback 
            = std::bind(nmiPiano::default_callback, std::placeholders::_1, 10);
    
    {
        nmiPiano nmipiano(bound_f, bound_df, bound_g, bound_prox_g, 
                nmi_options, nmi_bound_callback);

        Eigen::MatrixXf nmi_x_star;
        float nmi_f_x_star;
        nmipiano.optimize(nmi_x_star, nmi_f_x_star);

        cv::Mat image_nmi(image.rows, image.cols, CV_8UC1);
        for (int i = 0; i < image_nmi.rows; i++)
        {
            for (int j = 0; j < image_nmi.cols; j++)
            {
                image_nmi.at<unsigned char> (i, j) = std::min(255.f, std::max(0.f, nmi_x_star(i, j)*255));
            }
        }

        cv::imwrite(image_file.stem().string() + "_nmipiano_absolute.png", image_nmi);
    }
            
    ////////////////////////////////////////////////////////////////////
    // nmiPiano squared
    ////////////////////////////////////////////////////////////////////

    {
        nmiPiano nmipiano(bound_f, bound_df, bound_g_squared, bound_prox_g_squared, 
                nmi_options, nmi_bound_callback);

        Eigen::MatrixXf nmi_x_star;
        float nmi_f_x_star;
        nmipiano.optimize(nmi_x_star, nmi_f_x_star);

        cv::Mat image_nmi(image.rows, image.cols, CV_8UC1);
        for (int i = 0; i < image_nmi.rows; i++)
        {
            for (int j = 0; j < image_nmi.cols; j++)
            {
                image_nmi.at<unsigned char> (i, j) = std::min(255.f, std::max(0.f, nmi_x_star(i, j)*255));
            }
        }

        cv::imwrite(image_file.stem().string() + "_nmipiano_squared.png", image_nmi);
    }
    
    ////////////////////////////////////////////////////////////////////
    // iPiano absolute
    ////////////////////////////////////////////////////////////////////

    iPiano::Options i_options;
    i_options.x_0 = perturbed_signal;
    i_options.max_iter = iterations;
    i_options.beta_0m1 = beta;
    i_options.c_1 = c1;
    i_options.c_2 = c2;
    i_options.eta = eta;
    i_options.epsilon = epsilon;
    
    std::function<void(const typename iPiano::Iteration &iteration)> i_bound_callback 
            = std::bind(iPiano::default_callback, std::placeholders::_1, 10);
    
    {
        iPiano ipiano(bound_f, bound_df, bound_g, bound_prox_g, 
                i_options, i_bound_callback);

        Eigen::MatrixXf i_x_star;
        float i_f_x_star;
        ipiano.optimize(i_x_star, i_f_x_star);

        cv::Mat image_i(image.rows, image.cols, CV_8UC1);
        for (int i = 0; i < image_i.rows; i++)
        {
            for (int j = 0; j < image_i.cols; j++)
            {
                image_i.at<unsigned char> (i, j) = std::min(255.f, std::max(0.f, i_x_star(i, j)*255));
            }
        }

        cv::imwrite(image_file.stem().string() + "_ipiano_absolute.png", image_i);
    }
            
    ////////////////////////////////////////////////////////////////////
    // iPiano squared
    ////////////////////////////////////////////////////////////////////
            
    {
        iPiano ipiano(bound_f, bound_df, bound_g_squared, bound_prox_g_squared, 
                i_options, i_bound_callback);

        Eigen::MatrixXf i_x_star;
        float i_f_x_star;
        ipiano.optimize(i_x_star, i_f_x_star);

        cv::Mat image_i(image.rows, image.cols, CV_8UC1);
        for (int i = 0; i < image_i.rows; i++)
        {
            for (int j = 0; j < image_i.cols; j++)
            {
                image_i.at<unsigned char> (i, j) = std::min(255.f, std::max(0.f, i_x_star(i, j)*255));
            }
        }

        cv::imwrite(image_file.stem().string() + "_ipiano_squared.png", image_i);
    }       
}

