/**
 * Using iPiano, as described in [1] and [2], for image segmentation.
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
 * $ ./phase_field_color_cli/phase_field_color_cli --helpAllowed options:
 *   --image arg           image
 *   -h [ --help ]         produce help message
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

int main(int argc, char** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("image", boost::program_options::value<std::string>(), "image")
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
        std::cout << "Image file doe snot exist." << std::endl;
        return 1;
    }
    
    const float epsilon = 1;
    const float lambda = 100;
    
    cv::Mat image = cv::imread(image_file.string());
    
    if (image.channels() != 3)
    {
        std::cout << "Meant for color images!" << std::endl;
        return 1;
    }
    
    cv::cvtColor(image, image, CV_BGR2XYZ);
    
    cv::Mat float_image;
    image.convertTo(float_image, CV_32FC3);
    float_image /= 255;
    
    // TODO better initialization for Cp, Cm
    Eigen::Vector3f Cp = Eigen::Vector3f::Zero()*0.25f;
//    Cp(2, 0) = 0.f;
    
    Eigen::Vector3f Cm = Eigen::Vector3f::Zero()*0.75f;
//    Cm(0, 0) = 1.f;
    
    Eigen::MatrixXf x_0;
    Functionals::Util::convertOpenCVToEigen_Color(float_image, x_0);
    
    Eigen::MatrixXf x = Eigen::MatrixXf::Zero(float_image.rows, float_image.cols);
    
    const float sigma = 0.05f;
    std::random_device random;
    std::mt19937 generator(random());
    std::normal_distribution<float> gaussian(0, sigma);

    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            x(i, j) += gaussian(generator);
        }
    }
    
    const int K = 10;
    for (int k = 0; k < K; k++)
    {
        std::function<float(const Eigen::MatrixXf&)> bound_f 
                = std::bind(Functionals::PhaseField_Color::f, std::placeholders::_1, epsilon);
        std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&)> bound_df 
                = std::bind(Functionals::PhaseField_Color::df, std::placeholders::_1, 
                std::placeholders::_2, epsilon);
        std::function<float(const Eigen::MatrixXf&)> bound_g
                = std::bind(Functionals::PhaseField_Color::g_absolute, std::placeholders::_1, 
                x_0, Cp, Cm, lambda);
        std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&, float)> bound_prox_g
                = std::bind(Functionals::PhaseField_Color::prox_g_absolute, std::placeholders::_1, 
                x_0, std::placeholders::_2, Cp, Cm, lambda, std::placeholders::_3);

        nmiPiano::Options options;
        options.x_0 = x;
        options.max_iter = 250;
        options.beta = 0.5f;
//        options.beta_0m1 = 0.5;
        options.L_0m1 = 1.f;
//        options.c_1 = 1e-16;
//        options.c_2 = 1e-16;
        options.eta = 1.1f;
        options.epsilon = 1e-2;

        std::function<void(const nmiPiano::Iteration &iteration)> bound_callback 
                = std::bind(nmiPiano::default_callback, std::placeholders::_1, 10);
        nmiPiano ipiano(bound_f, bound_df, bound_g, bound_prox_g, options, 
                bound_callback);
        
        float f_x;
        ipiano.optimize(x, f_x);
        
        Functionals::PhaseField_Color::Cp(x, x_0, Cp);
        Functionals::PhaseField_Color::Cm(x, x_0, Cm);
        std::cout << "[" << k << "] C_p = " << Cp(0, 0) << "," << Cp(1, 0) << "," << Cp(2, 0) 
                << "; C_m = " << Cm(0, 0) << "," << Cm(1, 0) << "," << Cm(2, 0) << std::endl;
    }
    
    for (float threshold = -0.8f; threshold <= 0.85f; threshold += 0.1f) {
        cv::Mat segmentation(float_image.rows, float_image.cols, CV_8UC1, cv::Scalar(0));

        for (int i = 0; i < segmentation.rows; i++)
        {
            for (int j = 0; j < segmentation.cols; j++)
            {
                if (x(i, j) > threshold)
                {
                    segmentation.at<unsigned char>(i, j) = 255;
                }
            }
        }

        boost::filesystem::path segmentation_file = image_file.parent_path()
                / boost::filesystem::path(image_file.stem().string() + "_" + std::to_string(threshold) + "_seg.png");
        cv::imwrite(segmentation_file.string(), segmentation);
    }
}

