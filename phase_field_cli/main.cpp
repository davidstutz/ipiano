/* 
 * File:   main.cpp
 * Author: david
 *
 * Created on January 9, 2016, 3:04 PM
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
    
    if (image.channels() > 1)
    {
        cv::cvtColor(image, image, CV_BGR2GRAY);
    }
    
    cv::Mat float_image;
    image.convertTo(float_image, CV_32FC1);
    float_image /= 255;
    
    // TODO better initialization for Cp, Cm
    float Cp = 1;
    float Cm = 0;
    
    Eigen::MatrixXf x_0;
    Functionals::Util::convertOpenCVToEigen(float_image, x_0);
    
    Eigen::MatrixXf x = Eigen::MatrixXf::Zero(x_0.rows(), x_0.cols());
    
    const float sigma = 0.1;
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
                = std::bind(Functionals::PhaseField::f, std::placeholders::_1, 
                epsilon);
        std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&)> bound_df 
                = std::bind(Functionals::PhaseField::df, std::placeholders::_1, 
                std::placeholders::_2, epsilon);
        std::function<float(const Eigen::MatrixXf&)> bound_g
                = std::bind(Functionals::PhaseField::g_absolute, std::placeholders::_1, 
                x_0, Cp, Cm, lambda);
        std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&, float)> bound_prox_g
                = std::bind(Functionals::PhaseField::prox_g_absolute, std::placeholders::_1, 
                x_0, std::placeholders::_2, Cp, Cm, lambda, std::placeholders::_3);

        nmiPiano::Options options;
        options.x_0 = x;
        options.max_iter = 250;
        options.beta = 0.5f;
//        options.beta_0m1 = 0.5;
//        options.L_0m1 = 10.f;
//        options.c_1 = 1e-16;
//        options.c_2 = 1e-16;
        options.eta = 1.05f;
        options.epsilon = 1e-2;

        std::function<void(const nmiPiano::Iteration &iteration)> bound_callback 
                = std::bind(nmiPiano::default_callback, std::placeholders::_1, 10);
        nmiPiano ipiano(bound_f, bound_df, bound_g, bound_prox_g, options, 
                bound_callback);
        
        float f_x;
        ipiano.optimize(x, f_x);
        
        Cp = Functionals::PhaseField::Cp(x, x_0);
        Cm = Functionals::PhaseField::Cm(x, x_0);
        std::cout << "[" << k << "] C_p = " << Cp << "; C_m = " << Cm << std::endl;
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

