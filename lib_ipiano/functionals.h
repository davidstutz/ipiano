/**
 * Implementation of several utilities and functions for applying iPiano and nmiPiano, 
 * as described in [1] and [2], to computer vision problems:
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
#ifndef FUNCTIONALS_H
#define	FUNCTIONALS_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

/** \brief A collection of functionals for different computer vision/image processing
 * tasks used to demonstrate the use of nmiPiano/iPiano optimization.
 * \author David Stutz
 */
class Functionals
{
public:
    
    /** \brief Utilities for using nmiPiano/iPiano for computer vision/image processing. */
    class Util
    {
    public:
        
        /** \brief Convert Eigen matrix to OpenCV matrix. 
         * \param[in] eigen
         * \param[out] opencv
         */
        static void convertEigenToOpenCV(const Eigen::MatrixXf &eigen, cv::Mat &opencv);
        
        /** \brief Convert OpenCV matrix to Eigen matrix.
         * \param[in] opencv
         * \param[out] eigen
         */
        static void convertOpenCVToEigen(const cv::Mat &opencv, Eigen::MatrixXf &eigen);
        
        /** \brief Convert Eigen matrix to OpenCV color matrix. 
         * \param[in] eigen
         * \param[out] opencv
         */
        static void convertEigenToOpenCV_Color(const Eigen::MatrixXf &eigen, cv::Mat &opencv);
        
        /** \brief Convert OpenCV color matrix to Eigen matrix.
         * \param[in] opencv
         * \param[out] eigen
         */
        static void convertOpenCVToEigen_Color(const cv::Mat &opencv, Eigen::MatrixXf &eigen);
    };
    
    /** \brief Denoising functionals (for grayscale images!. */
    class Denoising
    {
    public:
        
        /** \brief Generate noisy images. */
        class Noise
        {
        public:

            /** \brief Add additive Gaussian noise. 
             * \param[in] image
             * \param[in] sigma
             * \param[out] noisy_image
             */
            static void addGaussianAdditive(const Eigen::MatrixXf &image, float sigma, 
                    Eigen::MatrixXf &noisy_image);

            /** \brief Add salt and pepper noise.
             * \param[in] image
             * \param[in] p
             * \param[out] noisy_image
             */
            static void addSaltPepper(const Eigen::MatrixXf &image, float p, 
                    Eigen::MatrixXf &noisy_image);
        };
        
        /** \brief Evaluation of denoising functionals. */
        class Evaluation
        {
        public:
            /** \brief Compute PSNR to evaluate denoising applications.
             * \param[in] image
             * \param[in] denoised_image
             * \param[in] range
             */
            static float computePSNR(const Eigen::MatrixXf &image, 
                    const Eigen::MatrixXf &denoised_image, float range = 1);
        };
        
        /** \brief Lorentzian pairwise term. 
         * \param[in] x
         * \param[in] sigma
         * \param[in] lambda
         */
        static float f_lorentzianPairwise(const Eigen::MatrixXf &x, float sigma, 
                float lambda);
        
        /** \brief Derivative of Lorentzian pairwise term. 
         * \param[in] x
         * \param[out] df_x
         * \param[in] sigma
         * \param[in] lambda
         */
        static void df_lorentzianPairwise(const Eigen::MatrixXf &x, Eigen::MatrixXf &df_x,
                float sigma, float lambda);
        
        /** \brief Squared pairwise term. 
         * \param[in] x
         * \param[in] sigma
         * \param[in] lambda
         */
        static float f_squaredPairwise(const Eigen::MatrixXf &x, float lambda);
        
        /** \brief Derivative of squared pairwise term.
         * \param[in] x
         * \param[out] df_x
         * \param[in] sigma
         * \param[in] lambda
         */
        static void df_squaredPairwise(const Eigen::MatrixXf &x, Eigen::MatrixXf &df_x,
                float lambda);
        
        /** \brief Smooth absolute pairwise term. 
         * \param[in] x
         * \param[in] sigma
         * \param[in] lambda
         */
        static float f_smoothAbsolutePairwise(const Eigen::MatrixXf &x, float epsilon, 
                float lambda);
        
        /** \brief Derivative of smooth absolute pairwise term. 
         * \param[in] x
         * \param[out] df_x
         * \param[in] sigma
         * \param[in] lambda
         */
        static void df_smoothAbsolutePairwise(const Eigen::MatrixXf &x, Eigen::MatrixXf &df_x,
                float epsilon, float lambda);
        
        /** \brief Absolute unary, i.e. data, term.
         * \param[in] x
         * \param[in] x_0
         * \return
         */
        static float g_absoluteUnary(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0);
        
        /** \brief Proximal map of absolute unary term.
         * \param[in] x
         * \param[in] x_0
         * \param[out] prox_f_x
         * \param[in] alpha
         */
        static void prox_g_absoluteUnary(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
                Eigen::MatrixXf &prox_f_x, float alpha);
        
        /** \brief Squared unary, i.e. data, term.
         * \param[in] x
         * \param[in] x_0
         * \return
         */
        static float g_squaredUnary(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0);
        
        /** \brief Proximal map of squared unary term.
         * \param[in] x
         * \param[in] x_0
         * \param[out] prox_f_x
         * \param[in] alpha
         */
        static void prox_g_squaredUnary(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
                Eigen::MatrixXf &prox_f_x, float alpha);
        
    };
    
    /** \brief Two Phase-Field for segmentation, see [1]. 
     *  [1] Shen.
     *      Gamma-Convergence Approximation to Piecewise Constant Mumford-Shah Segmentation.
     *      International Conference on Advanced Concepts of Intelligent Vision Systems, 2005.
     */
    class PhaseField
    {
    public:
        
        /** \brief Differentiable but non-convex part: \int 9 \eps \|\nabla z\|^2 + (1-z^2)^2/(64\eps) dx. 
         * \param[in] x
         * \param[in] epsilon
         */
        static float f(const Eigen::MatrixXf &x, float epsilon);
        
        /** \brief Derivative of f.
         * \param[in] x
         * \param[out] df_x
         * \param[in] epsilon
         */
        static void df(const Eigen::MatrixXf &x, Eigen::MatrixXf &df_x, 
                float epsilon);
        
        /** \brief Non-differentiable but convex part: \lambda \int (1 +- z)^2/4 (x_0 + Cpm)^2 dx. 
         * \param[in] x
         * \param[in] x_0
         * \param[in] Cp
         * \param[in] Cm
         * \param[in] epsilon
         * \param[in] lambda
         */
        static float g(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0, 
                float Cp, float Cm, float lambda);
        
        /** \brief Proximal map of g.
         * \param[in] x
         * \param[in] x_0
         * \param[out] prox_g_x
         * \param[in] Cp
         * \param[in] Cm
         * \param[in] epsilon
         * \param[in] lambda
         * \param[in] alpha
         */
        static void prox_g(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
                Eigen::MatrixXf &prox_g_x, float Cp, float Cm, float lambda, 
                float alpha);
        
        /** \brief Non-differentiable but convex part: \lambda \int |1 +- z| |x_0 + Cpm| dx.
         * \param[in] x
         * \param[in] x_0
         * \param[in] Cp
         * \param[in] Cm
         * \param[in] epsilon
         * \param[in] lambda
         */
        static float g_absolute(const Eigen::MatrixXf &x, const Eigen::MatrixXf &X_0,
                float Cp, float Cm, float lambda);
        
        /** \brief Proximal map of g_absolute.
         * \param[in] x
         * \param[in] x_0
         * \param[out] prox_g_x
         * \param[in] Cp
         * \param[in] Cm
         * \param[in] epsilon
         * \param[in] lambda
         * \param[in] alpha
         */
        static void prox_g_absolute(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
                Eigen::MatrixXf &prox_g_x, float Cp, float Cm, float lambda, 
                float alpha);
        
        /** \brief Compute Cp from the given labels x and the image x_0.
         * \param[in] x
         * \param[in] x_0
         */
        static float Cp(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0);
        
        /** \brief Compute Cp#m from the given labels x and the image x_0.
         * \param[in] x
         * \param[in] x_0
         */
        static float Cm(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0);
    };
    
    /** \brief Two Phase-Field for color image segmentation, see [1]; adapted to color. 
     *  [1] Shen.
     *      Gamma-Convergence Approximation to Piecewise Constant Mumford-Shah Segmentation.
     *      International Conference on Advanced Concepts of Intelligent Vision Systems, 2005.
     */
    class PhaseField_Color
    {
    public:
        
        /** \brief Differentiable but non-convex part: \int 9 \eps \|\nabla z\|^2 + (1-z^2)^2/(64\eps) dx. 
         * \param[in] x
         * \param[in] epsilon
         */
        static float f(const Eigen::MatrixXf &x, float epsilon);
        
        /** \brief Derivative of f.
         * \param[in] x
         * \param[out] df_x
         * \param[in] epsilon
         */
        static void df(const Eigen::MatrixXf &x, Eigen::MatrixXf &df_x, 
                float epsilon);
        
        /** \brief Non-differentiable but convex part: \lambda \int (1 +- z)^2/4 (x_0 + Cpm)^2 dx. 
         * \param[in] x
         * \param[in] x_0
         * \param[in] Cp
         * \param[in] Cm
         * \param[in] epsilon
         * \param[in] lambda
         */
        static float g(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0, 
                const Eigen::Vector3f &Cp, const Eigen::Vector3f &Cm, float lambda);
        
        /** \brief Proximal map of g.
         * \param[in] x
         * \param[in] x_0
         * \param[out] prox_g_x
         * \param[in] Cp
         * \param[in] Cm
         * \param[in] epsilon
         * \param[in] lambda
         * \param[in] alpha
         */
        static void prox_g(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
                Eigen::MatrixXf &prox_g_x, const Eigen::Vector3f &Cp, 
                const Eigen::Vector3f &Cm, float lambda, float alpha);
        
        /** \brief Non-differentiable but convex part: \lambda \int |1 +- z| |x_0 + Cpm| dx.
         * \param[in] x
         * \param[in] x_0
         * \param[in] Cp
         * \param[in] Cm
         * \param[in] epsilon
         * \param[in] lambda
         */
        static float g_absolute(const Eigen::MatrixXf &x, const Eigen::MatrixXf &X_0,
                const Eigen::Vector3f &Cp, const Eigen::Vector3f &Cm, float lambda);
        
        /** \brief Proximal map of g_absolute.
         * \param[in] x
         * \param[in] x_0
         * \param[out] prox_g_x
         * \param[in] Cp
         * \param[in] Cm
         * \param[in] epsilon
         * \param[in] lambda
         * \param[in] alpha
         */
        static void prox_g_absolute(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
                Eigen::MatrixXf &prox_g_x, const Eigen::Vector3f &Cp, const Eigen::Vector3f &Cm, 
                float lambda, float alpha);
        
        /** \brief Compute Cp from the given labels x and the image x_0.
         * \param[in] x
         * \param[in] x_0
         * \param[out] Cp
         */
        static void Cp(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0, Eigen::Vector3f &Cp);
        
        /** \brief Compute Cp#m from the given labels x and the image x_0.
         * \param[in] x
         * \param[in] x_0
         * \param[out] Cm
         */
        static void Cm(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0, Eigen::Vector3f &Cm);
    };
    
    /** \brief Compressive sensing example using convex optimization as discussed in [1]:
     * [1] G. Kutyniok.
     *     Compressed Sensing: Theory and Applications.
     *     Computing Research Repository, abs/1203.3815, 2012.
     */
    class CompressiveSensing
    {
    public:
        /** \brief Differentiable but non-convex part: \|Ax - y\|_2^2. 
         * \param[in] x
         * \param[in] epsilon
         */
        static float f(const Eigen::MatrixXf &x, const Eigen::MatrixXf &A, 
                const Eigen::MatrixXf &y, float lambda);
        
        /** \brief Derivative of f.
         * \param[in] x
         * \param[out] df_x
         * \param[in] epsilon
         */
        static void df(const Eigen::MatrixXf &x, const Eigen::MatrixXf &A, 
                const Eigen::MatrixXf &y, Eigen::MatrixXf &df_x, float lambda);
        
        /** \brief Non-differentiable but convex part: \|x\|_1. 
         * \param[in] x
         * \param[in] x_0
         * \param[in] Cp
         * \param[in] Cm
         * \param[in] epsilon
         * \param[in] lambda
         */
        static float g(const Eigen::MatrixXf &x);
        
        /** \brief Proximal map of g.
         * \param[in] x
         * \param[in] x_0
         * \param[out] prox_g_x
         * \param[in] Cp
         * \param[in] Cm
         * \param[in] epsilon
         * \param[in] lambda
         * \param[in] alpha
         */
        static void prox_g(const Eigen::MatrixXf &x, Eigen::MatrixXf &prox_g_x, 
                float alpha);
    };
};

////////////////////////////////////////////////////////////////////////////////
// Functionals::Util::convertEigenToOpenCV
////////////////////////////////////////////////////////////////////////////////

void Functionals::Util::convertEigenToOpenCV(const Eigen::MatrixXf &matrix, cv::Mat &image)
{
    image.create(matrix.rows(), matrix.cols(), CV_32FC1);
    
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            image.at<float>(i, j) = matrix(i, j);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::Util::convertOpenCVToEigen
////////////////////////////////////////////////////////////////////////////////

void Functionals::Util::convertOpenCVToEigen(const cv::Mat &image, Eigen::MatrixXf &matrix)
{
    LOG_IF(FATAL, image.type() != CV_32FC1) << "Invalid image type.";
    LOG_IF(FATAL, image.channels() != 1) << "Only one channel supported, use convertOpenCVToEigen_Color instead!";
    
    matrix = Eigen::MatrixXf(image.rows, image.cols);
    
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            matrix(i, j) = image.at<float>(i, j);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::Util::convertEigenToOpenCV
////////////////////////////////////////////////////////////////////////////////

void Functionals::Util::convertEigenToOpenCV_Color(const Eigen::MatrixXf &matrix, cv::Mat &image)
{
    LOG_IF(FATAL, matrix.rows()%3 != 0) << "Eigen matrix expected to encode a color OpenCV matrix, i.e. number of rows needs to be divisible by three!";
    
    image.create(matrix.rows()/3, matrix.cols(), CV_32FC3);
    
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            image.at<cv::Vec3f>(i, j)[0] = matrix(i, j);
            image.at<cv::Vec3f>(i, j)[1] = matrix(image.rows + i, j);
            image.at<cv::Vec3f>(i, j)[2] = matrix(2*image.rows + i, j);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::Util::convertOpenCVToEigen
////////////////////////////////////////////////////////////////////////////////

void Functionals::Util::convertOpenCVToEigen_Color(const cv::Mat &image, Eigen::MatrixXf &matrix)
{
    LOG_IF(FATAL, image.type() != CV_32FC3) << "Invalid image type.";
    LOG_IF(FATAL, image.channels() != 3) << "Only three channels supported, use convertOpenCVToEigen instead!";
    
    matrix = Eigen::MatrixXf(3*image.rows, image.cols);
    
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            matrix(i, j) = image.at<cv::Vec3f>(i, j)[0];
            matrix(image.rows + i, j) = image.at<cv::Vec3f>(i, j)[1];
            matrix(2*image.rows + i, j) = image.at<cv::Vec3f>(i, j)[2];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::Noise::addGaussianAdditive
////////////////////////////////////////////////////////////////////////////////

void Functionals::Denoising::Noise::addGaussianAdditive(const Eigen::MatrixXf &image, 
        float sigma, Eigen::MatrixXf &noisy_image)
{
    std::random_device random;
    std::mt19937 generator(random());
    std::normal_distribution<float> gaussian(0, sigma);
    
    noisy_image = Eigen::MatrixXf::Zero(image.rows(), image.cols());
    for (int i = 0; i < image.rows(); i++)
    {
        for (int j = 0; j < image.cols(); j++)
        {
            noisy_image(i, j) = image(i, j) + gaussian(generator);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::Noise::addSaltPepper
////////////////////////////////////////////////////////////////////////////////

void Functionals::Denoising::Noise::addSaltPepper(const Eigen::MatrixXf &image, 
        float p, Eigen::MatrixXf &noisy_image)
{
    
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::Denoising::Evaluation::computePSNR
////////////////////////////////////////////////////////////////////////////////

float Functionals::Denoising::Evaluation::computePSNR(const Eigen::MatrixXf &image, 
        const Eigen::MatrixXf &denoised_image, float range)
{
    LOG_IF(FATAL, image.rows() != denoised_image.rows() || image.cols() != denoised_image.cols())
            << "Image dimensions do not match!";
    LOG_IF(FATAL, image.rows() == 0 || image.cols() == 0)
            << "Image empty.";
    
    float psnr = 0;
    
    // Compute mse.
    for (int i = 0; i < image.rows(); i++)
    {
        for (int j = 0; j < image.cols(); j++)
        {
            psnr += (image(i, j) - denoised_image(i, j))*(image(i, j) - denoised_image(i, j));
        }
    }
    
    psnr /= image.rows()*image.cols();
    psnr = 10*std::log(range/psnr)/std::log(10);
    
    return psnr;
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::Denoising::f_lorentzianPairwise
////////////////////////////////////////////////////////////////////////////////

float rho(float z, float sigma)
{
    return std::log(1.f + 1.f/2.f*(z*z)/(sigma*sigma));
}

float Functionals::Denoising::f_lorentzianPairwise(const Eigen::MatrixXf &x, float sigma, float lambda)
{
    float f = 0;
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            if (i > 0)
            {
                f += lambda*rho(x(i, j) - x(i - 1, j), sigma);
            }
            
            if (j > 0)
            {
                f += lambda*rho(x(i, j) - x(i, j - 1), sigma);
            }
        }
    }
    
    return f;
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::Denoising::df_lorentzianPairwise
////////////////////////////////////////////////////////////////////////////////

float drho(float z, float sigma)
{
    return 1.f/(1.f + 1.f/2.f*(z*z)/(sigma*sigma))*(z/(sigma*sigma));
}

void Functionals::Denoising::df_lorentzianPairwise(const Eigen::MatrixXf &x, Eigen::MatrixXf &df_x, 
        float sigma, float lambda)
{
    df_x = Eigen::MatrixXf::Zero(x.rows(), x.cols());
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            if (i > 0)
            {
                df_x(i, j) += lambda*drho(x(i, j) - x(i - 1, j), sigma);
            }
            
            if (j > 0)
            {
                df_x(i, j) += lambda*drho(x(i, j) - x(i, j - 1), sigma);
            }
            
            if (i < x.rows() - 1)
            {
                df_x(i, j) -= lambda*drho(x(i + 1, j) - x(i, j), sigma);
            }
            
            if (j < x.cols() - 1)
            {
                df_x(i, j) -= lambda*drho(x(i, j + 1) - x(i, j), sigma);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::Denoising::f_squaredPairwise
////////////////////////////////////////////////////////////////////////////////

float Functionals::Denoising::f_squaredPairwise(const Eigen::MatrixXf &x, float lambda)
{
    float f = 0;
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            if (i > 0)
            {
                f += lambda*(x(i, j) - x(i - 1, j))*(x(i, j) - x(i - 1, j));
            }
            
            if (j > 0)
            {
                f += lambda*(x(i, j) - x(i, j - 1))*(x(i, j) - x(i, j - 1));
            }
        }
    }
    
    return f;
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::Denoising::df_squaredPairwise
////////////////////////////////////////////////////////////////////////////////

void Functionals::Denoising::df_squaredPairwise(const Eigen::MatrixXf &x, Eigen::MatrixXf &df_x, 
        float lambda)
{
    df_x = Eigen::MatrixXf::Zero(x.rows(), x.cols());
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            if (i > 0)
            {
                df_x(i, j) += lambda*2*(x(i, j) - x(i - 1, j));
            }
            
            if (j > 0)
            {
                df_x(i, j) += lambda*2*(x(i, j) - x(i, j - 1));
            }
            
            if (i < x.rows() - 1)
            {
                df_x(i, j) -= lambda*2*(x(i + 1, j) - x(i, j));
            }
            
            if (j < x.cols() - 1)
            {
                df_x(i, j) -= lambda*2*(x(i, j + 1) - x(i, j));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::Denoising::g_absoluteUnary
////////////////////////////////////////////////////////////////////////////////

float Functionals::Denoising::g_absoluteUnary(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0)
{
    float g = 0;
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            g += std::abs(x(i, j) - x_0(i, j));
        }
    }
    
    return g;
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::Denoising::prox_g_absoluteUnary
////////////////////////////////////////////////////////////////////////////////

// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T>
int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

void Functionals::Denoising::prox_g_absoluteUnary(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
        Eigen::MatrixXf &prox_g_x, float alpha)
{
    prox_g_x = Eigen::MatrixXf::Zero(x.rows(), x.cols());
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            prox_g_x(i, j) = std::max(0.f, std::abs(x(i, j) - x_0(i, j)) - alpha)*sign(x(i, j) - x_0(i, j)) + x_0(i, j);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::Denoising::g_squaredUnary
////////////////////////////////////////////////////////////////////////////////

float Functionals::Denoising::g_squaredUnary(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0)
{
    float g = 0;
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            g += (x(i, j) - x_0(i, j))*(x(i, j) - x_0(i, j));
        }
    }
    
    return g;
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::Denoising::prox_g_squaredUnary
////////////////////////////////////////////////////////////////////////////////

void Functionals::Denoising::prox_g_squaredUnary(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
        Eigen::MatrixXf &prox_g_x, float alpha)
{
    prox_g_x = Eigen::MatrixXf::Zero(x.rows(), x.cols());
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            prox_g_x(i, j) = (x(i, j) + alpha*x_0(i, j))/(1 + alpha);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField::f
////////////////////////////////////////////////////////////////////////////////

float Functionals::PhaseField::f(const Eigen::MatrixXf &x, float epsilon)
{
    float f_x = 0;
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            if (i > 0)
            {
                f_x += epsilon*(x(i - 1, j) - x(i, j))*(x(i - 1, j) - x(i, j));
            }
            
            if (j > 0)
            {
                f_x += epsilon*(x(i, j - 1) - x(i, j))*(x(i, j - 1) - x(i, j));
            }
            
            f_x += (1 - x(i, j)*x(i, j))*(1 - x(i, j)*x(i, j))/epsilon;
        }
    }
    
    return f_x;
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField::df
////////////////////////////////////////////////////////////////////////////////

void Functionals::PhaseField::df(const Eigen::MatrixXf &x, Eigen::MatrixXf &df_x, 
        float epsilon)
{
    df_x = Eigen::MatrixXf::Zero(x.rows(), x.cols());
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            df_x(i, j) = - 4*(1 - x(i, j)*x(i, j))*x(i, j)/epsilon;
            
            if (i > 0)
            {
                df_x(i, j) -= 2*epsilon*(x(i - 1, j) - x(i, j));
            }
            
            if (j > 0)
            {
                df_x(i, j) -= 2*epsilon*(x(i, j - 1) - x(i, j));
            }
            
            if (i < x.rows() - 1)
            {
                df_x(i, j) += 2*epsilon*(x(i, j) - x(i + 1, j));
            }
            
            if (j < x.cols() - 1)
            {
                df_x(i, j) += 2*epsilon*(x(i, j) - x(i, j + 1));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField::g
////////////////////////////////////////////////////////////////////////////////

float Functionals::PhaseField::g(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
        float Cp, float Cm, float lambda)
{
    float g_x = 0;
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            g_x += (1 + x(i, j))*(1 + x(i, j))*(x_0(i, j) - Cp)*(x_0(i, j) - Cp)/4
                    + (1 - x(i, j))*(1 - x(i, j))*(x_0(i, j) - Cm)*(x_0(i, j) - Cm)/4;
        }
    }
    
    return lambda*g_x;
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField::prox_g
////////////////////////////////////////////////////////////////////////////////

void Functionals::PhaseField::prox_g(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
        Eigen::MatrixXf &prox_g_x, float Cp, float Cm, float lambda, float alpha)
{
    prox_g_x = Eigen::MatrixXf::Zero(x.rows(), x.cols());
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            float cp = (x_0(i, j) - Cp)*(x_0(i, j) - Cp);
            float cm = (x_0(i, j) - Cm)*(x_0(i, j) - Cm);
            
            prox_g_x(i, j) = (x(i, j) - lambda*alpha*cp/2 + lambda*alpha*cm/2)
                    /(1 + lambda*alpha*cp/2 + lambda*alpha*cm/2);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField::g
////////////////////////////////////////////////////////////////////////////////

float Functionals::PhaseField::g_absolute(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
        float Cp, float Cm, float lambda)
{
    float g_x = 0;
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            float cp = std::abs(x_0(i, j) - Cp);
            float cm = std::abs(x_0(i, j) - Cm);
            
            g_x += std::abs(1 + x(i, j))*cp + std::abs(1 - x(i, j))*cm;
        }
    }
    
    return lambda*g_x;
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField::prox_g
////////////////////////////////////////////////////////////////////////////////

void Functionals::PhaseField::prox_g_absolute(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
        Eigen::MatrixXf &prox_g_x, float Cp, float Cm, float lambda, float alpha)
{
    prox_g_x = Eigen::MatrixXf::Zero(x.rows(), x.cols());
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            float cp = alpha*lambda*std::abs(x_0(i, j) - Cp);//*(x_0(i, j) - Cp);
            float cm = alpha*lambda*std::abs(x_0(i, j) - Cm);//*(x_0(i, j) - Cm);
            
            if (x(i, j) > 1 + (cp + cm))
            {
                prox_g_x(i, j) = -(cp + cm) + x(i, j);
            }
            else if (x(i, j) >= 1 + (cp - cm) && x(i, j) <= 1 + (cp + cm))
            {
                prox_g_x(i, j) = 1;
            }
            else if (x(i, j) > -1 + (cp - cm) && x(i, j) < 1 + (cp - cm))
            {
                prox_g_x(i, j) = (-cp + cm) + x(i, j);
            }
            else if (x(i, j) >= -1 - (cp + cm) && x(i, j) <= -1 + (cp - cm))
            {
                prox_g_x(i, j) = -1;
            }
            else
            {
                LOG_IF(FATAL, x(i, j) >= -1 - (cp + cm)) << "Invalid case in PhaseField::prox_g_absolute!";
                prox_g_x(i, j) = (cp + cm) + x(i, j);
            }
            
//            std::cout << -1 - (cp + cm) << "," << -1 + (cp - cm) << "," << 1 + (cp - cm) << "," << 1 + (cp + cm) << std::endl;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField::Cp
////////////////////////////////////////////////////////////////////////////////

float Functionals::PhaseField::Cp(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0)
{
    float Cp_nom = 0;
    float Cp_denom = 0;
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
//            if (x(i, j) > 0)
//            {
                Cp_nom += (1 + x(i, j))*(1 + x(i, j))*x_0(i, j);
                Cp_denom += (1 + x(i, j))*(1 + x(i, j));
//            }
        }
    }
    
    if (Cp_denom > 0)
    {
        Cp_nom /= Cp_denom;
    }
    
    return Cp_nom;
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField::Cm
////////////////////////////////////////////////////////////////////////////////

float Functionals::PhaseField::Cm(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0)
{
    float Cm_nom = 0;
    float Cm_denom = 0;
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
//            if (x(i, j) <= 0)
//            {
                Cm_nom += (1 - x(i, j))*(1 - x(i, j))*x_0(i, j);
                Cm_denom += (1 - x(i, j))*(1 - x(i, j));
//            }
        }
    }
    
    if (Cm_denom > 0)
    {
        Cm_nom /= Cm_denom;
    }
    
    return Cm_nom;
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField_Color::f
////////////////////////////////////////////////////////////////////////////////

float Functionals::PhaseField_Color::f(const Eigen::MatrixXf &x, float epsilon)
{
    float f_x = 0;
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            if (i > 0)
            {
                f_x += epsilon*(x(i - 1, j) - x(i, j))*(x(i - 1, j) - x(i, j));
            }
            
            if (j > 0)
            {
                f_x += epsilon*(x(i, j - 1) - x(i, j))*(x(i, j - 1) - x(i, j));
            }
            
            f_x += (1 - x(i, j)*x(i, j))*(1 - x(i, j)*x(i, j))/epsilon;
        }
    }
    
    return f_x;
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField_Color::df
////////////////////////////////////////////////////////////////////////////////

void Functionals::PhaseField_Color::df(const Eigen::MatrixXf &x, Eigen::MatrixXf &df_x, 
        float epsilon)
{
    df_x = Eigen::MatrixXf::Zero(x.rows(), x.cols());
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            df_x(i, j) = - 4*(1 - x(i, j)*x(i, j))*x(i, j)/epsilon;
            
            if (i > 0)
            {
                df_x(i, j) -= 2*epsilon*(x(i - 1, j) - x(i, j));
            }
            
            if (j > 0)
            {
                df_x(i, j) -= 2*epsilon*(x(i, j - 1) - x(i, j));
            }
            
            if (i < x.rows() - 1)
            {
                df_x(i, j) += 2*epsilon*(x(i, j) - x(i + 1, j));
            }
            
            if (j < x.cols() - 1)
            {
                df_x(i, j) += 2*epsilon*(x(i, j) - x(i, j + 1));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField_Color::g
////////////////////////////////////////////////////////////////////////////////

float Functionals::PhaseField_Color::g(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
        const Eigen::Vector3f &Cp, const Eigen::Vector3f &Cm, float lambda)
{
    float g_x = 0;
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            float cp = (x_0(i, j) - Cp(0, 0))*(x_0(i, j) - Cp(0, 0))
                    + (x_0(x.rows() + i, j) - Cp(1, 0))*(x_0(x.rows() + i, j) - Cp(1, 0))
                    + (x_0(2*x.rows() + i, j) - Cp(2, 0))*(x_0(2*x.rows() + i, j) - Cp(2, 0));
            cp = cp;
            
            float cm = (x_0(i, j) - Cm(0, 0))*(x_0(i, j) - Cm(0, 0))
                    + (x_0(x.rows() + i, j) - Cm(1, 0))*(x_0(x.rows() + i, j) - Cm(1, 0))
                    + (x_0(2*x.rows() + i, j) - Cm(2, 0))*(x_0(2*x.rows() + i, j) - Cm(2, 0));
            cm = cm;
            
            g_x += (1 + x(i, j))*(1 + x(i, j))*cp/4
                    + (1 - x(i, j))*(1 - x(i, j))*cm/4;
        }
    }
    
    return lambda*g_x;
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField_Color::prox_g
////////////////////////////////////////////////////////////////////////////////

void Functionals::PhaseField_Color::prox_g(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
        Eigen::MatrixXf &prox_g_x, const Eigen::Vector3f &Cp, const Eigen::Vector3f &Cm, float lambda, float alpha)
{
    prox_g_x = Eigen::MatrixXf::Zero(x.rows(), x.cols());
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            float cp = (x_0(i, j) - Cp(0, 0))*(x_0(i, j) - Cp(0, 0))
                    + (x_0(x.rows() + i, j) - Cp(1, 0))*(x_0(x.rows() + i, j) - Cp(1, 0))
                    + (x_0(2*x.rows() + i, j) - Cp(2, 0))*(x_0(2*x.rows() + i, j) - Cp(2, 0));
            cp = cp;
            
            float cm = (x_0(i, j) - Cm(0, 0))*(x_0(i, j) - Cm(0, 0))
                    + (x_0(x.rows() + i, j) - Cm(1, 0))*(x_0(x.rows() + i, j) - Cm(1, 0))
                    + (x_0(2*x.rows() + i, j) - Cm(2, 0))*(x_0(2*x.rows() + i, j) - Cm(2, 0));
            cm = cm;
            
            prox_g_x(i, j) = (x(i, j) - lambda*alpha*cp/2 + lambda*alpha*cm/2)
                    /(1 + lambda*alpha*cp/2 + lambda*alpha*cm/2);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField::g
////////////////////////////////////////////////////////////////////////////////

float Functionals::PhaseField_Color::g_absolute(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
        const Eigen::Vector3f &Cp, const Eigen::Vector3f &Cm, float lambda)
{
    float g_x = 0;
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            float cp = std::abs(x_0(i, j) - Cp(0, 0))
                    + std::abs(x_0(x.rows() + i, j) - Cp(1, 0))
                    + std::abs(x_0(2*x.rows() + i, j) - Cp(2, 0));
            
            float cm = std::abs(x_0(i, j) - Cm(0, 0))
                    + std::abs(x_0(x.rows() + i, j) - Cm(1, 0))
                    + std::abs(x_0(2*x.rows() + i, j) - Cm(2, 0));
            
            g_x += std::abs(1 + x(i, j))*cp + std::abs(1 - x(i, j))*cm;
        }
    }
    
    return lambda*g_x;
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField::prox_g
////////////////////////////////////////////////////////////////////////////////

void Functionals::PhaseField_Color::prox_g_absolute(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0,
        Eigen::MatrixXf &prox_g_x, const Eigen::Vector3f &Cp, const Eigen::Vector3f &Cm, 
        float lambda, float alpha)
{
    prox_g_x = Eigen::MatrixXf::Zero(x.rows(), x.cols());
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            float cp = std::abs(x_0(i, j) - Cp(0, 0))
                    + std::abs(x_0(x.rows() + i, j) - Cp(1, 0))
                    + std::abs(x_0(2*x.rows() + i, j) - Cp(2, 0));
            cp *= alpha*lambda;
            
            float cm = std::abs(x_0(i, j) - Cm(0, 0))
                    + std::abs(x_0(x.rows() + i, j) - Cm(1, 0))
                    + std::abs(x_0(2*x.rows() + i, j) - Cm(2, 0));
            cm *= alpha*lambda;
            
            if (x(i, j) > 1 + (cp + cm))
            {
                prox_g_x(i, j) = -(cp + cm) + x(i, j);
            }
            else if (x(i, j) >= 1 + (cp - cm) && x(i, j) <= 1 + (cp + cm))
            {
                prox_g_x(i, j) = 1;
            }
            else if (x(i, j) > -1 + (cp - cm) && x(i, j) < 1 + (cp - cm))
            {
                prox_g_x(i, j) = (-cp + cm) + x(i, j);
            }
            else if (x(i, j) >= -1 - (cp + cm) && x(i, j) <= -1 + (cp - cm))
            {
                prox_g_x(i, j) = -1;
            }
            else
            {
                LOG_IF(FATAL, x(i, j) >= -1 - (cp + cm)) << "Invalid case in PhaseField::prox_g_absolute!";
                prox_g_x(i, j) = (cp + cm) + x(i, j);
            }
            
//            std::cout << -1 - (cp + cm) << "," << -1 + (cp - cm) << "," << 1 + (cp - cm) << "," << 1 + (cp + cm) << std::endl;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField_Color::Cp
////////////////////////////////////////////////////////////////////////////////

void Functionals::PhaseField_Color::Cp(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0, 
        Eigen::Vector3f &Cp)
{
    Eigen::Vector3f Cp_nom = Eigen::Vector3f::Zero();
    float Cp_denom = 0;
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
//            if (x(i, j) > 0)
//            {
                Cp_nom(0, 0) += (1 + x(i, j))*(1 + x(i, j))*x_0(i, j);
                Cp_nom(1, 0) += (1 + x(i, j))*(1 + x(i, j))*x_0(x.rows() + i, j);
                Cp_nom(2, 0) += (1 + x(i, j))*(1 + x(i, j))*x_0(2*x.rows() + i, j);
                
                Cp_denom += (1 + x(i, j))*(1 + x(i, j));
//            }
        }
    }
    
    if (Cp_denom > 0)
    {
        Cp_nom(0, 0) /= Cp_denom;
        Cp_nom(1, 0) /= Cp_denom;
        Cp_nom(2, 0) /= Cp_denom;
    }
    
    Cp = Cp_nom;
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::PhaseField_Color::Cm
////////////////////////////////////////////////////////////////////////////////

void Functionals::PhaseField_Color::Cm(const Eigen::MatrixXf &x, const Eigen::MatrixXf &x_0, 
        Eigen::Vector3f &Cm)
{
    Eigen::Vector3f Cm_nom = Eigen::Vector3f::Zero();
    float Cm_denom = 0;
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
//            if (x(i, j) <= 0)
//            {
                Cm_nom(0, 0) += (1 - x(i, j))*(1 - x(i, j))*x_0(i, j);
                Cm_nom(1, 0) += (1 - x(i, j))*(1 - x(i, j))*x_0(x.rows() + i, j);
                Cm_nom(2, 0) += (1 - x(i, j))*(1 - x(i, j))*x_0(2*x.rows() + i, j);
                
                Cm_denom += (1 - x(i, j))*(1 - x(i, j));
//            }
        }
    }
    
    if (Cm_denom > 0)
    {
        Cm_nom(0, 0) /= Cm_denom;
        Cm_nom(1, 0) /= Cm_denom;
        Cm_nom(2, 0) /= Cm_denom;
    }
    
    Cm = Cm_nom;
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::CompressiveSensing::f
////////////////////////////////////////////////////////////////////////////////

float Functionals::CompressiveSensing::f(const Eigen::MatrixXf &x, const Eigen::MatrixXf &A,
        const Eigen::MatrixXf &y, float lambda)
{
    Eigen::MatrixXf y_diff = A*x - y;
    return 1.f/lambda*0.5f*y_diff.squaredNorm();
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::CompressiveSensing::df
////////////////////////////////////////////////////////////////////////////////

void Functionals::CompressiveSensing::df(const Eigen::MatrixXf &x, const Eigen::MatrixXf &A,
        const Eigen::MatrixXf &y, Eigen::MatrixXf &df_x, float lambda)
{
    df_x = 1.f/lambda*((A.transpose()*A*x) - A.transpose()*y);
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::CompressiveSensing::g
////////////////////////////////////////////////////////////////////////////////

float Functionals::CompressiveSensing::g(const Eigen::MatrixXf &x)
{
    return x.lpNorm<1>();
}

////////////////////////////////////////////////////////////////////////////////
// Functionals::CompressiveSensing::prox_g
////////////////////////////////////////////////////////////////////////////////

void Functionals::CompressiveSensing::prox_g(const Eigen::MatrixXf &x, 
        Eigen::MatrixXf &prox_g_x, float alpha)
{
    prox_g_x = Eigen::MatrixXf::Zero(x.rows(), x.cols());
    
    for (int i = 0; i < x.rows(); i++)
    {
        for (int j = 0; j < x.cols(); j++)
        {
            if (x(i, j) >= alpha)
            {
                prox_g_x(i, j) = x(i, j) - alpha;
            }
            else if (std::abs(x(i, j)) < alpha)
            {
                prox_g_x(i, j) = 0;
            }
            else
            {
                prox_g_x(i, j) = x(i, j) + alpha;
            }
        }
    }
}

#endif	/* FUNCTIONALS_H */

