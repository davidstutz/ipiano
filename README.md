# iPiano

[![Build Status](https://travis-ci.org/davidstutz/ipiano.svg?branch=master)](https://travis-ci.org/davidstutz/ipiano)

iPiano, proposed in [1], is an optimization algorithm combining forward-backward splitting with an inertial force. This repository contains a C++ implementation of iPiano with applications to computer vision tasks. The implementation was submitted as part of a seminar paper [2] written at RWTH Aachen University and advised by [Prof. Berkels](https://www.aices.rwth-aachen.de/people/berkels).

    [1] P. Ochs, Y. Chen, T. Brox, T. Pock.
        iPiano: Inertial Proximal Algorithm for Nonconvex Optimization
        SIAM Journal of Imaging Sciences, colume 7, number 2, 2014.
    [2] D. Stutz.
        Seminar paper "iPiano: Inertial Proximal Algorithm for Non-Convex Optimization"
        https://github.com/davidstutz/seminar-ipiano

**Doxygen documentation now available at [http://davidstutz.github.io/ipiano/](http://davidstutz.github.io/ipiano/).**

## Introduction

Similar to forward-backward splitting, the algorithm tackles problems of the form

    h(x) = f(x) + g(x)

where h and g are functions defined on `\mathbb{R}^n` with different properties. In the most general setting, f is required to be smooth and g is required to be convex. The iterative algorithm is then described by the following update equation:

    x^{(n + 1)} = \prox_{\alpha_n g}(x{(n)} - \nabla f(x{(n)}) + \beta_n (x^{(n)} - x^{(n - 1)}))

where `x^{(n)}` is the `n`-th iterate, `\alpha_n` and `\beta_n` are parameters, `\prox_{\alpha_n g}` is the proximal mapping of g and `\nabla f` is the gradient of f. Details can be found in [1] or the seminar paper corresponding to this implementation [2].

This implementation provides two variants of the algorithm: nmiPiano and iPiano. Ochs et al. [1] proved convergence of the latter one, while the former one is a simplified version. The algorithm can be applied to various tasks, as for example segmentation:

![Example: segmentations obtained for different thresholds, see [2].](screenshot.png?raw=true "Example: segmentations obtained for different thresholds, see [2]")

## Building

The project is based on [CMake](https://cmake.org/), [Boost](http://www.boost.org/doc/libs/1_57_0/doc/html/boost_random/performance.html), [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), [GLog](https://github.com/google/glog) as well as [OpenCV](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation) (tested with OpenCV 2.x) and has been tested on Ubuntu 12.04 and Ubuntu 14.04:

    sudo apt-get install build-essential cmake libeigen3-dev libboost-all-dev libopencv-dev libgoogle-glog-dev libeigen3-dev

The project is then compiled using:

    git clone https://github.com/davidstutz/ipiano
    cd ipiano
    mkdir build
    cd build
    cmake ..
    make
    Scanning dependencies of target signal_denoising_cli
	[ 25%] Building CXX object signal_denoising_cli/CMakeFiles/signal_denoising_cli.dir/main.cpp.o
	Linking CXX executable signal_denoising_cli
	[ 25%] Built target signal_denoising_cli
	Scanning dependencies of target image_denoising_cli
	[ 50%] Building CXX object image_denoising_cli/CMakeFiles/image_denoising_cli.dir/main.cpp.o
	Linking CXX executable image_denoising_cli
	[ 50%] Built target image_denoising_cli
	Scanning dependencies of target phase_field_cli
	[ 75%] Building CXX object phase_field_cli/CMakeFiles/phase_field_cli.dir/main.cpp.o
	Linking CXX executable phase_field_cli
	[ 75%] Built target phase_field_cli
	Scanning dependencies of target phase_field_color_cli
	[100%] Building CXX object phase_field_color_cli/CMakeFiles/phase_field_color_cli.dir/main.cpp.o
	Linking CXX executable phase_field_color_cli
	[100%] Built target phase_field_color_cli

**The modules in `cmake/` may have to be adapted depending on the Eigen and GLog installations!**

The documentation can be built using Doxygen:

    doxygen config.doxygen

## Examples

This repository contains several examples; the usage gets apparent when using the `--help` option.

The signal denoising example demonstrates the usage of iPiano to denoise one-dimensional signals:

    $ ./signal_denoising_cli/signal_denoising_cli --help
    Allowed options:
      --seed arg (=1462728373) seed
      -h [ --help ]            produce help message

The image denoising example takes an input image (e.g. `3096.jpg` from the Berkeley Segmentation Dataset [3]), applies Gaussian noise and denoises the result using iPiano:

    $ ./image_denoising_cli/image_denoising_cli --help
    Allowed options:
      --image arg                    image file
      --sigma arg (=0.0500000007)    noise level
      --lambda arg (=0.100000001)    balancing term between unary and pairwise term
      --iterations arg (=250)        number of iterations
      --eta arg (=1.04999995)        eta, i.e. coefficient to choose local 
                                     Lipschitz constant
      --beta arg (=0.5)              momentum parameter (for nmiPiano) or 
                                     initialization of momentum parameter (iPiano)
      --c1 arg (=9.99999996e-13)     c1 for iPiano
      --c2 arg (=9.99999996e-13)     c2 for iPiano
      --epsilon arg (=0.00100000005) epsilon for iPiano
      -h [ --help ]                  produce help message

    [3] P. Arbelaez, M. Maire, C. Fowlkes and J. Malik.
        Contour Detection and Hierarchical Image Segmentation
        Pattern Analysis and Machine Learning, vol. 33, 2011.

The two phase field examples (phase field on grayscale and phase field on color) demonstrate the usage of iPiano for image segmentation:

    $ ./phase_field_cli/phase_field_cli --help
    Allowed options:
      --image arg           image
      -h [ --help ]         produce help message

The compressive sensing example demonstrates the usage of iPiano for recovering a one-dimensional sparse signal (e.g. see [4] for details):

    $ ./compressive_sensing_cli/compressive_sensing_cli --help
    Allowed options:
      --seed arg (=1462728204) seed
      -h [ --help ]            produce help message

    [4] G. Kutyniok.
        Compressed Sensing: Theory and Applications.
        Computing Research Repository, abs/1203.3815, 2012.

## Usage

Usage can be illustrated using the example of one-dimensional signal denoising as done in `signal_denoising_cli`. The below code example shows the basic usage, also note the discussion below.

    // We randomly sample a signal in the form of a Nx1 Eigen matrix:
    Eigen::MatrixXf signal;
    sampleSignal(signal);
    
    // perturbed_signal is the noisy signal we intend to denoise:
    Eigen::MatrixXf perturbed_signal = signal;
    perturbSignal(perturbed_signal);
    
    // Basic denoising functionals are provided in functionals.h
    // f_forentzianPairwise is a regularizer based on the lorentzian function;
    // i.e. it is differentiable but not convex.
    // g_absoluteUnary is an absolute data term which is convex but not differentiable.
    // The corresponding gradient and proximal mapping are implemented in functionals.h
    // and described in detail in [2].
    std::function<float(const Eigen::MatrixXf&)> bound_f 
            = std::bind(Functionals::Denoising::f_lorentzianPairwise, std::placeholders::_1, sigma, lambda);
    std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&)> bound_df 
            = std::bind(Functionals::Denoising::df_lorentzianPairwise, std::placeholders::_1, std::placeholders::_2, sigma, lambda);
    std::function<float(const Eigen::MatrixXf&)> bound_g
            = std::bind(Functionals::Denoising::g_absoluteUnary, std::placeholders::_1, perturbed_signal);
    std::function<void(const Eigen::MatrixXf&, Eigen::MatrixXf&, float)> bound_prox_g
            = std::bind(Functionals::Denoising::prox_g_absoluteUnary, std::placeholders::_1, perturbed_signal, std::placeholders::_2, std::placeholders::_3);
    
    // The initial iterate (i.e. starting point) will be a random signal.
    Eigen::MatrixXf x_0 = Eigen::MatrixXf::Zero(M, 1);
    perturbSignal(x_0);
    
    // nmiPiano provides the following options, see nmipiano.h or the discussion below.
    nmiPiano::Options nmi_options;
    nmi_options.x_0 = x_0;
    nmi_options.max_iter = 1000;
    nmi_options.L_0m1 = 100.f;
    nmi_options.beta = 0.5;
    nmi_options.eta = 1.05;
    nmi_options.epsilon = 1e-8;
    
    // Both nmiPiano and iPiano provide callbacks to monitor progress.
    // The default_callback writes progress to std::cout, other callbacks to
    // write the progress to file are available in nmipiano.h and ipiano.h
    std::function<void(const nmiPiano::Iteration &iteration)> nmi_bound_callback 
            = std::bind(nmiPiano::default_callback, std::placeholders::_1, 10);
    
    // For initialization, we provide f and g as well as their gradient/proximal mapping,
    // and the callback defined above.
    nmiPiano nmipiano(bound_f, bound_df, bound_g, bound_prox_g, nmi_options, 
            nmi_bound_callback);
    
    // Optimization is done via .optimize expecting two arguments which will be the
    // final iterate as well as the corresponding function value (i.e. h = f + g)
    Eigen::MatrixXf nmi_x_star;
    float nmi_f_x_star;
    nmipiano.optimize(nmi_x_star, nmi_f_x_star);

The corresponding usage of iPiano can be found in `signal_denoising_cli`. The functional to be optimized has to be provided as `std::function` and is expected to have the following form:

    static float f(const Eigen::MatrixXf &x);
    static void df_lorentzianPairwise(const Eigen::MatrixXf &x, Eigen::MatrixXf &df_x);
    static float g_absoluteUnary(const Eigen::MatrixXf &x);
    static void prox_g_absoluteUnary(const Eigen::MatrixXf &x, Eigen::MatrixXf &prox_f_x, float alpha);

Additional parameters are possible but have to be provided through `std::bind`, as for example done with `g_absoluteUnary` which in addition to the above parameters also expects the noisy signal in order to implement the absolute data term:

    // Bind g_absoluteUnary such that the resulting std::function matches the above form!
    // Use std::placeholder::_1, std::placeholder::_2 etc. ...
    std::function<float(const Eigen::MatrixXf&)> bound_g
            = std::bind(Functionals::Denoising::g_absoluteUnary, std::placeholders::_1, perturbed_signal);

nmiPiano provides the following parameters (default values and documentation can also be found in `nmipiano.h`):

* `x_0`: the initial iterate;
* `max_iter`: maximum number of iterations;
* `beta`: fixed `\beta`, i.e. the parameter governing the momentum term/inertial force;
* `eta`: parameter for backtracking to find the local Lipschitz-constant L_n in each iteration, see [1] or [2];
* `L_0m1`: initial estimate of the local Lipschitz-constant; during initialization, the local Lipschitz-constant is estimated around `x_0` and the maximum of the estimate and `L_0m1` is taken;
* `BOUND_L_N`: if true, the local Lipschitz-constant is always bounded below by `L_0m1`;
* `epsilon`: stopping criterion; if `epsilon` is greater than zero, iterations stop if the squared norm of the difference of two consecutive iterates is smaller than `epsilon`.

Choosing these parameters needs some practice; reading [1] and/or [2] is highly recommended. In addition, the parameters strongly depend on the function to be optimized (e.g. if nmiPiano or iPiano do not converge for a given functional, try starting with a higher `L_0m1`).

iPiano **additionally** provides the following parameters (also see `ipiano.h`):

* `beta_0m1`: initial `\beta`, i.e. the parameter governing the momentum term/inertial force - after the first iteration, `\beta` is adapted automatically;
* `c_1`: `c_1` from [1] and [2], usually close to zero is fine, e.g. `c_1 = 1e-6` to `c_1 = 1e-12`;
* `c_2`: same as `c_1`;
* `steps`: governs the resolution of finding appropriate `\alpha_n` and `\beta_n` in each iteration in order to guarantee convergence, see [2]; starting with high `steps > 10000` is recommended - it can later be reduced depending on the functional.

Independent of the function or the used variant, **reading [1] and [2] is highly recommended!**

Further functionals can be found in `functionals.h` and another example can be found in `image_denoising_cli` or `phase_field_color_cli`.

## Examples

This repository contains 4 examples for using nmiPiano and iPiano:

* signal denoising: `signal_denoising_cli`;
* image denoising: `image_denoising_cli`;
* phase field segmentation of grayscale images: `phase_field_cli`;
* and phase field segmentation of color images: `phase_field_color_cli`.

The corresponding functions are detailed in [2]. Some examples are given below:

    cd build
    cmake ..
    make
    # Opencv 4 windows containing the original, noisy and denoised signals!
    ./signal_denoising_cli/signal_denoising_cli
    [0] 79.8015 (Delta_n = 4.23812e-38; L_n = 100; alpha_n = 0.01)
    [10] 50.5144 (Delta_n = 0.0272214; L_n = 653.026; alpha_n = 0.00153133)
    [20] 48.6092 (Delta_n = 0.0286435; L_n = 480.716; alpha_n = 0.00208023)
    [30] 46.641 (Delta_n = 0.0291762; L_n = 463.074; alpha_n = 0.00215948)
    [40] 44.6816 (Delta_n = 0.0290498; L_n = 461.657; alpha_n = 0.00216611)
    [50] 42.7609 (Delta_n = 0.0285756; L_n = 462.503; alpha_n = 0.00216215)
    # ...
    # Applies different functionals for denoising the image with added Gaussian noise:
    ./image_denoising_cli/image_denoising_cli ../3096.jpg
    [0] 16963.7 (Delta_n = 1.65681e-37; L_n = 14.1274; alpha_n = 0.0707846)
    [10] 6109.42 (Delta_n = 1.39415; L_n = 209.762; alpha_n = 0.00476731)
    [20] 6004.02 (Delta_n = 0.078446; L_n = 180.711; alpha_n = 0.00553371)
    [30] 6002.3 (Delta_n = 0.0352952; L_n = 181.006; alpha_n = 0.00552467)
    [40] 6001.82 (Delta_n = 0.0147341; L_n = 181.044; alpha_n = 0.00552353)
    [50] 6001.69 (Delta_n = 0.0121218; L_n = 181.051; alpha_n = 0.00552331)
    # Applies a color phase field to segmentation (iteratively):
    ./phase_field_color_cli/phase_field_color_cli ../3096.jpg
    [0] 4.49466e+07 (Delta_n = 6.45593e-38; L_n = 8.61429; alpha_n = 0.116086)
    [10] 4.48568e+07 (Delta_n = 34.6824; L_n = 8.81607; alpha_n = 0.113429)
    [20] 4.48357e+07 (Delta_n = 19.3647; L_n = 6.77839; alpha_n = 0.147528)
    [30] 4.48274e+07 (Delta_n = 13.7736; L_n = 7.1557; alpha_n = 0.139749)
    [40] 4.48222e+07 (Delta_n = 10.3676; L_n = 7.2935; alpha_n = 0.137108)
    [50] 4.48188e+07 (Delta_n = 9.12528; L_n = 6.54973; alpha_n = 0.152678)
    # ... 
    [0] C_p = 0.43245,0.454655,0.534252; C_m = 0.44876,0.471163,0.554002
    # ...
    [1] C_p = 0.342906,0.36354,0.445063; C_m = 0.488703,0.512195,0.593856
    # ...
    [2] C_p = 0.303374,0.322877,0.399622; C_m = 0.479937,0.503389,0.586667
    # ...

The provided example image is taken from the Berkeley Segmentation Dataset [3].

    [3] P. Arbelaez, M. Maire, C. Fowlkes and J. Malik.
        Contour Detection and Hierarchical Image Segmentation.
        Transactions on Pattern Analysis and Machine Intelligence, volume 33, number 5, 2011.

Example segmentations are shown in the introduction.

## License

Licenses for source code corresponding to:

D. Stutz. **iPiano: Inertial Proximal Algorithm for Non-Convex Optimization.** Seminar Report, Aachen Institute for Advanced Study in Computational Engineering Science, 2016.

Note that the provided image is taken from the [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).

Copyright (c) 2016-2018 David Stutz

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the corresponding papers (see above) in documents and papers that report on research using the Software.
