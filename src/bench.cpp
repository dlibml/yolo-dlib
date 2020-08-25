#include "darknet.h"

#include <chrono>
#include <resnet.h>

namespace chrono = std::chrono;
using fms = chrono::duration<float, std::milli>;

template <typename net_type>
auto measure(const std::string& name, net_type& net, const int iterations = 100)
{
    dlib::resizable_tensor x;
    dlib::matrix<dlib::rgb_pixel> image(224, 224);
    dlib::assign_all_pixels(image, dlib::rgb_pixel(0, 0, 0));
    std::vector<dlib::matrix<dlib::rgb_pixel>> batch(1, image);
    dlib::running_stats<double> rs;
    net.to_tensor(batch.begin(), batch.end(), x);
    // warmup for 10 iterations
    for (int i = 0; i < 10; ++i)
    {
        net.forward(x);
    }
    for (int i = 0; i < iterations; ++i)
    {
        const auto t0 = chrono::steady_clock::now();
        net.forward(x);
        const auto t1 = chrono::steady_clock::now();
        rs.add(chrono::duration_cast<fms>(t1 - t0).count());
    }
    std::cout << name << " inference: " << rs.mean() << " ms  (#params: ";
    std::cout << dlib::count_parameters(net) << ")\n";
    std::cin.get();
}

auto main(/*const int argc, const char** argv*/) -> int
try
{
    setenv("CUDA_LAUNCH_BLOCKING", "1", 1);
    std::cout << std::fixed << std::setprecision(3);

    {
        resnet::infer_50 resnet50;
        measure("resnet50", resnet50);
    }

    {
        resnet::infer_101 resnet101;
        measure("resnet101", resnet101);
    }


    {
        resnet::infer_152 resnet152;
        measure("resnet152", resnet152);
    }

    {
        darknet::classifier19_infer darknet19;
        measure("darknet19", darknet19);
    }

    {
        darknet::classifier53_infer darknet53;
        measure("darknet53", darknet53);
    }

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
}
