#include "darknet.h"

#include <chrono>
#include <resnet.h>

namespace chrono = std::chrono;
using fms = chrono::duration<float, std::milli>;

template <typename net_type>
auto measure(const std::string& name, net_type& net, const int iterations = 100)
{
    dlib::set_all_bn_inputs_no_bias(net);
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
        resnet::train_50 resnet50;
        measure(" resnet50", resnet50);
    }

    {
        resnet::train_101 resnet101;
        measure("resnet101", resnet101);
    }


    {
        resnet::train_152 resnet152;
        measure("resnet152", resnet152);
    }

    {
        darknet::classifier19_train darknet19;
        measure("darknet19", darknet19);
    }

    {
        darknet::classifier53_train darknet53;
        measure("darknet53", darknet53);
    }

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
}

// Benchmark results on NVIDIA RTX 5000, input 1,3,224,224, with and without bias

// resnet50 inference: 6.935 ms  (#params: 22803176), VRAM: 889 MiB
// resnet50 inference: 6.490 ms  (#params: 22780456), VRAM: 875 MiB

// resnet101 inference: 11.690 ms  (#params: 41821416), VRAM: 1078 MiB
// resnet101 inference: 10.577 ms  (#params: 41772584), VRAM: 1064 MiB

// resnet152 inference: 16.922 ms  (#params: 57488104), VRAM: 1262 MiB
// resnet152 inference: 15.406 ms  (#params: 57416232), VRAM: 1248 MiB

// darknet19 inference: 3.723 ms  (#params: 20849576), VRAM: 944 MiB
// darknet19 inference: 3.283 ms  (#params: 20842376), VRAM: 934 MiB

// darknet53 inference: 9.451 ms  (#params: 41627784), VRAM: 1116 MiB
// darknet53 inference: 8.485 ms  (#params: 41609928), VRAM: 1096 MiB
