#include "resnet.h"
#include "yolo.h"

#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

namespace detector
{
    using namespace dlib;
    template <template <typename> class BN>
    using net_type = loss_yolo<
        con<5 + 2, 1, 1, 1, 1,
        typename resnet::def<BN>::template backbone_18<
        input_rgb_image_sized<608>
    >>>;

    using train = net_type<bn_con>;
    using infer = net_type<affine>;
}

auto main(const int argc, const char** argv) -> int
try
{
    std::vector<std::string> label_names = {"person", "sunglasses"};
    dlib::yolo_options options(608, 32, label_names);
    detector::train net(options);
    std::vector<dlib::matrix<dlib::rgb_pixel>> images;
    std::vector<std::vector<dlib::mmod_rect>> boxes;
    std::vector<dlib::yolo_options::map_type> labels;
    dlib::load_image_dataset(images, boxes, "sunglasses.xml");
    for (size_t i = 0; i < images.size(); ++i)
    {
        auto[image, label_map] = options.generate_map(images[i], boxes[i]);
        images[i] = image;
        labels.push_back(label_map);
    }
    // net(images[0]);
    // std::cout << net << '\n';

    auto trainer = dlib::dnn_trainer(net);
    trainer.be_verbose();
    trainer.set_mini_batch_size(16);
    std::cout << trainer << '\n';
    trainer.train(images, labels);
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
}
