#include "yolo.h"
#include "resnet_backbone.h"

#include <iostream>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <resnet.h>

namespace detector
{
    using namespace dlib;
    template <template <typename> class BN, template <typename> class ACT>
    using net_type = loss_yolo<
        con<5 + 2, 1, 1, 1, 1,
        typename resnet_backbone::def<BN, ACT>::template backbone_50<
        input_rgb_image
    >>>;

    using train = net_type<bn_con, mish>;
    using infer = net_type<affine, mish>;
}

auto main(const int argc, const char** argv) -> int
try
{

    std::vector<dlib::matrix<dlib::rgb_pixel>> images;
    std::vector<std::vector<dlib::mmod_rect>> boxes;
    dlib::load_image_dataset(images, boxes, "sunglasses.xml");

    dlib::yolo_options options(608, 32, boxes);
    detector::train net(options);
    dlib::deserialize("resnet50_pretrained_backbone.dnn") >> net.subnet().subnet();

    std::vector<dlib::yolo_options::map_type> labels;
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
    trainer.set_learning_rate(000.1);
    trainer.set_iterations_without_progress_threshold(100000);
    trainer.set_mini_batch_size(4);
    std::cout << trainer << '\n';
    dlib::rand rnd;
    while(true)
    {
        decltype(images) minibatch_images;
        decltype(labels) minibatch_labels;
        while(minibatch_images.size() < 4)
        {
            auto idx = rnd.get_random_32bit_number() % images.size();
            minibatch_images.push_back(images[idx]);
            minibatch_labels.push_back(labels[idx]);
        }
        trainer.train_one_step(minibatch_images, minibatch_labels);
    }
    // trainer.train(images, labels);
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
}
