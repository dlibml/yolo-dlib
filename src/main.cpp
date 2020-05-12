#include "yolo.h"
#include "resnet_backbone.h"

#include <iostream>
#include <dlib/data_io.h>
#include <resnet.h>

#undef VISUAL
#include <dlib/gui_widgets.h>

namespace detector
{
    using namespace dlib;
    template <template <typename> class BN, template <typename> class ACT>
    using net_type = loss_yolo<
        con<1, 1, 1, 1, 1,
        typename resnet_backbone::def<BN, ACT>::template backbone_50<
        input_rgb_image
    >>>;

    using train = net_type<bn_con, mish>;
    using infer = net_type<affine, mish>;
}

struct training_sample
{
    dlib::matrix<dlib::rgb_pixel> image;
    std::vector<dlib::mmod_rect> boxes;
    dlib::yolo_options::map_type yolo_map;
};

auto main(const int argc, const char** argv) -> int
try
{
    if (argc > 1)
    {
        dlib::image_window win;
        detector::train net;
        dlib::deserialize("./yolo-resnet50-backbone.dnn") >> net;
        std::cout << net << '\n';
        dlib::matrix<dlib::rgb_pixel> image, input_image;
        for (int i = 1; i < argc; ++i)
        {
            dlib::load_image(image, argv[i]);
            dlib::scale_aspect_ratio(image, net.loss_details().get_options().get_input_size(), input_image);
            win.set_image(input_image);
            const auto detections = net.process(input_image, -1);
            std::cout << "found " << detections.size() << " detections\n";
            for (const auto& det : detections)
            {
                std::cout << det.label << ", " << det.detection_confidence << ", " << det.rect << '\n';
                win.add_overlay(det.rect, dlib::rgb_pixel(0, 255, 0), det.label);
            }
            std::cin.get();
            win.clear_overlay();
        }
    }
    else
    {
        std::vector<dlib::matrix<dlib::rgb_pixel>> images;
        std::vector<std::vector<dlib::mmod_rect>> bboxes;
        dlib::load_image_dataset(images, bboxes, "./pascal.xml");
        std::cout << "image dataset loaded: " << images.size() << " images\n";

        dlib::yolo_options options(608, 32, bboxes);
        detector::train net(options);
        net.subnet().layer_details().set_num_filters(options.get_labels().size() + 5);
        dlib::deserialize("resnet50_pretrained_backbone.dnn") >> net.subnet().subnet();
        dlib::set_all_learning_rate_multipliers(net.subnet().subnet(), 0.01);

        dlib::pipe<training_sample> training_data(1000);

#ifdef VISUAL
        dlib::image_window win;
        auto data_loader = [&win, &training_data, &images, &bboxes, &options](time_t seed)
#else
        auto data_loader = [&training_data, &images, &bboxes, &options](time_t seed)
#endif
        {
            dlib::rand rnd(time(nullptr) + seed);

            const long min_size = options.get_downsampling_factor() * 1.5;
            const long chip_dim = options.get_input_size();
            dlib::random_cropper cropper;
            cropper.set_seed(time(nullptr) + seed);
            cropper.set_chip_dims(chip_dim, chip_dim);
            cropper.set_translate_amount(0.1);
            cropper.set_background_crops_fraction(0);
            cropper.set_randomly_flip(true);
            cropper.set_max_rotation_degrees(30);
            cropper.set_min_object_size(min_size, min_size);
            cropper.set_max_object_size(0.9);

            training_sample sample;
            while (training_data.is_enabled())
            {
                const auto idx = rnd.get_random_32bit_number() % images.size();
                const auto& image = images[idx];
                const auto& boxes = bboxes[idx];
                cropper(image, boxes, sample.image, sample.boxes);
                const auto temp = options.generate_map(sample.image, sample.boxes);
                sample.image = temp.first;
                sample.yolo_map = temp.second;
                dlib::disturb_colors(sample.image, rnd);

#ifdef VISUAL
                win.set_image(sample.image);
                for (const auto& box : sample.boxes)
                {
                    win.add_overlay(box.rect, dlib::rgb_pixel(0, 255, 0), box.label);
                }
                std::cin.get();
                win.clear_overlay();
                win.set_image(options.overlay_map(sample.image, sample.boxes));
                std::cin.get();
#endif

                training_data.enqueue(sample);
            }
        };

        std::vector<std::thread> data_loaders;
        for (int i = 0; i < 1; ++i)
        {
            data_loaders.emplace_back([&data_loader, i](){ data_loader(i + 1); });
        }

        auto trainer = dlib::dnn_trainer(net);
        trainer.be_verbose();
        trainer.set_learning_rate(0.01);
        trainer.set_iterations_without_progress_threshold(10000);
        trainer.set_mini_batch_size(4);
        std::cout << trainer << '\n';
        std::vector<dlib::matrix<dlib::rgb_pixel>> minibatch_images;
        std::vector<dlib::yolo_options::map_type> minibatch_labels;
        while(trainer.get_learning_rate() > 1e-6)
        {
            minibatch_images.clear();
            minibatch_labels.clear();
            training_sample sample;
            while (minibatch_images.size() < trainer.get_mini_batch_size())
            {
                training_data.dequeue(sample);
                minibatch_images.push_back(sample.image);
                minibatch_labels.push_back(sample.yolo_map);
            }
            trainer.train_one_step(minibatch_images, minibatch_labels);
        }

        training_data.disable();
        for (auto& dl : data_loaders)
        {
            dl.join();
        }
        trainer.get_net();
        net.clean();
        dlib::serialize("yolo-resnet50-backbone.dnn") << net;
    }
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
}
