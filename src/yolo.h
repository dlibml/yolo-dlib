#ifndef yolo_h_INCLUDED
#define yolo_h_INCLUDED

#include <dlib/dnn.h>

namespace dlib
{
    struct yolo_options
    {
        // the YOLO ground truth map, where each index corresponds to:
        //     0: objectness
        //     1: cx
        //     2: cy
        //     3: width
        //     4: height
        //     5: label_idx
        using map_type = std::array<matrix<float, 0, 0>, 6>;

        yolo_options() = default;

        yolo_options(const long input_size_, const long downsampling_factor_, const std::vector<std::string>& labels_) :
            input_size(input_size_),
            downsampling_factor(downsampling_factor_),
            labels(labels_)
        {
            DLIB_CASSERT(input_size % downsampling_factor == 0, "input_size is not a multiple of downsampling_factor");
            DLIB_CASSERT(labels.size() > 0, "labels must not be empty");
        }

        long get_input_size() const { return input_size; };

        double get_downsampling_factor() const { return downsampling_factor; };

        long get_grid_size() const { return input_size / downsampling_factor; };

        std::string get_label(const size_t idx) const { return labels[idx]; }

        const std::vector<std::string>& get_labels() const { return labels; }

        // generates a YOLO truth map from an image and its set of bounding boxes
        std::pair<matrix<rgb_pixel>, map_type> generate_map(
            const matrix<rgb_pixel>& image,
            const std::vector<mmod_rect>& boxes
        )
        {
            const long grid_size = get_grid_size();
            map_type label_map;
            for (auto& m : label_map)
            {
                m.set_size(grid_size, grid_size);
                assign_all_pixels(m, -1);
            }

            matrix<rgb_pixel> resized_image(input_size, input_size);
            resize_image(image, resized_image);
            const auto[rf, cf] = get_row_col_scales(image.nr(), image.nc());
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                // reshape the bounding box to the resized_image
                const auto bbox = scale_rectangle(boxes[i].rect, rf, cf);
                // normalized width of the reshaped bounding box
                const auto width = bbox.width() / input_size;
                // normalized height of the reshaped bounding box
                const auto height = bbox.height() / input_size;
                // the center of the original bounding box annotation
                const auto oc = center(boxes[i].rect);
                // the row in the ouput map
                const long r = oc.y() / downsampling_factor;
                // the column in the ouput map
                const long c = oc.x() / downsampling_factor;
                // the y offset from the top left corner
                const double offset_y = static_cast<double>(oc.y()) / downsampling_factor - r;
                // the x offset from the top left corner
                const double offset_x = static_cast<double>(oc.x()) / downsampling_factor - c;
                // the class index
                const size_t idx = std::find(labels.begin(), labels.end(), boxes[i].label) - labels.begin();
                if (idx < labels.size())
                {
                    label_map[0](r, c) = 1;
                    label_map[1](r, c) = offset_x;
                    label_map[2](r, c) = offset_y;
                    label_map[3](r, c) = width;
                    label_map[4](r, c) = height;
                    label_map[5](r, c) = idx;
                }
            }
            return std::make_pair(resized_image, label_map);
        }

        friend inline void serialize(const yolo_options& item, std::ostream& out);
        friend inline void deserialize(yolo_options& item, std::istream& in);

    private:

        long input_size;
        long downsampling_factor;
        std::vector<std::string> labels;

        std::pair<double, double> get_row_col_scales(const long nr, const long nc)
        {
            return std::make_pair(static_cast<double>(input_size) / nr, static_cast<double>(input_size) / nc);
        }

        template <typename rect_type>
        drectangle scale_rectangle(rect_type rect, const double row_factor, const double col_factor)
        {
            return drectangle(
                rect.left() * col_factor,
                rect.top() * row_factor,
                rect.right() * col_factor,
                rect.bottom() * row_factor
            );
        }


    };

    void inline serialize(const yolo_options& item, std::ostream& out)
    {
        int version = 0;
        serialize(version, out);
        serialize(item.input_size, out);
        serialize(item.downsampling_factor, out);
        serialize(item.labels, out);
    }

    void inline deserialize(yolo_options& item, std::istream& in)
    {
        int version = 0;
        deserialize(version, in);
        if (version != 0)
            throw serialization_error("Unexpected version found with deserializing dlib::yolo_options");
        long input_size, downsampling_factor;
        std::vector<std::string> labels;
        deserialize(input_size, in);
        deserialize(downsampling_factor, in);
        deserialize(labels, in);
        item = yolo_options(input_size, downsampling_factor, labels);
    }

    class loss_yolo_
    {
    public:

        typedef yolo_options::map_type training_label_type;
        typedef std::vector<mmod_rect> output_label_type;

        loss_yolo_() {}

        loss_yolo_(yolo_options options_) : options(options_) {}

        const yolo_options& get_options (
        ) const { return options; };

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label(
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter,
            double adjust_threshold = 0
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            DLIB_CASSERT(sub.sample_expansion_factor() == 1, "expansion factor should be 1");
            DLIB_CASSERT(output_tensor.k() >= 6, "YOLO layer requires at least 6 channels");
            DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples(), "num_samples mismatch");

            const float* const out_data = output_tensor.host();

            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        const float objectness = out_data[tensor_index(output_tensor, i, 0, r, c)];
                        if (objectness > adjust_threshold)
                        {
                            const float offset_x = out_data[tensor_index(output_tensor, i, 1, r, c)];
                            const float offset_y = out_data[tensor_index(output_tensor, i, 2, r, c)];
                            const float width = out_data[tensor_index(output_tensor, i, 3, r, c)];
                            const float height = out_data[tensor_index(output_tensor, i, 4, r, c)];
                            float max_value = out_data[tensor_index(output_tensor, i, 5, r, c)];
                            size_t label_idx = 0;
                            for (long k = 6; k < output_tensor.k(); ++k)
                            {
                                const float value = out_data[tensor_index(output_tensor, i, k, r, c)];
                                if (value > max_value)
                                {
                                    max_value = value;
                                    label_idx = k - 5;
                                }
                            }
                            mmod_rect rect;
                            rect.detection_confidence = objectness;
                            rect.label = options.get_label(label_idx);
                            const double df = options.get_downsampling_factor();
                            const auto center = dpoint(offset_x * df + c, offset_y * df + r);
                            rect.rect = rectangle(std::round(center.x() - width / 2 * df),
                                                  std::round(center.y() - height / 2 * df),
                                                  std::round(center.x() + width / 2 * df),
                                                  std::round(center.y() + height / 2 * df));
                            iter->push_back(rect);
                        }
                    }
                }
            }
        }

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const const_label_iterator truth,
            SUBNET& sub
        ) const
        {
            const tensor& output_tensor = sub.get_output();
            tensor& grad = sub.get_gradient_input();
            resizable_tensor alias_grad;

            DLIB_CASSERT(sub.sample_expansion_factor() == 1, "expansion factor should be 1");
            DLIB_CASSERT(input_tensor.num_samples() != 0, "we need at least 1 training sample");
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples(), "num_samples mismatch");
            DLIB_CASSERT(output_tensor.k() >= 6, "YOLO layer requires at least 6 channels");
            DLIB_CASSERT(output_tensor.nr() == grad.nr() &&
                         output_tensor.nc() == grad.nc() &&
                         output_tensor.k() == grad.k(), "grad shape mismatch");

            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                const_label_iterator truth_map_ptr = (truth + i);
                DLIB_CASSERT(truth_map_ptr->at(0).nr() == output_tensor.nr() &&
                             truth_map_ptr->at(0).nc() == output_tensor.nc(),
                             "truth size = " << truth_map_ptr->at(0).nr() << " x " << truth_map_ptr->at(0).nc() << ", "
                             "output size = " << output_tensor.nr() << " x " << output_tensor.nc());
            }

            const float* const out_data = output_tensor.host();
            float* g = grad.host();
            // The loss we output is the average loss of over the minibatch and also over each objectness element
            const double scale = 1.0 / (output_tensor.num_samples() * output_tensor.nr() * output_tensor.nc());

            // objectness classifier (loss binary log per pixel)
            alias_tensor obj_tensor(output_tensor.num_samples(), 1, output_tensor.nr(), output_tensor.nc());
            const size_t obj_offset = 0;
            auto aobj = obj_tensor(output_tensor, obj_offset);
            alias_grad.set_size(obj_tensor.num_samples(), obj_tensor.k(), obj_tensor.nr(), obj_tensor.nc());
            tt::sigmoid(alias_grad, aobj);
            double obj_loss = 0;
            float* ag = alias_grad.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        const float y = (*truth)[0](r, c);
                        const size_t idx = tensor_index(output_tensor, i, 0, r, c);
                        const size_t alias_idx = tensor_index(aobj, i, 0, r, c);
                        if (y > 0.f)
                        {
                            const float temp = log1pexp(-out_data[idx]);
                            obj_loss += y * scale * temp;
                            g[idx] = y * scale * (ag[alias_idx] - 1);
                        }
                        else if (y < 0.f)
                        {
                            const float temp = -(-out_data[idx] - log1pexp(-out_data[idx]));
                            obj_loss += -y * temp;
                            g[idx] = -y * scale * ag[alias_idx];
                        }
                        else
                        {
                            g[idx] = 0.f;
                        }
                    }
                }
            }

            // bounding box regression (loss mean squared per channel and pixel)
            alias_tensor bbr_tensor(output_tensor.num_samples(), 4, output_tensor.nr(), output_tensor.nc());
            double bbr_loss = 0;
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        for (long k = 0; k < bbr_tensor.k(); ++k)
                        {
                            const float y = (*truth)[k](r, c);
                            const size_t idx = tensor_index(output_tensor, i, k + 1, r, c);
                            if (y == -1)
                            {
                                g[idx] = 0.f;
                            }
                            else
                            {
                                const float temp = y - out_data[idx];
                                bbr_loss += temp * temp;
                                g[idx] = -scale * temp;
                            }
                        }
                    }
                }
            }

            // category classifier (loss multiclass log per pixel)
            alias_tensor cls_tensor(output_tensor.num_samples(), output_tensor.k() - 5, output_tensor.nr(), output_tensor.nc());
            const size_t cls_offset = obj_tensor.size() + bbr_tensor.size();
            alias_grad.set_size(cls_tensor.num_samples(), cls_tensor.k(), cls_tensor.nr(), cls_tensor.nc());
            tt::softmax(alias_grad, cls_tensor(output_tensor, cls_offset));
            double cls_loss = 0;
            ag = alias_grad.host();
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        const float y = (*truth)[5](r, c);
                        DLIB_CASSERT(static_cast<long>(y) < cls_tensor.k(), "y: " << y << ", output_tensor.k(): " << output_tensor.k());
                        // if there is no object in this position, we shouldn't update the gradients
                        if (y == -1)
                        {
                            for (long k = 0; k < cls_tensor.k(); ++k)
                            {
                                const size_t idx = tensor_index(output_tensor, i, k + 5, r, c);
                                g[idx] = 0.f;
                            }
                        }
                        else
                        {
                            for (long k = 0; k < cls_tensor.k(); ++k)
                            {
                                const size_t alias_idx = tensor_index(cls_tensor(output_tensor, cls_offset), i, k, r, c);
                                const size_t idx = tensor_index(output_tensor, i, k + 5, r, c);
                                if (k == y)
                                {
                                    cls_loss += -safe_log(ag[alias_idx]);
                                    g[idx] = scale * (ag[alias_idx] - 1);
                                }
                                else
                                {
                                    g[idx] = scale * ag[alias_idx];
                                }
                            }
                        }
                    }
                }
            }
            return (obj_loss + bbr_loss + cls_loss) * scale;
        }

        friend void serialize(const loss_yolo_& item, std::ostream& out)
        {
            serialize("loss_yolo_", out);
            serialize(item.options, out);
        }

        friend void deserialize(loss_yolo_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_yolo_")
                throw serialization_error("Unexpected version found when deserializing dlib::loss_yolo_");
            deserialize(item.options, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_yolo_& item)
        {
            out << "loss_yolo\t (";
            auto& opts = item.options;
            out << "input_size: " << opts.get_input_size();
            out << ", downsampling_factor: " << opts.get_downsampling_factor();
            out << ", grid_size: " << opts.get_grid_size();
            out << ", labels: ";
            const auto labels = opts.get_labels();
            for (size_t i = 0; i < labels.size() - 1; ++i)
            {
                out << labels[i] << ',';
            }
            out << labels[labels.size() - 1];
            out << ')';
            return out;
        }

        friend void to_xml(const loss_yolo_ /*item*/, std::ostream& out)
        {
            out << "<loss_yolo/>";
        }

    private:
        yolo_options options;

        static size_t tensor_index(const tensor& t, long sample, long k, long row, long column)
        {
            // See: https://github.com/davisking/dlib/blob/4dfeb7e186dd1bf6ac91273509f687293bd4230a/dlib/dnn/tensor_abstract.h#L38
            return ((sample * t.k() + k) * t.nr() + row) * t.nc() + column;
        }
    };

    template <typename SUBNET>
    using loss_yolo = add_loss_layer<loss_yolo_, SUBNET>;
}

#endif // yolo_h_INCLUDED
