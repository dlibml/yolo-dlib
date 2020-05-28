#ifndef yolo_h_INCLUDED
#define yolo_h_INCLUDED

#include <dlib/dnn.h>

namespace dlib
{
    // Resize the input image to a square image and zero-pad the shortest dimension
    std::pair<double, point> scale_aspect_ratio(
        const matrix<rgb_pixel>& image,
        const int size,
        matrix<rgb_pixel>& square_image
    )
    {
        // figure out if we have to verticallt or horizontally scale the image
        const double ratio = static_cast<double>(image.nr()) / image.nc();
        double scale = 1.0;
        if (ratio > 1)
        {
            scale = static_cast<double>(size) / image.nr();
        }
        else
        {
            scale = static_cast<double>(size) / image.nc();
        }

        // early return if the image has already the requested size
        if (scale == 1 and ratio == 1)
        {
            square_image = image;
            return {1, point(0, 0)};
        }

        // black background
        square_image.set_size(size, size);
        assign_all_pixels(square_image, dlib::rgb_pixel(0, 0, 0));

        // resize the image so that it fits into a size x size image
        auto temp = image;
        resize_image(scale, temp);

        // get the row and column offsets (the padding size)
        const point offset((size - temp.nc()) / 2, (size - temp.nr()) / 2);
        for (long r = 0; r < temp.nr(); r++)
        {
            for (long c = 0; c < temp.nc(); c++)
            {
                square_image(offset.y() + r, offset.x() + c) = temp(r, c);
            }
        }

        return {scale, offset};
    }

    inline double sigmoid(double val)
    {
        return 1.0 / (1.0 + std::exp(-val));
    }

    struct yolo_options
    {
        // the YOLO ground truth map, where each index corresponds to:
        //     0: objectness
        //     1: cx
        //     2: cy
        //     3: height
        //     4: width
        //     5: label_idx
        using map_type = std::array<matrix<float, 0, 0>, 6>;

        yolo_options() = default;

        yolo_options(
            const long input_size_,
            const long downsampling_factor_,
            const std::vector<std::vector<mmod_rect>>& boxes
        ) :
            input_size(input_size_),
            downsampling_factor(downsampling_factor_)
        {
            DLIB_CASSERT(input_size % downsampling_factor == 0, "input_size is not a multiple of downsampling_factor");
            std::set<std::string> labels_set;
            for (const auto& v : boxes)
            {
                for (const auto& box : v)
                {
                    labels_set.insert(box.label);
                }
            }
            for (const auto& label : labels_set)
            {
                labels.push_back(label);
            }
            DLIB_CASSERT(labels.size() > 0, "labels must not be empty");
        }

        long get_input_size() const { return input_size; };

        double get_downsampling_factor() const { return downsampling_factor; };

        long get_grid_size() const { return input_size / downsampling_factor; };

        std::string get_label(const size_t idx) const { return labels[idx]; }

        const std::vector<std::string>& get_labels() const { return labels; }

        // generates a YOLO truth map from an image and its set of bounding boxes
        std::pair<matrix<rgb_pixel>, map_type> generate_map
        (
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
            const auto[scale, offset] = scale_aspect_ratio(image, input_size, resized_image);
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                // adjust the bounding box
                const auto scaled_bbox = scale_rect(boxes[i].rect, scale);
                auto bbox = translate_rect(scaled_bbox, offset);
                const rectangle image_rect(0, 0, input_size - 1, input_size - 1);
                // the center of the adjusted bounding box annotation
                const auto oc = center(bbox);
                // only generate maps for boxes whose center is inside of the image
                if (image_rect.contains(oc))
                {
                    // normalized height of the reshaped bounding box
                    const auto height = static_cast<double>(bbox.height()) / input_size;
                    // normalized width of the reshaped bounding box
                    const auto width = static_cast<double>(bbox.width()) / input_size;
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
                        DLIB_CASSERT(0 <= offset_x && offset_x <= 1, "wrong offset_x: " << offset_x);
                        label_map[1](r, c) = offset_x;
                        DLIB_CASSERT(0 <= offset_y && offset_y <= 1, "wrong offset_y: " << offset_y);
                        label_map[2](r, c) = offset_y;
                        label_map[3](r, c) = std::sqrt(height > 1 ? 1 : height);
                        label_map[4](r, c) = std::sqrt(width > 1 ? 1 : width);
                        label_map[5](r, c) = idx;
                    }
                }
            }
            return std::make_pair(resized_image, label_map);
        }

        matrix<rgb_pixel> overlay_map
        (
            const matrix<rgb_pixel>& image,
            const std::vector<mmod_rect>& boxes
        )
        {
            auto[resized_image, yolo_map] = generate_map(image, boxes);
            for (long r = 0; r < get_grid_size(); ++r)
            {
                for (long c = 0; c < get_grid_size(); ++c)
                {

                    if (yolo_map[0](r, c) == 1)
                    {
                        // center
                        point center((c + yolo_map[1](r, c)) * downsampling_factor, (r + yolo_map[2](r, c)) * downsampling_factor);
                        draw_solid_circle(resized_image, center, 3, rgb_pixel(255, 255, 255));
                        draw_solid_circle(resized_image, center, 2, rgb_pixel(0, 0, 0));
                        // paint the grid
                        for (long gr = 0; gr < downsampling_factor; ++gr)
                        {
                            for (long gc = 0; gc < downsampling_factor; ++gc)
                            {
                                rgb_alpha_pixel p(0, 255, 0, 64);
                                assign_pixel(resized_image(r * downsampling_factor + gr, c * downsampling_factor + gc), p);
                            }
                        }
                        // bounding boxes
                        const long h = std::pow(yolo_map[3](r, c), 2) * input_size;
                        const long w = std::pow(yolo_map[4](r, c), 2) * input_size;
                        draw_rectangle(
                            resized_image,
                            rectangle(center.x() - w / 2, center.y() - h / 2, center.x() + w / 2, center.y() + h / 2),
                            rgb_pixel(255, 0, 0),
                            1);
                    }
                    // grid
                    draw_rectangle(
                        resized_image,
                        rectangle(
                            c * downsampling_factor,
                            r * downsampling_factor,
                            c * downsampling_factor + downsampling_factor,
                            r * downsampling_factor + downsampling_factor),
                        rgb_pixel(0, 0, 0),
                        1);
                }
            }
            return resized_image;
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
        deserialize(item.input_size, in);
        deserialize(item.downsampling_factor, in);
        deserialize(item.labels, in);
        DLIB_CASSERT(item.input_size % item.downsampling_factor == 0, "input_size is not a multiple of downsampling_factor");
    }

    bool operator<(const mmod_rect& a, const mmod_rect& b) { return a.detection_confidence < b.detection_confidence; }

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
        void to_label (
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
                std::vector<mmod_rect> candidates;
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        const float objectness = out_data[tensor_index(output_tensor, i, 0, r, c)];
                        if (objectness > adjust_threshold)
                        {
                            const float offset_x = sigmoid(out_data[tensor_index(output_tensor, i, 1, r, c)]);
                            const float offset_y = sigmoid(out_data[tensor_index(output_tensor, i, 2, r, c)]);
                            const float height = sigmoid(out_data[tensor_index(output_tensor, i, 3, r, c)]);
                            const float width = sigmoid(out_data[tensor_index(output_tensor, i, 4, r, c)]);
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
                            std::cout << "offset_x: " << offset_x << '\n';
                            std::cout << "offset_y: " << offset_y << '\n';
                            std::cout << "height: " << height << '\n';
                            std::cout << "width: " << width << '\n';
                            rect.detection_confidence = objectness;
                            rect.label = options.get_label(label_idx);
                            const double df = options.get_downsampling_factor();
                            const auto center = dpoint((c + offset_x) * df, (r + offset_y) * df);
                            rect.rect = centered_rect(center, width * df, height * df);
                            candidates.push_back(rect);
                        }
                    }
                }
                std::sort(candidates.rbegin(), candidates.rend());
                // TODO: perform NMS
                *iter++ = std::move(candidates);
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

            DLIB_CASSERT(sub.sample_expansion_factor() == 1, "expansion factor should be 1");
            DLIB_CASSERT(input_tensor.num_samples() != 0, "we need at least 1 training sample");
            DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples(), "num_samples mismatch");
            DLIB_CASSERT(output_tensor.k() >= 6, "YOLO layer requires at least 6 channels");
            DLIB_CASSERT(output_tensor.nr() == options.get_grid_size(), "wrong output nr: " << output_tensor.nr());
            DLIB_CASSERT(output_tensor.nc() == options.get_grid_size(), "wrong output nc: " << output_tensor.nc());
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

            const float* out_data = output_tensor.host();
            float* g = grad.host_write_only();

            // The loss we output is the average loss of over the minibatch and also over each objectness element
            const double scale = 1.0 / (output_tensor.num_samples() * output_tensor.nr() * output_tensor.nc());
            const double lambda_noobj = 0.5;
            const double lambda_coord = 5.0;

            // extract the subtensors from the output tensor
            resizable_tensor obj_tensor(output_tensor.num_samples(), 1, output_tensor.nr(), output_tensor.nc());
            resizable_tensor bbr_tensor(output_tensor.num_samples(), 4, output_tensor.nr(), output_tensor.nc());
            resizable_tensor cls_tensor(output_tensor.num_samples(), output_tensor.k() - 5, output_tensor.nr(), output_tensor.nc());
            float* obj_data = obj_tensor.host();
            float* bbr_data = bbr_tensor.host();
            float* cls_data = cls_tensor.host();

            int obj_idx = 0, bbr_idx = 0, cls_idx = 0;
            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        for (long k = 0; k < output_tensor.k(); ++k)
                        {
                            const size_t idx = tensor_index(output_tensor, i, k, r, c);
                            if (k == 0)
                            {
                                obj_data[obj_idx++] = out_data[idx];
                            }
                            else if (k < 5)
                            {
                                bbr_data[bbr_idx++] = out_data[idx];
                            }
                            else
                            {
                                cls_data[cls_idx++] = out_data[idx];
                            }
                        }
                    }
                }
            }
            obj_tensor.device();
            bbr_tensor.device();
            cls_tensor.device();

            // --------------------------------------------------------------------------------- //
            double loss_obj = 0;
            double loss_bbr = 0;
            double loss_cls = 0;

            resizable_tensor temp_obj_tensor(obj_tensor.num_samples(), obj_tensor.k(), obj_tensor.nr(), obj_tensor.nc());
            tt::sigmoid(temp_obj_tensor, obj_tensor);
            resizable_tensor temp_bbr_tensor(bbr_tensor.num_samples(), bbr_tensor.k(), bbr_tensor.nr(), bbr_tensor.nc());
            tt::sigmoid(temp_bbr_tensor, bbr_tensor);
            resizable_tensor temp_cls_tensor(cls_tensor.num_samples(), cls_tensor.k(), cls_tensor.nr(), cls_tensor.nc());
            tt::softmax(temp_cls_tensor, cls_tensor);

            float* temp_obj_data = temp_obj_tensor.host();
            float* temp_bbr_data = temp_bbr_tensor.host();
            float* temp_cls_data = temp_cls_tensor.host();

            for (long i = 0; i < output_tensor.num_samples(); ++i)
            {
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        // objectness classifier (loss binary log per pixel)
                        float y = (*truth)[0](r, c);
                        const size_t idx = tensor_index(output_tensor, i, 0, r, c);
                        obj_idx = tensor_index(obj_tensor, i, 0, r, c);
                        if (y > 0.f)
                        {
                            const float temp = log1pexp(-out_data[idx]);
                            loss_obj += y * scale * temp;
                            g[idx] = y * scale * (temp_obj_data[obj_idx] - 1);
                        }
                        else if (y < 0.f)
                        {
                            const float temp = -(-out_data[idx] - log1pexp(-out_data[idx]));
                            loss_obj += -y * temp * scale * lambda_noobj;
                            g[idx] = -y * temp_obj_data[obj_idx] * scale * lambda_noobj;
                        }
                        else
                        {
                            g[idx] = 0.f;
                        }

                        // bounding box regression (loss mean squared per channel and pixel)
                        for (long k = 1; k < 5; ++k)
                        {
                            const float y = (*truth)[k].operator()(r, c);
                            bbr_idx = tensor_index(bbr_tensor, i, k - 1, r, c);
                            const size_t idx = tensor_index(output_tensor, i, k, r, c);
                            if (y == -1)
                            {
                                g[idx] = 0.f;
                            }
                            else
                            {
                                float temp1;
                                if (k < 3) // center offset
                                    temp1 = y - temp_bbr_data[bbr_idx];
                                else // height and width
                                    temp1 = y - std::sqrt(temp_bbr_data[bbr_idx]);
                                const float temp2 = temp1 * scale * lambda_coord;
                                loss_bbr += temp1 * temp2;
                                g[idx] = -temp2;
                            }
                        }

                        // category classifier (loss multiclass log per pixel)
                        y = (*truth)[5](r, c);
                        DLIB_CASSERT(static_cast<long>(y) < cls_tensor.k(), "y: " << y << ", cls_tensor.k(): " << cls_tensor.k());
                        for (long k = 5; k < output_tensor.k(); ++k)
                        {
                            cls_idx = tensor_index(cls_tensor, i, k - 5, r, c);
                            const size_t idx = tensor_index(output_tensor, i, k, r, c);
                            if (k == y)
                            {
                                loss_cls += -safe_log(temp_cls_data[cls_idx]) * scale;
                                g[idx] = scale * (temp_cls_data[cls_idx] - 1);
                            }
                            else if (y == -1)
                            {
                                g[idx] = 0.f;
                            }
                            else
                            {
                                g[idx] = scale * temp_cls_data[cls_idx];
                            }
                        }
                    }
                }
            }

            // std::cout << "loss_obj: " << loss_obj * lambda_obj * scale;
            // std::cout << ", loss_bbr: " << loss_bbr * lambda_bbr * scale;
            // std::cout << ", loss_cls: " << loss_cls * lambda_cls * scale;
            // std::cout << std::endl;
            // return loss_obj * lambda_obj * scale;
            return loss_obj + loss_bbr + loss_cls;
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
            const auto& opts = item.options;
            out << "input_size: " << opts.get_input_size();
            out << ", downsampling_factor: " << opts.get_downsampling_factor();
            out << ", grid_size: " << opts.get_grid_size();
            const auto labels = opts.get_labels();
            out << ", " << labels.size() << " labels: ";
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

