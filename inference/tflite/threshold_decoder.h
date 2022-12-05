//
//  threshold_decoder.h
//  Tensorflow-lite
//
//  Created by david8862 on 2022/10/20.
//
//
#ifndef THRESHOLD_DECODER_H_
#define THRESHOLD_DECODER_H_

#include <math.h>
#include <vector>


namespace speech_commands {

// decode raw network output into a relatively linear threshold
// align with listen.py
class ThresholdDecoder {
public:
    // public interfaces
    explicit ThresholdDecoder(std::vector<float> mu_stds, float center_=0.5, int resolution=200, float min_z=-4, float max_z=4) {
        // NOTE: here we only support 1 set of mu/stds config
        float mu = mu_stds[0];
        float std = mu_stds[1];

        min_out = int(mu + min_z * std);
        max_out = int(mu + max_z * std);
        out_range = max_out - min_out;

        // combine _calc_pd() and np.cumsum() in threshold_decoder.py
        calc_cd(mu_stds, resolution);
        center = center_;
    }

    ~ThresholdDecoder() {
    }

    float decode(float raw_output) {
        if (raw_output == 1.0 || raw_output == 0.0) {
            return raw_output;
        }

        float cp;
        if (out_range == 0) {
            cp = int(raw_output > min_out);
        }
        else {
            float ratio = (asigmoid(raw_output) - min_out) / out_range;
            ratio = std::min(std::max(ratio, 0.0f), 1.0f);
            cp = cd[int(ratio * (cd.size() - 1) + 0.5)];
        }

        if (cp < center) {
            return 0.5 * cp / center;
        }
        else{
            return 0.5 + 0.5 * (cp - center) / (1 - center);
        }
    }

private:
    // inverse sigmoid (logit) for scalars"""
    inline float asigmoid(float x) {
        if (x > 0 && x < 1) {
            return -log(1 / x - 1);
        }
        else {
            return -10;
        }
    }

    // probability density function (normal distribution)
    inline float pdf(float x, float mu, float std) {
        if (std == 0) {
            return 0;
        }

        return (1.0 / (std * sqrt(2 * M_PI))) * exp(-pow(x - mu, 2.0) / (2 * pow(std, 2.0)));
    }

    void calc_cd(std::vector<float> mu_stds, int resolution) {
        // NOTE: here we only support 1 set of mu/stds config
        float mu = mu_stds[0];
        float std = mu_stds[1];

        std::vector<float> points;
        int point_num = resolution * out_range;
        float step = float(max_out - min_out) / float(point_num - 1);

        for (int i = 0; i < point_num; i++) {
            // NOTE: not need to sum for all mu_stds, since
            // we only support 1 set of mu/stds config
            float raw_val = min_out + i * step;
            float point_val = pdf(raw_val, mu, std) / resolution;
            points.emplace_back(point_val);
        }

        // do np.cumsum() to get cd from pd
        float sum = 0;
        for (int i = 0; i < point_num; i++) {
            sum += points[i];
            cd.emplace_back(sum);
        }
    }

    int min_out;
    int max_out;
    int out_range;

    std::vector<float> cd;
    float center;
};


}  // namespace speech_commands

#endif  // THRESHOLD_DECODER_H_
