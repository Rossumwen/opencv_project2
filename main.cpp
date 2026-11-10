#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    vector<string> img_names = {"images1.jpg", "images2.jpg", "images3.jpg"};

    for (int idx = 0; idx < img_names.size(); idx++) {
        cout << "正在处理: " << img_names[idx] << endl;
        Mat src_full = imread(img_names[idx]);
        if (src_full.empty()) continue;

        string prefix = to_string(idx + 1);
        //我真没招了，下面反光直接剪掉吧
        Rect crop_rect(0, 0, src_full.cols, src_full.rows * 0.80);
        Mat src = src_full(crop_rect).clone();

        Mat hsv, mask;
        cvtColor(src, hsv, COLOR_BGR2HSV);

        //老大，这调成青的比赛真没问题吗
        inRange(hsv, Scalar(35, 80, 150), Scalar(100, 255, 255), mask);

        Mat kernel_horiz = getStructuringElement(MORPH_RECT, Size(3, 1));
        morphologyEx(mask, mask, MORPH_OPEN, kernel_horiz);

        Mat kernel_vert = getStructuringElement(MORPH_RECT, Size(1, 5));
        morphologyEx(mask, mask, MORPH_CLOSE, kernel_vert);

        imwrite(prefix + "_02_binary.jpg", mask);

        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<RotatedRect> lightBars;
        Mat debug_candidates = src.clone();

        for (auto& cnt : contours) {
            double area = contourArea(cnt);
            if (area < 15 || area > 2000) continue; 

            RotatedRect rect = minAreaRect(cnt);
            float w = rect.size.width;
            float h = rect.size.height;
            if (w == 0 || h == 0) continue;

            float ratio = max(w, h) / min(w, h);
            if (ratio < 1.5 || ratio > 20.0) continue; 

            lightBars.push_back(rect);

            Point2f pts[4];
            rect.points(pts);
            for (int i = 0; i < 4; i++) {
                line(debug_candidates, pts[i], pts[(i + 1) % 4], Scalar(0, 255, 0), 2);
            }
        }

        imwrite(prefix + "_03_candidates.jpg", debug_candidates);

        vector<RotatedRect> armors;
        Mat debug_armors = src.clone();

        for (size_t i = 0; i < lightBars.size(); i++) {
            for (size_t j = i + 1; j < lightBars.size(); j++) {
                float h1 = max(lightBars[i].size.width, lightBars[i].size.height);
                float h2 = max(lightBars[j].size.width, lightBars[j].size.height);
                float avgLen = (h1 + h2) / 2.0;

                if (abs(h1 - h2) / max(h1, h2) > 0.5) continue;

                Point2f c1 = lightBars[i].center;
                Point2f c2 = lightBars[j].center;

                if (abs(c1.y - c2.y) > avgLen * 0.8) continue;

                float distance = norm(c1 - c2);
                if (distance < avgLen * 1.0 || distance > avgLen * 5.0) continue;

                Point2f center = (c1 + c2) / 2.0;
                RotatedRect armor(center, Size2f(distance, avgLen), (lightBars[i].angle + lightBars[j].angle) / 2.0);
                armors.push_back(armor);

                Point2f pts[4];
                armor.points(pts);
                for (int k = 0; k < 4; k++) {
                    line(debug_armors, pts[k], pts[(k + 1) % 4], Scalar(0, 0, 255), 2); 
                }
            }
        }

        imwrite(prefix + "_04_paired.jpg", debug_armors);
        imwrite(prefix + "_01_original.jpg", src);

        cout << "-> 图片 " << prefix << " 处理完毕。找到 " << lightBars.size() << " 个灯条，配对出 " << armors.size() << " 个装甲板。" << endl;
    }
    return 0;
}