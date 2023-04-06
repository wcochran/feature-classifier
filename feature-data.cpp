#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <set>
#include <map>
#include <algorithm>
#include "reconstruction.h"
#include <colmap/base/point3d.h>
#include <colmap/base/database.h>
#include <Eigen/Dense>

using KeypointIndex = std::pair<colmap::image_t, colmap::point2D_t>;

bool fileExists(std::string fname) { // note: https://en.wikipedia.org/wiki/Time-of-check_to_time-of-use
    std::ifstream is(fname);
    return is.good();
};

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " SfM_folder feature-labels.csv\n";
        exit(1);
    }
    
    const std::string SfM = argv[1];
    const std::string featureLabelsCSV = argv[2];

    //
    // Open input database and harvest keypoint info.
    //
    std::string databasePath = SfM + "/database.db";

    if (!fileExists(databasePath)) { // note ToC/ToU violation: https://tinyurl.com/2skbsuua
        std::cerr << "Database '" << databasePath << "' does not exist!\n";
        exit(-1);
    }
    colmap::Database database(databasePath);

    std::vector<colmap::Image> images = database.ReadAllImages();

    std::string reconstructionPath = SfM + "/sparse/0";
    if (!fileExists(reconstructionPath + "/cameras.bin") ||
        !fileExists(reconstructionPath + "/images.bin") ||
        !fileExists(reconstructionPath + "/points3D.bin")) {
        std::cerr << "Reconstruction '" << reconstructionPath << "' does not exist!\n";
        exit(-1);
    }
    Reconstruction reconstruction;
    reconstruction.ReadBinary(reconstructionPath);

    std::set<KeypointIndex> keypointsWith3DPoints;
    for (auto&& kv : reconstruction.images) {
        const auto& image = kv.second;
        const colmap::point2D_t numPoints2D = image.NumPoints2D();
        for (colmap::point2D_t point2d_idx = 0; point2d_idx < numPoints2D; point2d_idx++) {
            const colmap::Point2D& point2d = image.Point2D(point2d_idx);
            if (point2d.HasPoint3D())
                keypointsWith3DPoints.insert(std::make_pair(image.ImageId(), point2d_idx));
        }
    }

    std::map<KeypointIndex,size_t> matchCounts;
    std::map<KeypointIndex,size_t> inlierMatchCounts;

    auto increment = [](std::map<KeypointIndex,size_t>& counts, KeypointIndex kp) {
        auto iter = counts.find(kp);
        if (iter == counts.end()) {
            counts[kp] = 1;
        } else {
            iter->second++;
        }
    };

    for (auto&& imageA : images) {
        for (auto&& imageB : images) {
            if (imageA.ImageId() >= imageB.ImageId()) continue;
            if (!database.ExistsMatches(imageA.ImageId(),imageB.ImageId())) continue;
            const colmap::FeatureMatches matches = database.ReadMatches(imageA.ImageId(),imageB.ImageId());
            for (auto&& match : matches) {
                const KeypointIndex keypointA = std::make_pair(imageA.ImageId(),match.point2D_idx1);
                increment(matchCounts, keypointA);
            }
            if (!database.ExistsInlierMatches(imageA.ImageId(),imageB.ImageId())) continue;
            const colmap::TwoViewGeometry twoViewGeometry = database.ReadTwoViewGeometry(imageA.ImageId(),imageB.ImageId());
            const colmap::FeatureMatches& inlierMatches = twoViewGeometry.inlier_matches;
            for (auto&& match : inlierMatches) {
                const KeypointIndex keypointA = std::make_pair(imageA.ImageId(),match.point2D_idx1);
                increment(inlierMatchCounts, keypointA);
            }
        }
    }

    auto descriptorToString = [](const colmap::FeatureDescriptor& descriptor) -> std::string {
        std::stringstream ss;
        ss << std::hex;
        const int n = descriptor.cols();  // 128
        for (int i = 0; i < n; i++) {
            const uint8_t byte = descriptor(0,i);
            ss << std::setw(2) << std::setfill('0') << int(byte);
        }
        return ss.str();
    };

    size_t totalKeypoints = 0;
    for (auto&& image : images) {
        const colmap::image_t imageId = image.ImageId();
        const size_t numKeypoints = database.NumKeypointsForImage(imageId);
        totalKeypoints += numKeypoints;
    }
    std::cout << "total keypoints = " << totalKeypoints << "\n";

    std::ofstream csv(featureLabelsCSV);
    if (!csv.is_open()) {
        std::cerr << "Unable to open '" << featureLabelsCSV << "' for writing!\n";
        exit(-1);
    }

    csv << "N,IMGNAME,IMGID,I,KX,KY,A11,A12,A21,A22,MATCHES,INLIERS,HASPT3D,DESC\n";
    
    size_t n = 0;
    size_t featuresWithMatches = 0;
    size_t featuresWithInlierMatches = 0;
    size_t featuresWith3DPoints = 0;
    for (auto&& image : images) {
        const double progress = 100.0 * double(n)/totalKeypoints;
        std::cout << "\r" << progress << "% keypoints output" << std::flush;
        const colmap::image_t imageId = image.ImageId();
        const std::string name = image.Name();
        const colmap::FeatureKeypoints keypoints = database.ReadKeypoints(imageId);
        const colmap::FeatureDescriptors descriptors = database.ReadDescriptors(imageId);
        const size_t numKeypoints = database.NumKeypointsForImage(imageId);
        for (colmap::point2D_t i = 0; size_t(i) < numKeypoints; i++) {
            const KeypointIndex k = std::make_pair(imageId,i);
            const colmap::FeatureKeypoint& kp = keypoints[i];
            const colmap::FeatureDescriptor& desc = descriptors.row(i);
            const size_t matches = matchCounts[k];
            const size_t inlierMatches = inlierMatchCounts[k];
            const bool hasPoint3D = keypointsWith3DPoints.find(k) != keypointsWith3DPoints.end();
            if (matches > 0) featuresWithMatches++;
            if (inlierMatches > 0) featuresWithInlierMatches++;
            if (hasPoint3D) featuresWith3DPoints++;
            csv << n << "," << name << "," << imageId << "," << i << ","
                << std::fixed << std::setprecision(2)
                << kp.x << "," << kp.y << ","
                << std::setprecision(6)
                << kp.a11 << "," << kp.a12 << ","
                << kp.a21 << "," << kp.a22 << ","
                << matches << "," << inlierMatches << ","
                << std::boolalpha << hasPoint3D << ","
                << descriptorToString(desc) << "\n";
            n++;
        }
    }

    csv.close();

    std::cout << "total features ..............." << totalKeypoints << "\n"
              << "features w matches............" << featuresWithMatches << "\n"
              << "features w inliear matches...." << featuresWithInlierMatches << "\n"
              << "features w 3D Points.........." << featuresWith3DPoints << "\n";
    
    return 0;
}
