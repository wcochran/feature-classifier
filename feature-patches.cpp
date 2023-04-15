#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "rectpack2D/finders_interface.h"

std::vector<std::string> split(const std::string& str, char delim) {
    std::vector<std::string> strings;
    size_t start;
    size_t end = 0;
    while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
        end = str.find(delim, start);
        strings.push_back(str.substr(start, end - start));
    }
    return strings;
}

std::string trim(const std::string& str,  const std::string& whitespace = " \t\n") {
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos)
        return ""; // no content

    const auto strEnd = str.find_last_not_of(whitespace);
    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}

// N,IMGNAME,IMGID,I,KX,KY,A11,A12,A21,A22,MATCHES,INLIERS,HASPT3D,DESC
// 4,IMG_0013.JPG,1,4,826.35,532.91,507.408661,0.000000,66.599098,108.817360,1,1,false,<desc>

struct Feature {
    int num;
    std::string imageName;
    int index;
    Eigen::Vector2f keypoint;
    Eigen::Matrix2f A;
    int matches;
    int inlierMatches;
    bool hasPoint3D;
    std::string descriptorString;
};

std::vector<Feature> readFeatures(std::string path) {
    std::vector<Feature> features;
    std::ifstream is(path);
    if (!is.is_open()) {
        std::cerr << "Unable to open '" << path << "'\n";
        exit(-1);
    }
    std::string line;
    while (std::getline(is, line)) {
        if (line.length() <= 0 || line.at(0) == '#') continue;
        const std::vector<std::string> str = split(line, ',');
        if (str.size() != 14) break;
        Feature feature;
        try {
            feature.num = std::stoi(trim(str[0]));
            feature.imageName = trim(str[1]);
            feature.index = std::stoi(trim(str[3]));
            feature.keypoint(0) = std::stof(trim(str[4]));
            feature.keypoint(1) = std::stof(trim(str[5]));
            feature.A(0,0) = std::stof(trim(str[6]));
            feature.A(0,1) = std::stof(trim(str[7]));
            feature.A(1,0) = std::stof(trim(str[8]));
            feature.A(1,1) = std::stof(trim(str[9]));
            feature.matches = std::stoi(trim(str[10]));
            feature.inlierMatches = std::stoi(trim(str[11]));
            feature.hasPoint3D = trim(str[12]) == "true";
            feature.descriptorString = trim(str[13]);
        } catch (const std::logic_error& e) {
            continue;
        }
        features.emplace_back(std::move(feature));
    }
    return features;
}

int main(int argc, char *argv[]) {

    if (argc != 5) {
        std::cerr << "usage: " << argv[0] << " features.csv soure-images max-patches output-base\n";
        exit(-1);
    }

    const std::string featuresCSV(argv[1]);
    const std::string imageFolder(argv[2]);
    const size_t maxPatches = std::atoi(argv[3]);
    assert(maxPatches > 10);
    const std::string outputBase(argv[4]);

    //
    // Read in feature information.
    //
    std::vector<Feature> features = readFeatures(featuresCSV);
    const size_t N = features.size();

    //
    // Partition features into those features that have matches
    // and those that do not. We assume the features are arranged such
    // that features that from the same image are grouped together;
    // We use a stable partition to preserve this arrangement.
    //
    std::vector<size_t> featuresWithMatchesIndices;
    std::vector<size_t> featuresWithoutMatchesIndices;
    for (size_t i = 0; i < features.size(); i++)
        // XXX if (features[i].matches > 0)
        // XXX if (features[i].inlierMatches > 0)
        if (features[i].hasPoint3D)
            featuresWithMatchesIndices.push_back(i);
        else
            featuresWithoutMatchesIndices.push_back(i);

    //
    // Pick a evenly distributed subset of the features with
    // matches for output patches with matches.
    //
    const size_t M = featuresWithMatchesIndices.size();
    std::vector<size_t> patchesWithMatchesIndices;
    const size_t m = std::min(maxPatches, M);
    const size_t mdi = (M + m - 1)/m;
    for (size_t k = 0; k < M; k += mdi)
        patchesWithMatchesIndices.push_back(featuresWithMatchesIndices[k]);

    //
    // Similarly for patche without matches.
    //
    const size_t Q = featuresWithoutMatchesIndices.size();
    std::vector<size_t> patchesWithoutMatchesIndices;
    const size_t q = std::min(maxPatches,Q);
    const size_t qdi = (Q + q - 1)/q;
    for (size_t k = 0; k < Q; k += qdi)
        patchesWithoutMatchesIndices.push_back(featuresWithoutMatchesIndices[k]);

    //
    // Source image patch information that surrounds feature.
    //
    struct Patch {
        std::string imageName;
        cv::Rect rect;
    };

    //
    // Rectangle packing types and constants.
    //
    constexpr bool allow_flip = false;
    const auto runtime_flipping_mode = rectpack2D::flipping_option::DISABLED;
    using spaces_type = rectpack2D::empty_spaces<allow_flip, rectpack2D::default_empty_spaces>;
    using rect_type = rectpack2D::output_rect_t<spaces_type>;

    const size_t padding = 4;
    const auto max_side = 2000;
    const auto discard_step = -4;

    //
    // How to find a feature patch for a given features;
    //
    constexpr double bboxScale = 1.5;
    auto featureToPatch = [&](const Feature& f, Patch& patch) -> bool {
        const double scaleX = f.A.col(0).norm() * bboxScale;
        const double scaleY = f.A.col(1).norm() * bboxScale;
        const double orientation = std::atan2(f.A(1,0), f.A(0,0));
        const double c = cos(orientation), s = sin(orientation);
        const Eigen::Vector2d u = scaleX * Eigen::Vector2d(c, s);
        const Eigen::Vector2d v = scaleY * Eigen::Vector2d(-s, c);
        // https://iquilezles.org/articles/ellipses/
        const double bboxWidth  = std::sqrt(u.x()*u.x() + v.x()*v.x());
        const double bboxHeight = std::sqrt(u.y()*u.y() + v.y()*v.y());
        const double bboxLeft   = f.keypoint(0) - bboxWidth;
        const double bboxRight  = f.keypoint(0) + bboxWidth;
        const double bboxTop    = f.keypoint(1) - bboxHeight;
        const double bboxBottom = f.keypoint(1) + bboxHeight;
        // Note: I think x0,y0 can be non-integral (?)
        const int x0 = int(std::floor(bboxLeft));
        const int y0 = int(std::floor(bboxTop));
        const int W = int(std::ceil(bboxRight)) - x0;
        const int H = int(std::ceil(bboxBottom)) - y0;
        if (W < 2 || H < 2 || W + padding > max_side || H + padding > max_side || x0 < 0 || y0 < 0)
            return false;
        patch.imageName = f.imageName;
        patch.rect = cv::Rect(x0, y0, W, H);
        return true;;
    };

    //
    // Create source patch information for patches with matches.
    //
    std::vector<Patch> patchesWithMatches;
    for (size_t i : patchesWithMatchesIndices) {
        const Feature& f = features[i];
        Patch patch;
        if (featureToPatch(f, patch))
            patchesWithMatches.emplace_back(patch);
    }
    
    //
    // Create source patch information for patches without matches.
    //
    std::vector<Patch> patchesWithoutMatches;
    for (size_t i : patchesWithoutMatchesIndices) {
        const Feature& f = features[i];
        Patch patch;
        if (featureToPatch(f, patch))
            patchesWithoutMatches.emplace_back(patch);
    }

    //
    // Function that maps source image patches to output / packed rectangles.
    //
    auto packRectangles = [&](const std::vector<Patch>& patches,
                              std::vector<rect_type>& rectangles) -> rectpack2D::rect_wh {
        for (auto&& patch : patches) {
            rectpack2D::rect_xywh packedRect(0,0, patch.rect.width+padding, patch.rect.height+padding);
            rectangles.emplace_back(packedRect);
        }
        // XXX bool packing_success = true;
        size_t failCount = 0;
        const auto result_size = rectpack2D::find_best_packing<spaces_type>(
            rectangles,
            rectpack2D::make_finder_input(
                max_side,
                discard_step,
                [](rect_type& r) {
                    return rectpack2D::callback_result::CONTINUE_PACKING;
                },
                [&](rect_type& r) {
                    // XXX packing_success = false;
                    // XXX return rectpack2D::callback_result::ABORT_PACKING;
                    failCount++;
                    return rectpack2D::callback_result::CONTINUE_PACKING;
                },
                runtime_flipping_mode
                )
            );
        if (failCount > 0) {
            std::cerr << "warning: " << failCount << " failures!\n";
        }
        return result_size;
    };

    
    //
    // Find the output packing rectangles for feature patches with matches.
    //
    std::vector<rect_type> packedRectanglesWithMatches;
    const rectpack2D::rect_wh matchesSize = packRectangles(patchesWithMatches, packedRectanglesWithMatches);
    
    //
    // Find the output packing rectangles for feature patches without matches.
    //
    std::vector<rect_type> packedRectanglesWithoutMatches;
    const rectpack2D::rect_wh noMatchesSize = packRectangles(patchesWithoutMatches, packedRectanglesWithoutMatches);

    //
    // find_best_packing() in the packRectangles function above
    // rearranged the rectangles array so we create a
    // multi-map to find the appropriately sized
    // rectangle in the permuted rectangles array.
    //
    std::multimap<std::pair<int,int>,size_t> packedRectWithMatchesToIndex;  // maps w,h to index
    for (size_t i = 0; i < packedRectanglesWithMatches.size(); i++) {
        const auto& rect = packedRectanglesWithMatches[i];
        packedRectWithMatchesToIndex.insert({std::make_pair(rect.w,rect.h),i});
    }
    
    std::multimap<std::pair<int,int>,size_t> packedRectWithoutMatchesToIndex;  // maps w,h to index
    for (size_t i = 0; i < packedRectanglesWithoutMatches.size(); i++) {
        const auto& rect = packedRectanglesWithoutMatches[i];
        packedRectWithoutMatchesToIndex.insert({std::make_pair(rect.w,rect.h),i});
    }

    //
    // Function for creating a image containing the packed images.
    //
    cv::Mat currentImage;
    std::string currentImageName;
    cv::Rect currentImageRect;
    
    auto packedPatchesImage = [&](const std::vector<Patch>& patches,
                                  const std::vector<rect_type>& packedRectangles,
                                  std::multimap<std::pair<int,int>,size_t>& sizeToRectIndexMap,
                                  const rectpack2D::rect_wh& packedImageSize) -> cv::Mat {
        cv::Mat packedPatches(packedImageSize.h, packedImageSize.w, CV_8UC3, cv::Scalar(0, 0, 0));
        for (auto&& patch : patches) {
            if (patch.imageName != currentImageName) {
                const std::string imagePath = imageFolder + "/" + patch.imageName;
                currentImage = cv::imread(imagePath, cv::IMREAD_COLOR);
                currentImageName = patch.imageName;
                currentImageRect = cv::Rect(0,0, currentImage.cols, currentImage.rows);
            }
            const bool is_inside = (patch.rect & currentImageRect) == patch.rect;
            if (!is_inside) continue;
            cv::Mat sourcePatch = currentImage(patch.rect);
            const auto wh = std::make_pair(patch.rect.width + padding, patch.rect.height + padding);
            auto iter = sizeToRectIndexMap.find(wh);
            assert(iter != sizeToRectIndexMap.end());
            const size_t index = iter->second;
            const auto rect = packedRectangles[index];
            cv::Rect roi(rect.x,rect.y,rect.w - padding,rect.h - padding);
            sourcePatch.copyTo(packedPatches(roi));
            sizeToRectIndexMap.erase(iter);
        }
        return packedPatches;
    };

    cv::Mat packedPatchesImageWithMatches = packedPatchesImage(patchesWithMatches,
                                                               packedRectanglesWithMatches,
                                                               packedRectWithMatchesToIndex,
                                                               matchesSize);
    // XXX const std::string outputMatchesPath(outputBase + "-matches.png");
    // XXX const std::string outputMatchesPath(outputBase + "-inliers.png");
    const std::string outputMatchesPath(outputBase + "-has3D.png");
    cv::imwrite(outputMatchesPath, packedPatchesImageWithMatches);
    
    
    cv::Mat packedPatchesImageWithoutMatches = packedPatchesImage(patchesWithoutMatches,
                                                               packedRectanglesWithoutMatches,
                                                               packedRectWithoutMatchesToIndex,
                                                               noMatchesSize);
    // XXX const std::string outputNoMatchesPath(outputBase + "-no-matches.png");
    const std::string outputNoMatchesPath(outputBase + "-no-has3D.png");
    cv::imwrite(outputNoMatchesPath, packedPatchesImageWithoutMatches);

    return 0;
}
