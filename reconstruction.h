#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include <string>
#include <unordered_map>
#include <colmap/base/camera.h>
#include <colmap/base/image.h>
#include <colmap/base/point2d.h>
#include <colmap/base/point3d.h>
#include <colmap/util/misc.h>

class Reconstruction {
public:
	std::unordered_map<colmap::camera_t,colmap::Camera> cameras;
	std::unordered_map<colmap::image_t,colmap::Image> images;
	std::unordered_map<colmap::point3D_t,colmap::Point3D> points3D;

	void ReadBinary(const std::string& path);
	void WriteBinary(const std::string& path) const;

private:	
	void ReadCamerasBinary(const std::string& path);
	void ReadImagesBinary(const std::string& path);
	void ReadPoints3DBinary(const std::string& path);
	
	void WriteCamerasBinary(const std::string& path) const;
	void WriteImagesBinary(const std::string& path) const;
	void WritePoints3DBinary(const std::string& path) const;
};

#endif // RECONSTRUCTION_H
