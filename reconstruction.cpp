#include "reconstruction.h"

void Reconstruction::ReadBinary(const std::string& path) {
	ReadCamerasBinary(colmap::JoinPaths(path, "cameras.bin"));
	ReadImagesBinary(colmap::JoinPaths(path, "images.bin"));
	ReadPoints3DBinary(colmap::JoinPaths(path, "points3D.bin"));
}

void Reconstruction::WriteBinary(const std::string& path) const {
	WriteCamerasBinary(colmap::JoinPaths(path, "cameras.bin"));
	WriteImagesBinary(colmap::JoinPaths(path, "images.bin"));
	WritePoints3DBinary(colmap::JoinPaths(path, "points3D.bin"));
}

void Reconstruction::ReadCamerasBinary(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Unable to open '" << path << "' for reading!\n";
		exit(-1);
	}
	
	const size_t num_cameras = colmap::ReadBinaryLittleEndian<uint64_t>(&file);
	for (size_t i = 0; i < num_cameras; ++i) {
		colmap::Camera camera;
		camera.SetCameraId(colmap::ReadBinaryLittleEndian<colmap::camera_t>(&file));
		camera.SetModelId(colmap::ReadBinaryLittleEndian<int>(&file));
		camera.SetWidth(colmap::ReadBinaryLittleEndian<uint64_t>(&file));
		camera.SetHeight(colmap::ReadBinaryLittleEndian<uint64_t>(&file));
		colmap::ReadBinaryLittleEndian<double>(&file, &camera.Params());
		cameras.emplace(camera.CameraId(), camera);
	}
}

void Reconstruction::ReadImagesBinary(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Unable to open '" << path << "' for reading!\n";
		exit(-1);
	}

	const size_t num_reg_images = colmap::ReadBinaryLittleEndian<uint64_t>(&file);
	for (size_t i = 0; i < num_reg_images; ++i) {
		colmap::Image image;
		
		image.SetImageId(colmap::ReadBinaryLittleEndian<colmap::image_t>(&file));
		
		image.Qvec(0) = colmap::ReadBinaryLittleEndian<double>(&file);
		image.Qvec(1) = colmap::ReadBinaryLittleEndian<double>(&file);
		image.Qvec(2) = colmap::ReadBinaryLittleEndian<double>(&file);
		image.Qvec(3) = colmap::ReadBinaryLittleEndian<double>(&file);
		image.NormalizeQvec();

		image.Tvec(0) = colmap::ReadBinaryLittleEndian<double>(&file);
		image.Tvec(1) = colmap::ReadBinaryLittleEndian<double>(&file);
		image.Tvec(2) = colmap::ReadBinaryLittleEndian<double>(&file);

		image.SetCameraId(colmap::ReadBinaryLittleEndian<colmap::camera_t>(&file));

		char name_char;
		do {
			file.read(&name_char, 1);
			if (name_char != '\0') {
				image.Name() += name_char;
			}
		} while (name_char != '\0');
		
		const size_t num_points2D = colmap::ReadBinaryLittleEndian<uint64_t>(&file);

		std::vector<Eigen::Vector2d> points2D;
		points2D.reserve(num_points2D);
		std::vector<colmap::point3D_t> point3D_ids;
		point3D_ids.reserve(num_points2D);
		for (size_t j = 0; j < num_points2D; ++j) {
			const double x = colmap::ReadBinaryLittleEndian<double>(&file);
			const double y = colmap::ReadBinaryLittleEndian<double>(&file);
			points2D.emplace_back(x, y);
			point3D_ids.push_back(colmap::ReadBinaryLittleEndian<colmap::point3D_t>(&file));
		}

		image.SetPoints2D(points2D);

        for (colmap::point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
             ++point2D_idx) {
          if (point3D_ids[point2D_idx] != colmap::kInvalidPoint3DId) {
            image.SetPoint3DForPoint2D(point2D_idx, point3D_ids[point2D_idx]);
          }
        }
        
        image.SetRegistered(true);
		images.emplace(image.ImageId(), image);
	}
}

void Reconstruction::ReadPoints3DBinary(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Unable to open '" << path << "' for reading!\n";
		exit(-1);
	}

	const size_t num_points3D = colmap::ReadBinaryLittleEndian<uint64_t>(&file);
	for (size_t i = 0; i < num_points3D; ++i) {
		class colmap::Point3D point3D;

		const colmap::point3D_t point3D_id = colmap::ReadBinaryLittleEndian<colmap::point3D_t>(&file);

		point3D.XYZ()(0) = colmap::ReadBinaryLittleEndian<double>(&file);
		point3D.XYZ()(1) = colmap::ReadBinaryLittleEndian<double>(&file);
		point3D.XYZ()(2) = colmap::ReadBinaryLittleEndian<double>(&file);
		point3D.Color(0) = colmap::ReadBinaryLittleEndian<uint8_t>(&file);
		point3D.Color(1) = colmap::ReadBinaryLittleEndian<uint8_t>(&file);
		point3D.Color(2) = colmap::ReadBinaryLittleEndian<uint8_t>(&file);
		point3D.SetError(colmap::ReadBinaryLittleEndian<double>(&file));

		const size_t track_length = colmap::ReadBinaryLittleEndian<uint64_t>(&file);
		for (size_t j = 0; j < track_length; ++j) {
			const colmap::image_t image_id = colmap::ReadBinaryLittleEndian<colmap::image_t>(&file);
			const colmap::point2D_t point2D_idx = colmap::ReadBinaryLittleEndian<colmap::point2D_t>(&file);
			point3D.Track().AddElement(image_id, point2D_idx);
		}
		point3D.Track().Compress();

		points3D.emplace(point3D_id, point3D);
	}
}

void Reconstruction::WriteCamerasBinary(const std::string& path) const {
	std::ofstream file(path, std::ios::trunc | std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Unable to open '" << path << "' for writing!\n";
		exit(-1);
	}

	colmap::WriteBinaryLittleEndian<uint64_t>(&file, cameras.size());

	for (const auto& camera : cameras) {
		colmap::WriteBinaryLittleEndian<colmap::camera_t>(&file, camera.first);
		colmap::WriteBinaryLittleEndian<int>(&file, camera.second.ModelId());
		colmap::WriteBinaryLittleEndian<uint64_t>(&file, camera.second.Width());
		colmap::WriteBinaryLittleEndian<uint64_t>(&file, camera.second.Height());
		for (const double param : camera.second.Params()) {
			colmap::WriteBinaryLittleEndian<double>(&file, param);
		}
	}
}

void Reconstruction::WriteImagesBinary(const std::string& path) const {
	std::ofstream file(path, std::ios::trunc | std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Unable to open '" << path << "' for writing!\n";
		exit(-1);
	}
	
	colmap::WriteBinaryLittleEndian<uint64_t>(&file, images.size());

	for (const auto& image : images) {
		colmap::WriteBinaryLittleEndian<colmap::image_t>(&file, image.first);

		const Eigen::Vector4d normalized_qvec = image.second.Qvec();
		colmap::WriteBinaryLittleEndian<double>(&file, normalized_qvec(0));
		colmap::WriteBinaryLittleEndian<double>(&file, normalized_qvec(1));
		colmap::WriteBinaryLittleEndian<double>(&file, normalized_qvec(2));
		colmap::WriteBinaryLittleEndian<double>(&file, normalized_qvec(3));

		colmap::WriteBinaryLittleEndian<double>(&file, image.second.Tvec(0));
		colmap::WriteBinaryLittleEndian<double>(&file, image.second.Tvec(1));
		colmap::WriteBinaryLittleEndian<double>(&file, image.second.Tvec(2));

		colmap::WriteBinaryLittleEndian<colmap::camera_t>(&file, image.second.CameraId());

		const std::string name = image.second.Name() + '\0';
		file.write(name.c_str(), name.size());

		colmap::WriteBinaryLittleEndian<uint64_t>(&file, image.second.NumPoints2D());
		for (const colmap::Point2D& point2D : image.second.Points2D()) {
			colmap::WriteBinaryLittleEndian<double>(&file, point2D.X());
			colmap::WriteBinaryLittleEndian<double>(&file, point2D.Y());
			colmap::WriteBinaryLittleEndian<colmap::point3D_t>(&file, point2D.Point3DId());
		}
	}
}

void Reconstruction::WritePoints3DBinary(const std::string& path) const {
	std::ofstream file(path, std::ios::trunc | std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Unable to open '" << path << "' for writing!\n";
		exit(-1);
	}

	colmap::WriteBinaryLittleEndian<uint64_t>(&file, points3D.size());

	for (const auto& point3D : points3D) {
		colmap::WriteBinaryLittleEndian<colmap::point3D_t>(&file, point3D.first);
		colmap::WriteBinaryLittleEndian<double>(&file, point3D.second.XYZ()(0));
		colmap::WriteBinaryLittleEndian<double>(&file, point3D.second.XYZ()(1));
		colmap::WriteBinaryLittleEndian<double>(&file, point3D.second.XYZ()(2));
		colmap::WriteBinaryLittleEndian<uint8_t>(&file, point3D.second.Color(0));
		colmap::WriteBinaryLittleEndian<uint8_t>(&file, point3D.second.Color(1));
		colmap::WriteBinaryLittleEndian<uint8_t>(&file, point3D.second.Color(2));
		colmap::WriteBinaryLittleEndian<double>(&file, point3D.second.Error());

		colmap::WriteBinaryLittleEndian<uint64_t>(&file, point3D.second.Track().Length());
		for (const auto& track_el : point3D.second.Track().Elements()) {
			colmap::WriteBinaryLittleEndian<colmap::image_t>(&file, track_el.image_id);
			colmap::WriteBinaryLittleEndian<colmap::point2D_t>(&file, track_el.point2D_idx);
		}
	}
}
