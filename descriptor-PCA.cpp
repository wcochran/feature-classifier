#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>

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

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "usage : " << argv[0] << " descriptors.csv XXX\n";
        exit(1);
    }

    const std::string featuresCSV(argv[1]);

    std::vector<int> matchCounts;
    std::vector<std::string> descriptorStrings;

    std::cout << "reading descriptors..." << std::endl;
    {
        std::ifstream is(featuresCSV);
        if (!is.is_open()) {
            std::cerr << "Unable to open '" << featuresCSV << "'\n";
            exit(-1);
        }

        constexpr size_t skip = 100;
        size_t k = 0;
        
        std::string line;
        while (std::getline(is, line)) {
            if ((k++ % skip) != 0) continue;
            if (line.length() <= 0 || line.at(0) == '#') continue;
            std::vector<std::string> str = split(line, ',');
            if (str.size() < 14) break;
            try {
                const int matches = std::stoi(trim(str[10]));
                const std::string hexstr = trim(str[13]);
                matchCounts.push_back(matches);
                descriptorStrings.push_back(hexstr);
            } catch (std::exception &e) {
                continue;
            }
        }

    }

    const size_t N = descriptorStrings.size();
    constexpr size_t M = 32; // project everything onto the space of the first M eigenvectors
    assert(N >= M);
    
    std::cout << "creating 128x" << N << " descriptor matrix..." << std::endl;
    Eigen::Matrix<double,128,Eigen::Dynamic> descriptors(128,N); 
    for (size_t i = 0; i < N; i++) {
        const std::string hexstr = descriptorStrings[i];
        assert(hexstr.length() == 128*2);
        for (size_t j = 0; j < 128; j++) {
            const std::string s = hexstr.substr(2*j,2);
            const int h = std::stoi(s, nullptr, 16);
            descriptors(j,i) = double(h);
        }
    }

    descriptorStrings.clear();

    std::cout << "creating 128x128 covariance matrix..." << std::endl;
    Eigen::Matrix<double,128,1> mean = descriptors.rowwise().mean();
    Eigen::Matrix<double,128,Eigen::Dynamic> A = descriptors.colwise() - mean; // 128 x N
    Eigen::Matrix<double,128,128> covariance = 1.0/(N-1) * A * A.transpose();  // 128 x 128

    
    std::cout << "Eigen analysis..." << std::endl;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,128,128>> eigenSolver(covariance);
    Eigen::Matrix<double,128,1> lambda = eigenSolver.eigenvalues().reverse();

    {
        const std::string lambdaName = "lambda.txt";
        std::ofstream os(lambdaName);
        if (!os.is_open()) {
            std::cerr << "Unable to open '" <<  lambdaName << "' for writing!\n";
            exit(-1);
        }
        // plot [0:64] "lambda.txt" using 1:2 with boxes
        // plot [0:64] "lambda.txt" using 1:3 with boxes
        const double totalSum = lambda.sum();
        double sum = 0; // accumulated sum
        for (size_t i = 0; i < 128; i++) {
            sum += lambda(i);
            const double fsum = sum / totalSum;
            os << i << " " << lambda(i) << " " << fsum << "\n";
        }
    }

    // Eigen::Matrix<double,128,128> X = eigenSolver.eigenvectors().rowwise().reverse();
    // Eigen::Matrix<double,128,Eigen::Dynamic> Y = X.block(0,0,128,M); // 128 x M principal components
    // Eigen::MatrixXd C = Y.transpose() * A;  // (M x 128) * (128 x N) -> M x N
    
    return 0;
}
