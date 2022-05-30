/#ifndef ALEXNET_LABELS_H
#define ALEXNET_LABELS_H


#include <vector>
#include <fstream>

class Labels {
private:
    std::vector<int> labels;

public:
    void readFrom(const std::string& filename) {
        std::ifstream stream(filename);
        int label;
        while (stream >> label) {
            labels.push_back(label);
        }
        stream.close();
    }

    const std::vector<int> &getLabels() const {
        return labels;
    }
};

#endif //ALEXNET_LABELS_H
