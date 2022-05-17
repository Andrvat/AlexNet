//
// Created by andrvat on 17.05.22.
//

#ifndef ALEXNET_LABELS_H
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

    void printLabels(std::ostream &os) const {
        for (auto x : labels) {
            os << x << "\n";
        }
    }

};

#endif //ALEXNET_LABELS_H
