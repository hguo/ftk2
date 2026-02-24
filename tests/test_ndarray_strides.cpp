#include <ndarray/ndarray.hh>
#include <iostream>
#include <vector>

int main() {
    ftk::ndarray<float> arr({32, 32, 32, 10});
    auto lattice = arr.get_lattice();
    std::cout << "Rank: " << arr.nd() << std::endl;
    for (int i = 0; i < arr.nd(); ++i) {
        std::cout << "dim[" << i << "] = " << arr.dimf(i) << ", prod[" << i << "] = " << lattice.prod_[i] << std::endl;
    }
    return 0;
}
