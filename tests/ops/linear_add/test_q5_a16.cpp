#include "ops/linear_add/linear_add_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::test::linear_add::WeightFormat;

int q5_a16_conformance() {
    // Starts of the registered positive-T regions. run_shape checks b-1/b/b+1 for every start,
    // plus one interior point for every region, through the public Op. The first region now
    // reaches down to T=1 (T=1 is covered by the leading probe and its own interior point), and
    // 513 starts the whole-wave + narrow-tail composite, whose interiors cover a short, a
    // 128-column and a 256-column tail (the last one keeping the single wide launch).
    constexpr std::array<std::int32_t, 15> kInteriors{1,  2,   3,   8,   24,  40,  56, 64,
                                                      96, 128, 129, 256, 640, 768, 1024};
    constexpr std::array<std::int32_t, 5> kK6144RouteStarts{14, 33, 49, 193, 513};

    int failures = 0;
    failures += ninfer::test::linear_add::run_shape(
        "Q5_A16 LinearAdd", WeightFormat::Q5G64F16S,
        ShapeCase{5120, 6144, 401U, kK6144RouteStarts, kInteriors});
    constexpr std::array<std::int32_t, 5> kK17408RouteStarts{17, 33, 49, 193, 513};
    failures += ninfer::test::linear_add::run_shape(
        "Q5_A16 LinearAdd", WeightFormat::Q5G64F16S,
        ShapeCase{5120, 17408, 409U, kK17408RouteStarts, kInteriors});
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear_add::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q5_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q5_A16 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q5_A16 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
