#include <gtest/gtest.h>
#include "../data-type-space-grid.cuh" 

// for GridGeometry
TEST(GridGeometryTest, AssignmentOperator) {
    GridGeometry<3> g1 = {{1, 2, 3}, {0.1, 0.2, 0.3}, {0.0, 0.0, 0.0}};
    GridGeometry<3> g2 = g1;

    for (size_t i = 0; i < 3; i++) {
        ASSERT_EQ(g2.dims[i], g1.dims[i]);
        ASSERT_EQ(g2.block_size[i], g1.block_size[i]);
        ASSERT_EQ(g2.center_position[i], g1.center_position[i]);
    }
};

// similarly for Geometry
TEST(GeometryTest, AssignmentOperator) {
    double metric[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    double boundary[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    Boundary<3>::BoundaryCondition condition = Boundary<3>::BoundaryCondition::PERIODIC;

    Boundary<3> g1(metric, condition, boundary);
    Boundary<3> g2 = g1;

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            ASSERT_EQ(g2.metric[i][j], g1.metric[i][j]);
        }
        ASSERT_EQ(g2.boundary[i][0], g1.boundary[i][0]);
        ASSERT_EQ(g2.boundary[i][1], g1.boundary[i][1]);
    }

    ASSERT_EQ(g2.condition, g1.condition);
}


TEST(GridGeometryTest, StreamOperators) {
    GridGeometry<3> g1 = {{1, 2, 3}, {0.1, 0.2, 0.3}, {0.0, 0.0, 0.0}};
    std::stringstream ss;

    ss << g1;

    GridGeometry<3> g2;
    ss >> g2;

    ASSERT_EQ(g2.dims, g1.dims);
    ASSERT_EQ(g2.block_size, g1.block_size);
    ASSERT_EQ(g2.center_position, g1.center_position);
}


TEST(GridGeometryTest, FileStreamOperators) {
    GridGeometry<3> g1 = {{1, 2, 3}, {0.1, 0.2, 0.3}, {0.0, 0.0, 0.0}};

    // Write to a file
    std::ofstream ofs("test.txt");
    ofs << g1;
    ofs.close();

    // Read from a file
    GridGeometry<3> g2;
    std::ifstream ifs("test.txt");
    ifs >> g2;
    ifs.close();

    ASSERT_EQ(g2.dims, g1.dims);
    ASSERT_EQ(g2.block_size, g1.block_size);
    ASSERT_EQ(g2.center_position, g1.center_position);
}

// Similar adjustment for GeometryTest


TEST(GeometryTest, Constructor) {
    Boundary<3> g1;
    // Initialize g1...
    
    ASSERT_NO_THROW(Boundary<3> g2(g1.metric, g1.condition, g1.boundary));
}


