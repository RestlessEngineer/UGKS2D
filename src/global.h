#ifndef GLOBAL_VARIABLES_H
#define GLOBAL_VARIABLES_H

#include <Eigen/Dense>
#include <vector>

namespace ugks
{

    /// @brief order of calculations
    enum class precision : unsigned char
    {
        FIRST_ORDER,
        SECOND_ORDER // extra procedure interpolation of values
    };

    /// @brief order for calculating fluxes
    enum order
    {
        REVERSE = -1, // reverse order
        DIRECT = 1    // direct order
    };

    /// @brief boundary side 
    enum class boundary: unsigned char{
        LEFT,
        RIGHT,
        UP,
        DOWN
    };

    /// @brief indexes of directions
    enum direction
    {
        IDIR = 0, // i direction, along Y
        JDIR = 1, // j direction, along X
        XDIR = IDIR,
        YDIR = JDIR,
        DX = IDIR,
        DY = JDIR
    };

    /// @brief integration way for velocity space
    enum class integration
    {
        NEWTON_COTES,
        GAUSS
    };

    /// @brief alternative indexes for simple reading
    enum flow
    {
        RHO,   // density
        U,     // x-momentum
        V,     // y-momentum
        ENERGY // total energy
    };

    /// @brief mesh points
    struct point
    {
        double x, y; // coordinates
    };

    /// @brief velocity space parameters
    struct vel_space_param
    {
        size_t num_u{0}, num_v{0};   // number of velocity points
        double min_u{-1}, min_v{-1}; // smallest discrete velocity
        double max_u{1}, max_v{1};   // largest discrete velocity
    };

    /// @brief core of cell
    struct cell
    {
        double x, y;                       // cell center coordinates
        std::array<double, 2> length = {}; // length in i and j direction
        double area;                       // cell area
        // flow field
        Eigen::Array4d w = {};                 // density, x-momentum, y-momentum, total energy
        Eigen::ArrayXXd h, b;                  // distribution function
        std::vector<const cell*> neighbors;
        std::array<Eigen::ArrayXXd, 2> sh, sb; // slope of distribution function in i and j direction
    };

    /// @brief cell interface
    struct cell_interface
    {
        double length; // length of cell interface
        double nx, ny; // normals
        // flow flux
        Eigen::Array4d flux = {};       // mass flux, x and y momentum flux, energy flux
        Eigen::ArrayXXd flux_h, flux_b; // flux of distribution function
    };

}
#endif