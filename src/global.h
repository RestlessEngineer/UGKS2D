#ifndef GLOBAL_VARIABLES_H
#define GLOBAL_VARIABLES_H

#include <Eigen/Dense>
#include <vector>
#include <functional>
namespace ugks
{
    /// @brief order of calculations
    enum class boundary_type: unsigned char
    {
        EMPTY,
        GLUE,
        WALL,
        INPUT,
        OUTPUT,
        MIRROR,
        ROTATION,
        FUNCTIONAL
    };

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
    enum class boundary_side: unsigned char{
        LEFT,
        RIGHT,
        UP,
        DOWN,
        ERROR_SIZE
    };

    //TODO: erase this function
    inline boundary_side convert_to_side(std::string side){
        if(side == "LEFT")
            return boundary_side::LEFT;
        if(side == "RIGHT")
            return boundary_side::RIGHT;
        if(side == "UP")
            return boundary_side::UP;
        if(side == "DOWN")
            return boundary_side::DOWN;

        return boundary_side::ERROR_SIZE;
    }
    // inline string convert_to_side_name(boundary_side side){
    //     if(side == boundary_side::LEFT)
    //         return "LEFT";
    //     if(side == boundary_side::RIGHT)
    //         return "RIGHT";
    //     if(side == boundary_side::UP)
    //         return "UP";
    //     if(side == boundary_side::DOWN)
    //         return "DOWN";

    //     return "";
    // }

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
        double cosa, sina, p; // normals and perpendicular for normal equation of a line x*cosa + y*sina = p
        // flow flux
        Eigen::Array4d flux = {};       // mass flux, x and y momentum flux, energy flux
        Eigen::ArrayXXd flux_h, flux_b; // flux of distribution function
    };

    struct boundary_cell{
        Eigen::Array4d bound;
        // std::function<Eigen::Array4d(Eigen::Array4d, double, double)> func; //only for functional boundary
        boundary_type btype = boundary_type::EMPTY;
    };


    ///@brief structure for init physic values
    struct physic_val
    {
        double gamma;     // number whose value depends on the state of the gas
        double omega;     // temperature dependence index in HS/VHS/VSS model
        double Pr;        // Prandtl number
        double mu_ref;    // viscosity coefficient in reference state
        unsigned int DOF; // internal degree of freedom
    };

    ///@brief structure for getting simulations parameters
    struct simulation_val
    {
        // time parameters
        double dt;     // global time step
        double sitime; // current simulation time
        unsigned int cnt_iter;  // iteration

        // scheme parameters
        Eigen::Array4d res; // residual
        double CFL;         // global CFL number
        precision siorder;  // simulation order

        friend std::ostream& operator<<(std::ostream& os, const simulation_val& sim) {
            os << "simulation values:\n" << 
                "iter: "<< sim.cnt_iter <<
                " sitime: "<<sim.sitime << 
                " dt: "<< sim.dt << std::endl <<
                "residual: "<< sim.res.transpose();
            return os;
        }

    };

}
#endif