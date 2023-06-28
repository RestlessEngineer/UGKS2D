#ifndef SOLVER_H
#define SOLVER_H

#include "global.h"
#include <math.h>

//*******************************************************
// coordinate system                                    *
//                                                      *
//    Y ^                                               *
//      |              (i+1,j)                          *
//      |         ----------------                      *
//      |         |              |                      *
//      |         |              |                      *
//      |         |              |                      *
//      |    (i,j)|     (i,j)    |(i, j+1)              *
//      |         |              |                      *
//      |         |              |                      *
//      |         |              |                      *
//      |         ----------------                      *
//      |               (i,j)                           *
//      |                                               *
//      0--------------------------------------->       *
//                                             X        *
//*******************************************************

namespace ugks
{

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
        int cnt_iter;  // iteration

        // scheme parameters
        Eigen::Array4d res; // residual
        double CFL;         // global CFL number
        precision siorder;  // simulation order
    };

    class solver
    {

        size_t ysize = 0, xsize = 0; // index range in i and j direction

        //* physics constants
        double gamma;     // number whose value depends on the state of the gas
        double omega;     // temperature dependence index in HS/VHS/VSS model
        double Pr;        // Prandtl number
        double mu_ref;    // viscosity coefficient in reference state
        unsigned int DOF; // internal degree of freedom

        //* simulation parameters
        precision siorder = precision::SECOND_ORDER; // simulation order
        
        double CFL = 0.8;   // global CFL number
        Eigen::Array4d res; // simulation residual

        double dt;          // global time step
        double sitime = 0.; // current simulation time
        int cnt_iter = 0;   // count of iterations


        //* fluid space
        static const unsigned int MNUM = 7; // number of normal velocity moments
        static const unsigned int MTUM = 5; // number of tangential velocity moments

        Eigen::Array<point, -1, -1> mesh;                  // mesh (node coordinates)
        Eigen::Array<cell, -1, -1> core;                   // cell centers
        Eigen::Array<cell_interface, -1, -1> vface, hface; // vertical and horizontal interfaces

        Eigen::Array4d bc_L, bc_R, bc_U, bc_D; // boundary conditions at LEFT, RIGHT, UP and DOWN boundary
        boundary_type bc_typeL, bc_typeR, bc_typeU, bc_typeD; //boundary types WALL, INPUT, OUTPUT, MIRROR and others

        //* velocity space
        size_t usize = 0, vsize = 0;    // number of velocity points for u and v
        double umax, vmax;              // maximum micro velocity
        Eigen::ArrayXXd uspace, vspace; // u and v discrete velocity space
        Eigen::ArrayXXd weight;         // weight at velocity u_k and v_l

    public:
        /// @brief constructor ugks solver
        /// @param rows count cells along Y
        /// @param cols count cells along X
        solver(const size_t &rows, const size_t &cols);

        /// @brief constructor ugks solver
        /// @param rows count cells along Y
        /// @param cols count cells along X
        /// @param phys physic parameters
        solver(const size_t &rows, const size_t &cols, const physic_val &phys);

        /// @brief constructor ugks solver
        /// @param rows count cells along Y
        /// @param cols count cells along X
        /// @param phys physic parameters
        /// @param ord  simulation order
        solver(const size_t &rows, const size_t &cols, const physic_val &phys, const precision &_ord) : 
            solver(rows, cols, phys) { siorder = _ord;}

        /// @brief constructor ugks solver
        /// @param rows count cells along Y
        /// @param cols count cells along X
        /// @param phys physic parameters
        /// @param ord  simulation order
        /// @param CFL  Courant–Friedrichs–Lewy number
        solver(const size_t &rows, const size_t &cols, const physic_val &phys, const precision &_ord, const double &_CFL) : 
            solver(rows, cols, phys, _ord) { CFL = _CFL;}

        /// @brief constructor ugks solver
        /// @param rows count cells along Y
        /// @param cols count cells along X
        /// @param phys physic parameters
        /// @param CFL  Courant–Friedrichs–Lewy number
        /// @param ord  simulation order
        solver(const physic_val &phys, const double &CFL, const precision &ord,
               const size_t &rows, const size_t &cols);

        /// @brief set boundary condition
        /// @param side what is the boundary (LEFT, RIGHT, UP, DOWN)
        /// @param bound array with boundary values (density,u-velocity,v-velocity,lambda)
        /// @param type boundary type  WALL, INPUT, OUTPUT, MIRROR and others
        void set_boundary(boundary_side side, const Eigen::Array4d bound, boundary_type type = boundary_type::WALL);

        /// @brief initialize the mesh
        /// @param xlength,ylength :domain length in x and y direction
        void set_geometry(const double &xlength, const double &ylength);

        /// @brief initialize the mesh
        /// @param left_down left down point of the rectangular
        /// @param left_up left up point of the rectangular
        /// @param right_up  right up point of the rectangular
        /// @param right_down  right down point of the rectangular
        void set_geometry(const point &left_down, const point &left_up, const point &right_up, const point &right_down);

        /// @brief initialize the mesh
        /// @param up_wall left down point of the rectangular
        /// @param down_wall left up point of the rectangular
        void set_geometry(const std::vector<point>& up_wall, const std::vector<point>& down_wall);


        /// @brief set the initial gas condition
        /// @param init_gas initial condition
        void set_flow_field(const Eigen::Array4d &init_gas);

        /// @brief initialize velocity space 
        /// @param param parameters for filling
        /// @param integ integration way (Newton-Cotes or Gauss)
        void set_velocity_space(const vel_space_param& param, integration integ = integration::GAUSS);

        /// @brief solving for one time step
        /// @return simulation parameters
        inline simulation_val solve()
        {

            timestep();         // calculate time step
            interpolation();    // calculate the slope of distribution function
            flux_calculation(); // calculate flux across the interfaces
            update();           // update cell averaged value

            cnt_iter++;
            sitime += dt;

            return {dt, sitime, cnt_iter, res, CFL, siorder};
        }

        /// @brief writting current results
        void write_results(std::string file_name = "cavity.dat") const;
        
    private:
        
        /// @brief make acquaintances for neighbors
        void associate_neighbors();

        /// @brief allocation global arrays
        void allocation_velocity_space();

        /// @brief calculation of the time step
        void timestep();

        /// @brief calculation of the slope of distribution function
        void interpolation();

        /// @brief calculate the flux across the interfaces
        void flux_calculation();

        /// @brief updating of the cell averaged values
        void update();

        /// @brief calculate dx dy slopes by solving linear least square system
        /// @param core central cell 
        void least_square_solver(cell& core);

        /// @brief calculate flux of boundary interface, assuming left wall
        /// @param bc   boundary condition
        /// @param face the boundary interface
        /// @param cell cell next to the boundary interface
        /// @param btype boundary type (WALL, INPUT, OUTPUT, MIRROR)
        void calc_flux_boundary(const Eigen::Array4d &bc, cell_interface &face, cell cell, boundary_type btype, int side);
        
        /// @brief calculate flux of boundary interface
        /// @param bc   boundary condition
        /// @param face the boundary interface
        /// @param cell cell next to the boundary interface
        void calc_flux_boundary_wall(const Eigen::Array4d &bc, cell_interface &face, const cell& cell, int side);
        
        /// @brief calculate flux of boundary for input conditions
        /// @param bc   boundary condition
        /// @param face the boundary interface
        /// @param cell cell next to the boundary interface
        void calc_flux_boundary_input(const Eigen::Array4d &bc, cell_interface &face, const cell& cell, int side);
        
        /// @brief calculate flux of boundary for output
        /// @param bc   boundary condition
        /// @param face the boundary interface
        /// @param cell cell next to the boundary interface
        void calc_flux_boundary_output(const Eigen::Array4d &bc, cell_interface &face, const cell& cell, int side);
        
        /// @brief calculate flux of boundary for axisymmetric one
        /// @param bc   boundary condition
        /// @param face the boundary interface
        /// @param cell cell next to the boundary interface
        void calc_flux_boundary_mirror(const Eigen::Array4d &bc, cell_interface &face, const cell& cell, int side);


        /// @brief calculate micro slope of Maxwellian distribution
        /// @param prim primary variables
        /// @param sw   slope of W
        /// @return slope of Maxwellian distribution
        [[nodiscard]] Eigen::Array4d micro_slope(const Eigen::Array4d &prim, const Eigen::Array4d &sw);

        /// @brief calculate moments of velocity
        /// @param prim      primary variables
        /// @param Mu,Mv     <u^n>,<v^m>
        /// @param Mxi       <\xi^l>
        /// @param Mu_L,Mu_R <u^n>_{>0},<u^n>_{<0}
        void calc_moment_u(const Eigen::Array4d &prim,
                           Eigen::Array<double, MNUM, 1> &Mu, Eigen::Array<double, MTUM, 1> &Mv, Eigen::Array<double, 3, 1> &Mxi,
                           Eigen::Array<double, MNUM, 1> &Mu_L, Eigen::Array<double, MNUM, 1> &Mu_R);

        /// @brief calculate <u^\alpha*v^\beta*\xi^\delta*\psi>
        /// @param Mu,Mv      <u^\alpha>,<v^\beta>
        /// @param Mxi        <\xi^l>
        /// @param alpha,beta exponential index of u and v
        /// @param delta      exponential index of \xi
        /// @return  moment of <u^\alpha*v^\beta*\xi^\delta*\psi>
        [[nodiscard]] Eigen::Array4d moment_uv(Eigen::Array<double, MNUM, 1> &Mu, Eigen::Array<double, MTUM, 1> &Mv, Eigen::Array<double, 3, 1> &Mxi,
                                               const int alpha, const int beta, const int delta);

        /// @brief calculate <a*u^\alpha*v^\beta*\psi>
        /// @param a          micro slope of Maxwellian
        /// @param Mu,Mv      <u^\alpha>,<v^\beta>
        /// @param Mxi        <\xi^l>
        /// @param alpha,beta exponential index of u and v
        /// @return moment of <a*u^\alpha*v^\beta*\psi>
        [[nodiscard]] Eigen::Array4d moment_au(const Eigen::Array4d &a,
                                               Eigen::Array<double, MNUM, 1> &Mu, Eigen::Array<double, MTUM, 1> &Mv, Eigen::Array<double, 3, 1> &Mxi,
                                               const int alpha, const int beta);

        /// @brief calculate flux of inner interface
        /// @param cell_L cell left to the target interface
        /// @param face   the target interface
        /// @param cell_R cell right to the target interface
        /// @param idx    index indicating i or j direction
        void calc_flux(cell &cell_L, cell_interface &face, cell &cell_R, direction dir);

        /// @brief filling of mesh
        /// @param xupw x cords of up wall
        /// @param yupw y cords of up wall
        /// @param xdownw x cords of down wall 
        /// @param ydownw y cords of down wall
        void fill_mesh(const Eigen::ArrayXd& xupw, const Eigen::ArrayXd& yupw, const Eigen::ArrayXd& xdownw, const Eigen::ArrayXd& ydownw);
    };
}

#endif