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
        static const unsigned int MNUM = 7; // number of normal velosity moments
        static const unsigned int MTUM = 5; // number of tangential velosity moments

        Eigen::Array<point, -1, -1> mesh;                  // mesh (node coordinates)
        Eigen::Array<cell, -1, -1> core;                   // cell centers
        Eigen::Array<cell_interface, -1, -1> vface, hface; // vertical and horizontal interfaces

        Eigen::Array4d bc_L, bc_R, bc_U, bc_D; // boundary conditions at LEFT, RIGHT, UP and DOWN boundary

        //* velosity space
        size_t usize = 0, vsize = 0;    // number of velosity points for u and v
        double umax, vmax;              // maximum micro velosity
        Eigen::ArrayXXd uspace, vspace; // u and v discrete velosity space
        Eigen::ArrayXXd weight;         // weight at velosity u_k and v_l

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
        /// @param bound array with boundary values (density,u-velosity,v-velosity,lambda)
        /// @param type what is the boundary (LEFT, RIGHT, UP, DOWN)
        void set_boundary(const Eigen::Array4d bound, boundary type);

        /// @brief initialize the mesh
        /// @param xnum,ynum       :number of cells in x and y direction
        /// @param xlength,ylength :domain length in x and y direction
        void set_geometry(const double &xlength, const double &ylength);

        /// @brief set the initial gas condition
        /// @param init_gas initial condition
        void set_flow_field(const Eigen::Array4d &init_gas);

        /// @brief initialize velosity space 
        /// @param param parameters for filling
        /// @param integ integration way (Newton-Cotes or Gauss)
        void set_velosity_space(const vel_space_param& param, integration integ = integration::GAUSS);

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
        void write_results() const;

    private:
        
        /// @brief allocation global arrays
        void allocation_velosity_space();

        /// @brief calculation of the time step
        void timestep();

        /// @brief calculation of the slope of distribution function
        void interpolation();
        
        /// @brief calculate the flux across the interfaces
        void flux_calculation();

        /// @brief updating of the cell averaged values
        void update();

        /// @brief one-sided interpolation of the boundary cell
        /// @param cell_N the target boundary cell
        /// @param cell_L the left cell
        /// @param cell_R the right cell
        /// @param idx    the index indicating i or j direction
        void interp_boundary(cell &cell_N, cell &cell_L, cell &cell_R, direction dir);

        /// @brief interpolation of the inner cells
        /// @param cell_L the left cell
        /// @param cell_N the target cell
        /// @param cell_R the right cell
        /// @param idx    the index indicating i or j direction
        void interp_inner(cell &cell_L, cell &cell_N, cell &cell_R, direction dir);

        /// @brief calculate flux of boundary interface, assuming left wall
        /// @param bc   boundary condition
        /// @param face the boundary interface
        /// @param cell cell next to the boundary interface
        /// @param idx  index indicating i or j direction
        /// @param order indicating direct or revers order
        void calc_flux_boundary(const Eigen::Array4d &bc, cell_interface &face, cell cell, direction dir, int order);

        /// @brief calculate micro slope of Maxwellian distribution
        /// @param prim primary variables
        /// @param sw   slope of W
        /// @return slope of Maxwellian distribution
        [[nodiscard]] Eigen::Array4d micro_slope(const Eigen::Array4d &prim, const Eigen::Array4d &sw);

        /// @brief calculate moments of velosity
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
    };
}

#endif