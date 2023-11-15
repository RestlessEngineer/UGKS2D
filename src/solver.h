#ifndef SOLVER_H
#define SOLVER_H

#include "global.h"
#include "tools.h"
#include <math.h>
#ifdef DO_PROFILIZE
    #include <iostream>
#endif

#include <nlohmann/json.hpp>
using json = nlohmann::json;

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
    class block_solver;

    class solver
    {
        friend class block_solver;
        size_t ysize = 0, xsize = 0; // index range in i and j direction

        //* physics constants
        unsigned int DOF = 1;                                // internal degree of freedom
        double gamma = ugks::tools::get_gamma(DOF);          // number whose value depends on the state of the gas
        double omega = 0.72;                                 // temperature dependence index in HS/VHS/VSS model
        double Pr = 2.0/3.0;                                 // Prandtl number
        double mu_ref = ugks::tools::get_mu(0.01, 1, 0.5);   // viscosity coefficient in reference state


        //* simulation parameters
        precision siorder = precision::SECOND_ORDER; // simulation order
        
        double CFL = 0.8;   // global CFL number
        double sitime = 0.; // current simulation time
        unsigned int cnt_iter = 0;   // count of iterations


        //* fluid space
        static const unsigned int MNUM = 7; // number of normal velocity moments
        static const unsigned int MTUM = 5; // number of tangential velocity moments

        Eigen::Array<point, -1, -1> mesh;                  // mesh (node coordinates)
        Eigen::Array<cell, -1, -1> core;                   // cell centers
        Eigen::Array<cell_interface, -1, -1> vface, hface; // vertical and horizontal interfaces

        using Boundary = Eigen::Array<boundary_cell, -1, 1>; 

        Boundary lbound, rbound, ubound, dbound; // boundary at LEFT, RIGHT, UP and DOWN boundary    
        Eigen::Array<cell*, -1, 1> lcell, rcell, ucell, dcell; // for GLUING cells

        //* velocity space
        size_t usize = 0, vsize = 0;    // number of velocity points for u and v
        // maximum micro velocity
        double umax = std::numeric_limits<double>::max();
        double vmax = std::numeric_limits<double>::max();              
        Eigen::ArrayXXd uspace, vspace; // u and v discrete velocity space
        Eigen::ArrayXXd weight;         // weight at velocity u_k and v_l

    public:

        solver() = default;
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

        /// @brief set boundary condition
        /// @param side what is the boundary (LEFT, RIGHT, UP, DOWN)
        /// @param bound array with boundary values
        void set_boundary(boundary_side side, const Boundary& bound);

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
        simulation_val solve();

        /// @brief writting current results
        void write_results(const std::string& file_name = "cavity.dat") const;

        /// @brief saving mesh
        /// @param file_name mesh name 
        void write_mesh(const std::string& file_name = "mesh.dat") const;

        /// @brief filling inner values from file
        /// @param file_name file with results
        void init_inner_values_by_result(std::string file_name);
        
        /// @brief return neighbors from frontier values
        /// @param side with side
        /// @param range range for neighbors 
        /// @return vector pointers to frontier values from side  
        std::vector<cell* > get_frontier(boundary_side side, std::pair<size_t, size_t> range);

        /// @brief associate frontier neighbors
        /// @param neighbors frontier neighbors 
        /// @param side wich side
        /// @param range range for association
        void associate_neighbors(const std::vector<cell* > & neighbors, boundary_side side, std::pair<size_t, size_t> range);
        
        //TODO: only for rotation
        std::vector<point> get_boundary_points(boundary_side side, std::pair<size_t, size_t> range);

        /// @brief filling of mesh
        /// @param xupw x cords of up wall
        /// @param yupw y cords of up wall
        /// @param xdownw x cords of down wall 
        /// @param ydownw y cords of down wall
        void fill_mesh(const Eigen::ArrayXd& xupw, const Eigen::ArrayXd& yupw, const Eigen::ArrayXd& xdownw, const Eigen::ArrayXd& ydownw);
    
    private:
        
        /// @brief allocate all inner structures and arrays
        /// @param rows count of rows
        /// @param cols count of cols
        void allocate_memory(const size_t &rows, const size_t &cols);

        /// @brief make acquaintances for neighbors
        void associate_neighbors();

        /// @brief allocation global arrays
        void allocation_velocity_space();

        /// @brief calculation of the time step
        double timestep();

        /// @brief calculation of the slope of distribution function
        void interpolation();

        /// @brief calculate the flux across the interfaces
        void flux_calculation(double dt);

        /// @brief updating of the cell averaged values
        std::tuple<Eigen::Array4d, Eigen::Array4d> update(double dt);

        /// @brief calculate dx dy slopes by solving linear least square system
        /// @param core central cell 
        void least_square_solver(cell& core);

        /// @brief calculate flux of boundary interface, assuming left wall
        /// @param bc   boundary condition
        /// @param face the boundary interface
        /// @param cell cell next to the boundary interface
        /// @param btype boundary type (WALL, INPUT, OUTPUT, MIRROR)
        void calc_flux_boundary(double dt, const boundary_cell& bc, cell_interface &face, const cell& cell, int side);
        
        /// @brief calculate flux of boundary interface
        /// @param bc   boundary condition
        /// @param face the boundary interface
        /// @param cell cell next to the boundary interface
        void calc_flux_boundary_wall(double dt, const boundary_cell& bc, cell_interface &face, const cell& cell, int side);
        
        /// @brief calculate flux of boundary for input conditions
        /// @param bc   boundary condition
        /// @param face the boundary interface
        /// @param cell cell next to the boundary interface
        void calc_flux_boundary_input(double dt, const boundary_cell& bc, cell_interface &face, const cell& cell, int side);
        
        /// @brief calculate flux of boundary for output
        /// @param bc   boundary condition
        /// @param face the boundary interface
        /// @param cell cell next to the boundary interface
        void calc_flux_boundary_output(double dt, const boundary_cell& bc, cell_interface &face, const cell& cell, int side);
        
        /// @brief calculate flux of boundary for axisymmetric one
        /// @param bc   boundary condition
        /// @param face the boundary interface
        /// @param cell cell next to the boundary interface
        void calc_flux_boundary_mirror(double dt, const boundary_cell& bc, cell_interface &face, const cell& cell, int side);


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
        [[nodiscard]] Eigen::Array4d moment_uv(const Eigen::Array<double, MNUM, 1> &Mu, const Eigen::Array<double, MTUM, 1> &Mv, const Eigen::Array<double, 3, 1> &Mxi,
                                               const int alpha, const int beta, const int delta);

        /// @brief calculate <a*u^\alpha*v^\beta*\psi>
        /// @param a          micro slope of Maxwellian
        /// @param Mu,Mv      <u^\alpha>,<v^\beta>
        /// @param Mxi        <\xi^l>
        /// @param alpha,beta exponential index of u and v
        /// @return moment of <a*u^\alpha*v^\beta*\psi>
        [[nodiscard]] Eigen::Array4d moment_au(const Eigen::Array4d &a,
                                               const Eigen::Array<double, MNUM, 1> &Mu, const Eigen::Array<double, MTUM, 1> &Mv, const Eigen::Array<double, 3, 1> &Mxi,
                                               const int alpha, const int beta);

        /// @brief calculate flux of inner interface
        /// @param cell_L cell left to the target interface
        /// @param face   the target interface
        /// @param cell_R cell right to the target interface
        /// @param idx    index indicating i or j direction
        void calc_flux(double dt, const cell &cell_L, cell_interface &face, const cell &cell_R, direction dir);

    };

    class block_solver
    {
        size_t cnt_iter = 0;
        double sitime = 0.;
        std::map<int, ugks::solver* > m_solver_blocks;
        double CFL = 0.8;
        ugks::precision siorder;

        public:
        block_solver() = default;
        block_solver(std::map<int, ugks::solver* > solver_blocks): m_solver_blocks(solver_blocks){}

        void make_association(int id, const std::pair<ugks::boundary_side, json>& association_rules);
        simulation_val solve();

        void write_results(const std::string& file_name = "cavity.dat") const;

    };

}

#endif