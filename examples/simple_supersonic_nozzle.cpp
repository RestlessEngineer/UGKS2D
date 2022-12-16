
#include "global.h"
#include "tools.h"
#include "solver.h"
#include <Eigen/Dense>
#include <iostream>

int main(int argc, char *argv[]){

    const double residual = 1e-5;
    const double CFL = 0.8; // Courant–Friedrichs–Lewy number

    const double kn = 0.1;        // 0.075 Knudsen number in reference state
    const double alpha_ref = 1.0; // coefficient in HS model
    const double omega_ref = 0.5; // coefficient in HS model

    ugks::physic_val phys;
    phys.DOF = 1;
    phys.gamma = ugks::tools::get_gamma(phys.DOF); //ratio of specific heat
    phys.Pr = 2.0/3.0;
    phys.omega = 0.72;
    phys.mu_ref = ugks::tools::get_mu(kn, alpha_ref, omega_ref); //reference viscosity coefficient

    //create solver
    ugks::solver ugks_solver(50,50, phys, ugks::precision::SECOND_ORDER, CFL);

    //set geometry area. box
    ugks_solver.set_geometry({{-1.13261, 1.18567}, {-0.982613, 1.18567}, {0., 1.}, {1.25, 1.}},\
                             {{-1.13261, -1.18567}, {-0.982613, -1.18567}, {0., -1.}, {1.25, -1.}});

    //set velocity space param
    ugks::vel_space_param param;
    // largest discrete velocity
    param.max_u = 5;
    param.max_v = 3;
    // smallest discrete velocity
    param.min_u = -1;
    param.min_v = -3;
    // number of velocity points
    param.num_u = 30; 
    param.num_v = 30;

    ugks_solver.set_velocity_space(param, ugks::integration::GAUSS);

    // set boundary condition (density,u-velocity,v-velocity,lambda=1/temperature)
    ugks_solver.set_boundary(ugks::boundary_side::LEFT, {1.0, 2.0, 0.0, 1.0}, ugks::boundary_type::INPUT);
    ugks_solver.set_boundary(ugks::boundary_side::RIGHT, {1.0, 0.0, 0.0, 1.0}, ugks::boundary_type::OUTPUT);
    ugks_solver.set_boundary(ugks::boundary_side::UP, {1.0, 0.0, 0.0, 1.0}, ugks::boundary_type::MIRROR); 
    ugks_solver.set_boundary(ugks::boundary_side::DOWN, {1.0, 0.0, 0.0, 1.0}, ugks::boundary_type::MIRROR);

    // initial condition (density,u-velocity,v-velocity,lambda=1/temperature)
    ugks_solver.set_flow_field({1.0, 2.0, 0.0, 1.0});
    
    while( true ){

        auto sim = ugks_solver.solve();

        auto max_res = std::max_element(sim.res.begin(), sim.res.end());
        //check if exit
        if (*max_res < residual)
            break;

        if( sim.cnt_iter%10 == 0){
            std::cout << "iter: "<< sim.cnt_iter <<
             " sitime: "<<sim.sitime << 
            " dt: "<< sim.dt << std::endl;
            std::cout << "res: "<< sim.res << std::endl;
        }

    }

    // ugks_solver.write_results();

    return 0;

}