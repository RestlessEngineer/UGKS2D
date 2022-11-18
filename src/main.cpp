
#include "global.h"
#include "tools.h"
#include "solver.h"
#include <Eigen/Dense>
#include <iostream>


int main(int argc, char *argv[]){

    const double residual = 1e-5;
    const double CFL = 0.8;

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
    ugks::solver ugks_solver(45, 45, phys, ugks::precision::FIRST_ORDER, CFL);
    Eigen::Rotation2D<double> rot(0./180.*M_PI);
    Eigen::Vector2d p1 = {0.,0.};
    Eigen::Vector2d p2 = {0.,1.};
    Eigen::Vector2d p3 = {1.,1.};
    Eigen::Vector2d p4 = {1.,0.};
    Eigen::Vector2d uvec = {0.15,0.};

    Eigen::Vector2d rot_p1 = rot*p1;
    Eigen::Vector2d rot_p2 = rot*p2;
    Eigen::Vector2d rot_p3 = rot*p3;
    Eigen::Vector2d rot_p4 = rot*p4;
    Eigen::Vector2d rotu = rot*uvec;


    ugks::point pp1 = {rot_p1[0], rot_p1[1]};
    ugks::point pp2 = {rot_p2[0], rot_p2[1]};
    ugks::point pp3 = {rot_p3[0], rot_p3[1]};
    ugks::point pp4 = {rot_p4[0], rot_p4[1]};

    //set geometry area. box
    ugks_solver.set_geometry(pp1, pp2, pp3, pp4);


    //set velocity space param
    ugks::vel_space_param param;
    // largest discrete velocity
    param.max_u = 3;
    param.max_v = 3;
    // smallest discrete velocity
    param.min_u = -3;
    param.min_v = -3;
    // number of velocity points
    param.num_u = 25; 
    param.num_v = 25;

    ugks_solver.set_velocity_space(param, ugks::integration::NEWTON_COTES);

    // set boundary condition (density,u-velocity,v-velocity,lambda=1/temperature)
    ugks_solver.set_boundary({1.0, 0.0, 0.0, 1.0}, ugks::boundary::LEFT);
    ugks_solver.set_boundary({1.0, 0.0, 0.0, 1.0}, ugks::boundary::RIGHT);
    ugks_solver.set_boundary({1.0, rotu[0], rotu[1], 1.0}, ugks::boundary::UP); 
    ugks_solver.set_boundary({1.0, 0.0, 0.0, 1.0}, ugks::boundary::DOWN);

    // initial condition (density,u-velocity,v-velocity,lambda=1/temperature)
    ugks_solver.set_flow_field({1.0, 0.0, 0.0, 1.0});

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

    ugks_solver.write_results();

    return 0;

}