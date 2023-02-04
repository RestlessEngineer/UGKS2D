#include "global.h" 
#include "tools.h" 
#include "solver.h" 
#include <Eigen/Dense> 
#include <iostream> 
 
int main(int argc, char *argv[]){ 
 
    if(argc < 2){
        std::cout<<"wrong amount of arguments";
        return -1;
    }
    auto angle =  std::stof(argv[1]);
    std::string postfix = argv[1];

    const double residual = 1e-5; 
    const double CFL = 0.5; // Courantт•±Friedrichsт•±Lewy number 
 
    const double kn = 0.0025;     // 0.0025 Knudsen number in reference state 
    const double alpha_ref = 1.0; // coefficient in HS model 
    const double omega_ref = 0.81; // coefficient in HS model 
 
    ugks::physic_val phys; 
    phys.DOF = 1; 
    phys.gamma = ugks::tools::get_gamma(phys.DOF); //ratio of specific heat 
    phys.Pr = 2.0/3.0; 
    phys.omega = 0.81; 
    phys.mu_ref = ugks::tools::get_mu(kn, alpha_ref, omega_ref); //reference viscosity coefficient 
     
    //create solver 
    ugks::solver ugks_solver(331, 300, phys, ugks::precision::FIRST_ORDER, CFL); 
     
    double Theta = angle/180.*M_PI; 
    double Ma    = 2.0;           
    double xA    = -1.*std::cos(Theta) - 0.15; 
    double yA    =  1.*std::sin(Theta) + 1.; 
    double xB    = -1.*std::cos(Theta); 
    double yB    =  yA; 
    double xC    =  xA; 
    double yC    = -yA; 
    double xD    =  xB; 
    double yD    = -yA; 
 
    //set geometry area. box 
    ugks_solver.set_geometry({{400.*xA, 400.*yA}, {400.*xB, 400.*yB}, {0., 1.*400.}, {400.*1.25, 1.*400.}},\ 
                             {{400.*xC, 400.*yC}, {400.*xD, 400.*yD}, {0., -1.*400.}, {400.*1.25, -1.*400.}}); 
 
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
    ugks_solver.set_boundary(ugks::boundary_side::LEFT, {1.0, Ma*sqrt(phys.gamma/2.), 0.0, 1.0}, ugks::boundary_type::INPUT); 
    ugks_solver.set_boundary(ugks::boundary_side::RIGHT, {1.0, 0.0, 0.0, 1.0}, ugks::boundary_type::OUTPUT); 
    ugks_solver.set_boundary(ugks::boundary_side::UP, {1.0, 0.0, 0.0, 1.0}, ugks::boundary_type::MIRROR);  
    ugks_solver.set_boundary(ugks::boundary_side::DOWN, {1.0, 0.0, 0.0, 1.0}, ugks::boundary_type::MIRROR); 
 
    // initial condition (density,u-velocity,v-velocity,lambda=1/temperature) 
    ugks_solver.set_flow_field({1.0, Ma*sqrt(phys.gamma/2.), 0.0, 1.0}); 
     
    while( true ){ 
 
        auto sim = ugks_solver.solve(); 
 
        auto max_res = std::max_element(sim.res.begin(), sim.res.end()); 
        //check if exit 
        if (*max_res < residual) 
            break; 
 
        if( sim.cnt_iter%100 == 0){ 
            std::cout << "\riter: "<< sim.cnt_iter << 
             "; sitime: "<<sim.sitime <<  
            " dt: "<< sim.dt; 
            std::cout << "; res: "<< sim.res.transpose() << "\r" << std::flush; 
        } 
        if( sim.cnt_iter%500 == 0){ 
            std::cout<<"; write result from "<<sim.cnt_iter<<" iteration in cavity_temple_" + postfix + ".dat"<<std::endl; 
            ugks_solver.write_results("cavity_temple_" + postfix + ".dat"); 
        } 
 
    } 
 
    ugks_solver.write_results("cavity_results_" + postfix + ".dat"); 
 
    return 0; 
 
}