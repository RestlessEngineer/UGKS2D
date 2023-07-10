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

    std::string path_to_results = argv[1];
    auto pos_end = path_to_results.rfind(".dat");
    if(pos_end == std::string::npos){
        std::cout<<"wrong file type\n";
        return -1;
    }

    std::string postfix = "";
    auto pos = pos_end - 1;
    while(pos >= 0 && (std::isdigit(path_to_results[pos]) || path_to_results[pos] == '.'))
        postfix = path_to_results[pos--] + postfix;

    const double residual = 1e-5; 
    const double CFL = 0.5; // Courantт•±Friedrichsт•±Lewy number 
 
   
    ugks::physic_val phys;
    //TODO: temporary decision
    phys.gamma = ugks::tools::get_gamma(1); //ratio of specific heat  
     
    //create solver 
    ugks::solver ugks_solver(10, 10, phys, ugks::precision::SECOND_ORDER, CFL); 
     
    //init solver by last result
    ugks_solver.init_inner_values_by_result(path_to_results);
        
    double Theta = std::stod(postfix) /180.*M_PI; 
    //input velocity
    double Ma = 2.0;    
    double xA = -1.*std::cos(Theta) - 0.15; 
    double yA =  1.*std::sin(Theta) + 1.; 
    double xB = -1.*std::cos(Theta); 
    double yB =  yA; 
    double xC =  xA; 
    double yC = -yA; 
    double xD =  xB; 
    double yD = -yA; 
 
    //set geometry area. box 
    ugks_solver.set_geometry({{400.0*xA, 400.0*yA}, {400.0*xB, 400.0*yB}, {0., 1.* 400.0}, {400.0*1.25, 1. *400.0}},\ 
                             {{400.0*xC, 400.0*yC}, {400.0*xD, 400.0*yD}, {0., -1.*400.0}, {400.0*1.25, -1.*400.0}}); 
 
            
    // set boundary condition (density,u-velocity,v-velocity,lambda=1/temperature) 
    ugks_solver.set_boundary(ugks::boundary_side::LEFT, {1.0, Ma*sqrt(phys.gamma/2.), 0.0, 1.0}, ugks::boundary_type::INPUT); 
    ugks_solver.set_boundary(ugks::boundary_side::RIGHT, {1.0, 0.0, 0.0, 1.0}, ugks::boundary_type::OUTPUT); 
    ugks_solver.set_boundary(ugks::boundary_side::UP, {1.0, 0.0, 0.0, 1.0}, ugks::boundary_type::MIRROR);  
    ugks_solver.set_boundary(ugks::boundary_side::DOWN, {1.0, 0.0, 0.0, 1.0}, ugks::boundary_type::MIRROR); 
 
    ugks_solver.write_results("cavity_init_" + postfix + ".dat"); 
 
    std::cout<< "init of the task has been completed\n"<<std::endl; 
    while( true ){ 
 
        auto sim = ugks_solver.solve(); 
 
        auto max_res = std::max_element(sim.res.begin(), sim.res.end()); 
        //check if exit 
        if (*max_res < residual) 
            break; 
 
        if( sim.cnt_iter%10 == 0){ 
            std::cout << "iter: "<< sim.cnt_iter << 
             "; sitime: "<<sim.sitime <<  
            " dt: "<< sim.dt; 
            std::cout << "; res: "<< sim.res.transpose() << std::endl; 
            bool is_nan = false;
            for(auto res: sim.res)
                if(std::isnan(res))
                    is_nan = true;
            if(is_nan)
                break;
        } 
        if( sim.cnt_iter%500 == 0){ 
            std::cout<<"; write result from "<<sim.cnt_iter<<" iteration in cavity_temple_" + postfix + ".dat"<<std::endl; 
            ugks_solver.write_results("cavity_temple_" + postfix + ".dat"); 
        }

    } 
 
    ugks_solver.write_results("cavity_results_" + postfix + ".dat"); 
 
    return 0; 
 
}
