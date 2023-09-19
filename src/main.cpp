
#include "global.h"
#include "tools.h"
#include "solver.h"
#include <iostream>
#include <fstream>
#include <exception>
#include <vector>

#include <Eigen/Dense>
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

argparse::ArgumentParser arguments("UGKS");

void init_arguments_parser(argparse::ArgumentParser &arguments)
{

    arguments.add_argument("-i", "--init")
        .action([](const std::string &value) { // read a JSON file
            std::ifstream stream(value);
            json init;
            stream >> init;
            stream.close();
            return init;
        })
        .required()
        .help("Init file in .json format. It sets parameters for all programm");

    arguments.add_argument("-o", "--output")
        .default_value("cavity.dat")
        .help("Output file for results. All results will be write there in a .plt format");

    arguments.add_argument("-r", "--output-rate")
        .default_value(-1)
        .help("Rate defines how often output file will be written")
        .scan<'d', int>();

    arguments.add_argument("-s", "--sim-info-rate")
        .default_value(-1)
        .help("Rate defines how often calculation information will be in concole")
        .scan<'d', int>();
    
    arguments.add_argument("-t", "--threads-count")
        .default_value(0)
        .help("How many threads will be launched")
        .scan<'d', int>();
    

}


ugks::solver create_solver(const json& init_params)
{ 
        // calculation params
        double CFL = init_params["Calculation"]["CFL"]; // Courant–Friedrichs–Lewy number
        double residual = init_params["Calculation"]["residual"];
        ugks::precision prec = init_params["Calculation"]["Precision"] == 1 ? ugks::precision::FIRST_ORDER : ugks::precision::SECOND_ORDER;

        // physic parameters
        double kn = init_params["Physic"]["Knudsen"];         // Knudsen number in reference state
        double alpha_ref = init_params["Physic"]["AlphaRef"]; // coefficient in HS model
        double omega_ref = init_params["Physic"]["OmegaRef"]; // coefficient in HS model

        ugks::physic_val phys;
        phys.DOF = init_params["Physic"]["DOF"];
        phys.Pr = init_params["Physic"]["Prantl"];
        phys.omega = init_params["Physic"]["Omega"];
        phys.gamma = ugks::tools::get_gamma(phys.DOF);               // ratio of specific heat
        phys.mu_ref = ugks::tools::get_mu(kn, alpha_ref, omega_ref); // reference viscosity coefficient

        // geometry
        auto gsize = init_params["Geometry"]["size"];
        auto RawUpVal = init_params["Geometry"]["UpWall"];
        auto RawDownVal = init_params["Geometry"]["DownWall"];
        std::vector<ugks::point> upWall, downWall;
        auto insert_point = [](auto x) -> ugks::point
        { return {x[0], x[1]};};

        std::transform(RawUpVal.cbegin(), RawUpVal.cend(),
                       std::back_inserter(upWall), insert_point);

        std::transform(RawDownVal.cbegin(), RawDownVal.cend(),
                       std::back_inserter(downWall), insert_point);

        ugks::solver ugks_solver(gsize[0], gsize[1], phys, prec, CFL);
        ugks_solver.set_geometry(upWall, downWall);

        // velocity space
        // set velocity space param
        ugks::vel_space_param param;
        // largest discrete velocity
        auto vsize = init_params["VelocitySpace"]["size"];
        auto urange = init_params["VelocitySpace"]["XSpaceRange"];
        auto vrange = init_params["VelocitySpace"]["YSpaceRange"];

        std::string integ_name = init_params["VelocitySpace"]["IntegrationMethod"];

        ugks::integration integ;
        if (integ_name == "NEWTON_COTES")
        {
            integ = ugks::integration::NEWTON_COTES;
        }
        else if (integ_name == "GAUSS")
        {
            integ = ugks::integration::GAUSS;
        }
        else
        {
            throw std::invalid_argument("wrong integration type: " + integ_name);
        }

        param.max_u = urange[1];
        param.max_v = vrange[1];
        // smallest discrete velocity
        param.min_u = urange[0];
        param.min_v = vrange[0];
        // number of velocity points
        param.num_u = vsize[1];
        param.num_v = vsize[0];

        ugks_solver.set_velocity_space(param, integ);

        // boundary
        auto boundary_init = [&init_params](const std::string& side) -> std::tuple<ugks::boundary_type, Eigen::Array4d>
        {
            std::string stype = init_params["Boundaries"][side]["Type"];
            ugks::boundary_type type;
            if(stype == "WALL")
                type = ugks::boundary_type::WALL;
            else if(stype == "INPUT")
                type = ugks::boundary_type::INPUT;
            else if(stype == "MIRROR")
                type = ugks::boundary_type::MIRROR;
            else if(stype == "OUTPUT")
                type = ugks::boundary_type::OUTPUT;
            else
                throw std::invalid_argument("wrong boundary type: " + stype);
            

            auto in_bound = init_params["Boundaries"][side]["Init"]; 
            Eigen::Array4d bound;
            std::copy(in_bound.begin(), in_bound.end(), bound.begin());
            return {type, bound};
        };

        auto [ltype, lbound] = boundary_init("LEFT");
        auto [rtype, rbound] = boundary_init("RIGHT");
        auto [utype, ubound] = boundary_init("UP");
        auto [dtype, dbound] = boundary_init("DOWN");

        ugks_solver.set_boundary(ugks::boundary_side::LEFT,  lbound, ltype);
        ugks_solver.set_boundary(ugks::boundary_side::RIGHT, rbound, rtype);
        ugks_solver.set_boundary(ugks::boundary_side::UP,    ubound, utype);
        ugks_solver.set_boundary(ugks::boundary_side::DOWN,  dbound, dtype);

        // initial condition (density,u-velocity,v-velocity,lambda=1/temperature)
        auto in_field = init_params["FlowField"]; 
        Eigen::Array4d field;
        std::copy(in_field.begin(), in_field.end(), field.begin());
        ugks_solver.set_flow_field(field);
        
        return ugks_solver;
}


int main(int argc, char *argv[]){

    init_arguments_parser(arguments);

    try
    {
        arguments.parse_args(argc, argv);
        
        //create solver
        json init_params = arguments.get<json>("--init")[0];
        auto ugks_solver = create_solver(init_params);

        auto output_file = arguments.get<std::string>("--output");
        auto output_rate = arguments.get<int>("--output-rate");
        auto sim_info_rate = arguments.get<int>("--sim-info-rate");

        int threads_count = init_params["Calculation"]["ThreadsCount"];
        double residual = init_params["Calculation"]["residual"];
        if(arguments.get<int>("--threads-count") > 0)
            threads_count = arguments.get<int>("--threads-count");
        
        std::cout<<"count threads: "<<threads_count<<std::endl;
        omp_set_num_threads(threads_count);

        while (true)
        {
            auto sim = ugks_solver.solve();

            auto max_res = std::max_element(sim.res.begin(), sim.res.end());

            // check is nan
            bool is_nan = std::accumulate(sim.res.begin(), sim.res.end(), false,
                                          [](auto &&val1, auto &&val2)
                                          { return std::isnan(val1) || std::isnan(val2); });
            if (is_nan)
            {
                std::cout << "nan was detected; res: " << sim.res.transpose() << std::endl;
                return -1;
            }
            // check if exit
            if (*max_res < residual)
                break;

            if (sim_info_rate > 0 && sim.cnt_iter % sim_info_rate == 0)
            {
                std::cout << sim << std::endl;
            }
            if (output_rate > 0 && sim.cnt_iter % output_rate == 0)
            {   
                std::string temp_name = output_file;
                temp_name.insert(temp_name.rfind('.'), "_temp");
                std::cout <<"write result into "<<temp_name<<std::endl;
                ugks_solver.write_results(temp_name);
            }
        }

        ugks_solver.write_results(output_file);

    }
    catch (const std::runtime_error &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << arguments;
        std::exit(1);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error reading or parsing JSON: " << e.what() << std::endl;
        std::exit(1);
    }

    return 0;

}