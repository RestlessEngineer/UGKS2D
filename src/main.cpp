
#include "global.h"
#include "tools.h"
#include "solver.h"
#include <iostream>
#include <fstream>
#include <exception>
#include <vector>
#include <map>

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
        .default_value(1)
        .help("How many threads will be launched")
        .scan<'d', int>();
    
    arguments.add_argument("-p", "--precision")
        .default_value(1e-5)
        .help("Residual of the model")
        .scan<'g', double>();
    

}

auto read_coords = [](std::string file_name) -> std::tuple<Eigen::ArrayXd, Eigen::ArrayXd>

{
    std::ifstream file(file_name);
    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Error opening the file." << std::endl;
        return {{},{}};
    }

    std::string line;
    double num;
    std::vector<double> x, y;
    
    std::getline(file, line);
    {
        std::istringstream iss(line);
        while (iss >> num)
            x.push_back(num);
    }
    std::getline(file, line);
    {
        std::istringstream iss(line);
        while (iss >> num)
            y.push_back(num);
    }

    assert(x.size() == y.size());
    Eigen::ArrayXd xw(x.size()), yw(y.size());
    std::copy(x.begin(), x.end(), xw.begin());
    std::copy(y.begin(), y.end(), yw.begin());
    return {xw, yw};
};


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

        ugks::solver ugks_solver(gsize[0], gsize[1], phys, prec, CFL);
        
        try
        {   //TODO: fix at
            init_params["Geometry"].at("load");
            std::string file_name_up = init_params["Geometry"]["UpWall"];
            auto [xup, yup] = read_coords(file_name_up);
            std::string file_name_down = init_params["Geometry"]["DownWall"];
            auto [xdown, ydown] = read_coords(file_name_down);
            ugks_solver.fill_mesh(xup, yup, xdown, ydown);
        }    
        catch (json::out_of_range &e)
        {
            auto RawUpVal = init_params["Geometry"]["UpWall"];
            auto RawDownVal = init_params["Geometry"]["DownWall"];
            auto insert_point = [](auto x) -> ugks::point
            { return {x[0], x[1]};};
            
            std::vector<ugks::point> upWall, downWall;
            std::transform(RawUpVal.cbegin(), RawUpVal.cend(),
                           std::back_inserter(upWall), insert_point);
            std::transform(RawDownVal.cbegin(), RawDownVal.cend(),
                           std::back_inserter(downWall), insert_point);

            ugks_solver.set_geometry(upWall, downWall);
        }

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
        using Boundary = Eigen::Array<ugks::boundary_cell, -1, 1>;
        auto boundary_init = [&init_params, &ugks_solver](const std::string& side, size_t size) -> Boundary
        {
            using Boundary = Eigen::Array<ugks::boundary_cell, -1, 1>;
            using Range = std::pair<size_t, size_t>; 
            Boundary bound; bound.resize(size);

            std::string stype = init_params["Boundaries"][side]["Type"];
            
            auto chose_type = [](std::string stype){
                ugks::boundary_type type;
                if(stype == "WALL")
                    type = ugks::boundary_type::WALL;
                else if(stype == "INPUT")
                    type = ugks::boundary_type::INPUT;
                else if(stype == "MIRROR")
                    type = ugks::boundary_type::MIRROR;
                else if(stype == "OUTPUT")
                    type = ugks::boundary_type::OUTPUT;
                else if(stype == "GLUE")
                    type = ugks::boundary_type::GLUE;
                else if(stype == "ROTATION")
                    type = ugks::boundary_type::ROTATION;
                else
                    throw std::invalid_argument("wrong boundary type: " + stype);
                return type;
            };

            auto set_boundary = [&bound](Range range, const ugks::boundary_cell& cell){
                std::for_each(bound.begin() + range.first, bound.begin() + range.second, [&cell](auto& val){
                    val = cell;
                });
            };

            auto get_boundary_init = [](const json& init){
                Eigen::Array4d bound_el;
                std::copy(init.begin(), init.end(), bound_el.begin());
                return bound_el;   
            };

            auto set_rotation_boundary = [&bound, &ugks_solver, &get_boundary_init]
            (const json& boundary, ugks::boundary_side side, Range range){
                    auto init_val = get_boundary_init(boundary["Init"]);
                    double w = boundary["w"];
                    double R = boundary["R"];
                    int sign = boundary["direction"] == "back" ? -1 : 1;
                    auto wall = ugks_solver.get_boundary_points(side, range);
                    std::cout<<"boundary side: "<<  static_cast<std::underlying_type<ugks::boundary_side>::type>(side) << std::endl;
                    for(int i = 0, j = range.first; j < range.second; ++i, ++j){
                        double sqr = sqrt(
                            (wall[i+1].x - wall[i].x)*(wall[i+1].x - wall[i].x) +
                            (wall[i+1].y - wall[i].y)*(wall[i+1].y - wall[i].y));
                        
                        double cosb = sign*(wall[i+1].x - wall[i].x)/sqr;
                        double sinb = sign*(wall[i+1].y - wall[i].y)/sqr;
                        init_val[1] = cosb*w*R;
                        init_val[2] = sinb*w*R;
                        bound[j] = {init_val, ugks::boundary_type::WALL};
                        std::cout<< i <<", "<< j << ") sqr: "<< sqr <<" cosb: " << cosb <<" sinb: "<< sinb << std::endl;
                    } 
            };


            if(stype == "MIXED"){
                for(auto& fragment : init_params["Boundaries"][side]["fragments"]){
                    Range range = fragment["range"];
                    auto type = chose_type(fragment["Type"]);
                    if(type == ugks::boundary_type::ROTATION){
                        set_rotation_boundary(fragment, ugks::convert_to_side(side), range);
                    }else{
                        auto init_val = get_boundary_init(fragment["Init"]);
                        set_boundary(range, {init_val, type});
                    }
                }
                return bound;
            }

            auto type = chose_type(stype);
            if(type == ugks::boundary_type::ROTATION){
                set_rotation_boundary(init_params["Boundaries"][side], ugks::convert_to_side(side), {0, size});
                return bound;
            }
            
            auto init_val = get_boundary_init(init_params["Boundaries"][side]["Init"]); 
            set_boundary({0, size}, {init_val, type});
            
            return bound;
        };

        auto lbound = boundary_init("LEFT", gsize[0]);
        auto rbound = boundary_init("RIGHT", gsize[0]);
        auto ubound = boundary_init("UP", gsize[1]);
        auto dbound = boundary_init("DOWN", gsize[1]);

        ugks_solver.set_boundary(ugks::boundary_side::LEFT,  lbound);
        ugks_solver.set_boundary(ugks::boundary_side::RIGHT, rbound);
        ugks_solver.set_boundary(ugks::boundary_side::UP,    ubound);
        ugks_solver.set_boundary(ugks::boundary_side::DOWN,  dbound);

        // initial condition (density,u-velocity,v-velocity,lambda=1/temperature)
        try
        {
            std::string file_name = init_params.at("RestartData");
            ugks_solver.init_inner_values_by_result(file_name);
            return ugks_solver;
        }
        catch (json::out_of_range &e)
        {
        }

        auto in_field = init_params["FlowField"]; 
        Eigen::Array4d field;
        std::copy(in_field.begin(), in_field.end(), field.begin());
        ugks_solver.set_flow_field(field);
        
        return ugks_solver;
}


ugks::block_solver create_block_solver(const json& init_params)
{   

        std::map<int, ugks::solver* > solver_blocks;
        std::multimap<int, std::pair<ugks::boundary_side ,json>> association;
        auto make_association = [&association](const json& Boundary, int id, std::string side){
            if(Boundary[side]["Type"] == "MIXED"){
                for(auto & fragment : Boundary[side]["fragments"]){
                    if(fragment["Type"] == "GLUE")
                        association.insert({id, {ugks::convert_to_side(side), fragment}});
                }
            }
            else if(Boundary[side]["Type"] == "GLUE")
                association.insert({id, {ugks::convert_to_side(side), Boundary[side]}});
        };

        for(const auto & param: init_params["blocks"]){
            int id = param["id"];
            ugks::solver* solver = new ugks::solver(create_solver(param));
            solver_blocks[id] = solver;
            make_association(param["Boundaries"], id, "UP");
            make_association(param["Boundaries"], id, "DOWN");
            make_association(param["Boundaries"], id, "LEFT");
            make_association(param["Boundaries"], id, "RIGHT");            
        }

        ugks::block_solver block_solver = ugks::block_solver(solver_blocks);
        
        for(auto && [id, solver] : solver_blocks)
            for(auto rule = association.lower_bound(id), last_rule = association.upper_bound(id); rule != last_rule; rule++)
                block_solver.make_association(id, rule->second);

        return block_solver;
}




int main(int argc, char *argv[]){

    init_arguments_parser(arguments);

    try
    {
        arguments.parse_args(argc, argv);
        
        //create solver
        // std::cout << arguments.get<json>("--init").size() << std::endl;
        json init_params = arguments.get<json>("--init")[0];
        //TODO: temprorary 
        ugks::block_solver ugks_solver_block;
        ugks::solver ugks_solver;
        bool is_block_solver = false; 
        
        try{
            init_params.at("blocks");
            ugks_solver_block = create_block_solver(init_params);
            is_block_solver = true;
        }
        catch(const json::out_of_range& e){
            std::cout << "programm with one block " << e.what() <<std::endl;
        }

        //TODO: temprorary 
        if(!is_block_solver)
            ugks_solver = create_solver(init_params);
        
        auto output_file = arguments.get<std::string>("--output");
        auto output_rate = arguments.get<int>("--output-rate");
        auto sim_info_rate = arguments.get<int>("--sim-info-rate");

        int threads_count = arguments.get<int>("--threads-count");
        double residual = arguments.get<double>("--precision");

        std::cout<<"count threads: "<<threads_count<<std::endl;
        std::cout<<"residual: "<<residual<<std::endl;

        omp_set_num_threads(threads_count);
        
        while (true)
        {
            // TODO: temprorary decision
            auto sim = is_block_solver ? ugks_solver_block.solve() : ugks_solver.solve();

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
                if(is_block_solver)
                    ugks_solver_block.write_results(temp_name);
                else
                    ugks_solver.write_results(temp_name);
            }
        }
        std::cout<<"finish the calculations..."<<std::endl;
        std::cout<<"write final resulr into: "<< output_file << std::endl;
        if(is_block_solver)
            ugks_solver_block.write_results(output_file);
        else
            ugks_solver.write_mesh(output_file);

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