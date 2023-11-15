
#include "solver.h"
#include <algorithm>
#include <limits>
#include <cfloat>
#include <cmath>


namespace ugks{

    simulation_val block_solver::solve(){
        double dt = std::numeric_limits<double>::max();
        
        std::for_each(m_solver_blocks.begin(), m_solver_blocks.end(),
        [&dt](auto& elem){
            double ldt = elem.second->timestep();
            dt = std::min(ldt, dt);
        });

        std::for_each(m_solver_blocks.begin(), m_solver_blocks.end(),
        [&dt](auto& elem){
            elem.second->interpolation();
            elem.second->flux_calculation(dt);
        });

        Eigen::Array4d sum_res = Eigen::Array4d::Zero(), sum_avg = Eigen::Array4d::Zero();

        std::for_each(m_solver_blocks.begin(), m_solver_blocks.end(),
        [&sum_res, &sum_avg, &dt](auto&& elem){
            auto [lsum_res, lsum_avg] = elem.second->update(dt);
            sum_res += lsum_res;
            sum_avg += lsum_avg;
        });

        // final residual
        auto res = sqrt(sum_res) / (sum_avg + DBL_EPSILON);

        cnt_iter++;
        sitime += dt;
        return {dt, sitime, cnt_iter, res, CFL, siorder};
    }

    void block_solver::make_association(int id, const std::pair<ugks::boundary_side, json> &association_rules)
    {
        auto [side_from, rules] = association_rules;
        auto *solver_from = m_solver_blocks[id];
        for (auto &rule : rules["Gluing"])
        {
            auto *solver_to = m_solver_blocks[rule["to"]["id"]];
            auto side_to = ugks::convert_to_side(rule["to"]["side"]);

            std::pair<int, int> range_to = rule["to"]["range"];
            std::pair<int, int> range_from = rule["range"];

            solver_to->associate_neighbors(solver_from->get_frontier(side_from, range_from),
                                           side_to, range_to);
        }
    }

    void block_solver::write_results(const std::string& file_name) const{
        
        for(auto &[id, solver] : m_solver_blocks){
            auto pos = file_name.rfind('.');
            auto name = file_name.substr(0, pos);
            auto extention = file_name.substr(pos);
            solver->write_results(name + '_' + std::to_string(id) + extention);
        }
    }

}

