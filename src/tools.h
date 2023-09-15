#ifndef TOOLS_H
#define TOOLS_H

#include "global.h"
#include <tuple>

namespace ugks
{

    namespace tools
    {

        ///@brief obtain discretized Maxwellian distribution
        ///@param h,b   distribution function
        ///@param vn,vt normal and tangential velocity
        ///@param prim  primary variables
        void maxwell_distribution(Eigen::ArrayXXd &h, Eigen::ArrayXXd &b,
                                  const Eigen::ArrayXXd &vn,
                                  const Eigen::ArrayXXd &vt, const Eigen::Array4d &prim, const int &DOF);

        ///@brief calculate the Shakhov part H^+, B^+
        ///@param H_plus,B_plus Shakhov part
        ///@param H,B           Maxwellian distribution function
        ///@param vn,vt         normal and tangential velocity
        ///@param qf            heat flux
        ///@param prim          primary variables
        void shakhov_part(Eigen::ArrayXXd &H_plus, Eigen::ArrayXXd &B_plus,
                          const Eigen::ArrayXXd &H, const Eigen::ArrayXXd &B, const Eigen::ArrayXXd &vn,
                          const Eigen::ArrayXXd &vt, const std::array<double, 2> &qf,
                          const Eigen::Array4d &prim,
                          const double &Pr, const double &DOF);

        ///@brief convert primary variables to conservative variables
        ///@param  prim primary variables
        ///@return conservative variables
        Eigen::Array4d get_conserved(const Eigen::Array4d &prim, const double &gamma);

        ///@brief convert conservative variables to primary variables
        ///@param w conservative variables
        ///@return conservative variables
        Eigen::Array4d get_primary(const Eigen::Array4d &w, const double &gamma);

        ///@brief convert macro variables from local frame to global
        ///@param w     macro variables in local frame
        ///@param cosa,sina directional cosine
        ///@return macro variables in global frame
        Eigen::Array4d frame_global(const Eigen::Array4d &w, const double &cosa, const double &sina);

        ///@brief convert macro variables from global frame to local
        ///@param w      macro variables in global frame
        ///@param cosa,sina  directional cosine
        ///@return macro variables in local frame
        Eigen::Array4d frame_local(const Eigen::Array4d &w, const double &cosa, const double &sina);

        ///@brief obtain ratio of specific heat
        ///@param DOF internal degree of freedom
        ///@return ratio of specific heat
        constexpr double get_gamma(const int &DOF)
        {
            return double(DOF + 4) / double(DOF + 2);
        }

        ///@brief obtain speed of sound
        ///@param prim primary variables
        ///@return speed of sound
        double get_sos(const Eigen::Array4d &prim, const double &gamma);

        ///@brief calculate collision time
        ///@param prim primary variables
        ///@return collision time
        double get_tau(const Eigen::Array4d &prim, const double &mu_ref, const double &omega);

        ///@brief get heat flux
        ///@param h,b   distribution function
        ///@param vn,vt normal and tangential velocity
        ///@param prim  primary variables
        ///@return heat flux in normal and tangential direction
        std::array<double, 2> get_heat_flux(const Eigen::ArrayXXd &h, const Eigen::ArrayXXd &b,
                                            const Eigen::ArrayXXd &vn,
                                            const Eigen::ArrayXXd &vt,
                                            const Eigen::ArrayXXd &weight,
                                            const Eigen::Array4d &prim);

        ///@brief get temperature
        ///@param h,b   distribution function
        ///@param vn,vt normal and tangential velocity
        ///@param weight weights for integration
        ///@param prim  primary variables
        ///@return temperature
        double get_temperature(const Eigen::ArrayXXd &h, const Eigen::ArrayXXd &b,
                               const Eigen::ArrayXXd &vn,
                               const Eigen::ArrayXXd &vt,
                               const Eigen::ArrayXXd &weight,
                               const Eigen::Array4d &prim, const double &DOF);

        ///@brief get the nondimensionalized viscosity coefficient
        ///@param kn Knudsen number
        ///@param alpha,omega indexes related to HS/VHS/VSS model
        ///@return nondimensionalized viscosity coefficient
        constexpr double get_mu(const double &kn, const double &alpha, const double &omega)
        {
            return 5 * (alpha + 1) * (alpha + 2) * sqrt(M_PI) / (4 * alpha * (5 - 2 * omega) * (7 - 2 * omega)) * kn;
        }

        ///@brief get the nondimensionalized viscosity coefficient
        ///@param integ type of integration
        ///@param param parameters for filling velocity space
        ///@return uspace, vspace, weights for integration velocity space
        std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, double, double>  
        get_velocity_space(const vel_space_param& param, integration integ);
    }

}
#endif