#include "tools.h"
#include <sstream>
#include <spdlog/sinks/basic_file_sink.h>
#include <iostream>

namespace ugks
{

    namespace tools
    {

        void maxwell_distribution(Eigen::ArrayXXd &h, Eigen::ArrayXXd &b,
                                  const Eigen::ArrayXXd &vn,
                                  const Eigen::ArrayXXd &vt, const Eigen::Array4d &prim, const int &DOF)
        {
            h = prim[0] * (prim[3] / M_PI) * exp(-prim[3] * ((vn - prim[1]) * (vn - prim[1]) + (vt - prim[2]) * (vt - prim[2])));
            b = h * DOF / (2.0 * prim[3]);
        }
        
        void shakhov_part(Eigen::ArrayXXd &H_plus, Eigen::ArrayXXd &B_plus,
                          const Eigen::ArrayXXd &H, const Eigen::ArrayXXd &B, const Eigen::ArrayXXd &vn,
                          const Eigen::ArrayXXd &vt, const std::array<double, 2> &qf,
                          const Eigen::Array4d &prim,
                          const double &Pr, const double &DOF)
        {

            H_plus = 0.8 * (1 - Pr) * std::pow(prim[3], 2) / prim[0] *
                     ((vn - prim[1]) * qf[0] + (vt - prim[2]) * qf[1]) * (2 * prim[3] * ((vn - prim[1]) * (vn - prim[1]) + (vt - prim[2]) * (vt - prim[2])) + DOF - 5) * H;
            B_plus = 0.8 * (1 - Pr) * std::pow(prim[3], 2) / prim[0] *
                     ((vn - prim[1]) * qf[0] + (vt - prim[2]) * qf[1]) * (2 * prim[3] * ((vn - prim[1]) * (vn - prim[1]) + (vt - prim[2]) * (vt - prim[2])) + DOF - 3) * B;
        }

        Eigen::Array4d get_conserved(const Eigen::Array4d &prim, const double &gamma)
        {
            Eigen::Array4d get_conserved;

            get_conserved[0] = prim[0];
            get_conserved[1] = prim[0] * prim[1];
            get_conserved[2] = prim[0] * prim[2];
            get_conserved[3] = 0.5 * prim[0] / prim[3] / (gamma - 1.0) +
                               0.5 * prim[0] * (prim[1] * prim[1] + prim[2] * prim[2]);
            return get_conserved;
        }

        Eigen::Array4d get_primary(const Eigen::Array4d &w, const double &gamma)
        {
            Eigen::Array4d get_prim;
            get_prim[0] = w[0];
            get_prim[1] = w[1] / w[0];
            get_prim[2] = w[2] / w[0];
            get_prim[3] = 0.5 * w[0] / (gamma - 1.0) / (w[3] - 0.5 * (w[1] * w[1] + w[2] * w[2]) / w[0]);
            return get_prim;
        }

        Eigen::Array4d frame_global(const Eigen::Array4d &w, const double &nx, const double &ny)
        {
            Eigen::Array4d global_frame;

            global_frame[0] = w[0];
            global_frame[1] = w[1] * nx - w[2] * ny;
            global_frame[2] = w[1] * ny + w[2] * nx;
            global_frame[3] = w[3];

            return global_frame;
        }

        Eigen::Array4d frame_local(const Eigen::Array4d &w, const double &nx, const double &ny)
        {

            Eigen::Array4d local_frame;

            local_frame[0] = w[0];
            local_frame[1] = w[1] * nx + w[2] * ny;
            local_frame[2] = w[2] * nx - w[1] * ny;
            local_frame[3] = w[3];

            return local_frame;
        }

        double get_gamma(const int &DOF)
        {
            return double(DOF + 4) / double(DOF + 2);
        }

        double get_sos(const Eigen::Array4d &prim, const double &gamma)
        {
            return std::sqrt(0.5 * gamma / prim[3]);
        }

        double get_tau(const Eigen::Array4d &prim, const double &mu_ref, const double &omega)
        {
            return mu_ref * 2 * std::pow(prim[3], (1 - omega) / prim[0]);
        }

        double get_temperature(const Eigen::ArrayXXd &h, const Eigen::ArrayXXd &b,
                               const Eigen::ArrayXXd &vn,
                               const Eigen::ArrayXXd &vt,
                               const Eigen::ArrayXXd &weight,
                               const Eigen::Array4d &prim, const double &DOF)
        {
            return 2.0 * ((weight * ((vn - prim[1]) * (vn - prim[1]) + (vt - prim[2]) * (vt - prim[2])) * h).sum() + (weight * b).sum()) / (DOF + 2) / prim[0];
        }

        double get_mu(const double &kn, const double &alpha, const double &omega)
        {
            return 5 * (alpha + 1) * (alpha + 2) * sqrt(M_PI) / (4 * alpha * (5 - 2 * omega) * (7 - 2 * omega)) * kn;
        }

        std::array<double, 2> get_heat_flux(const Eigen::ArrayXXd &h, const Eigen::ArrayXXd &b,
                                            const Eigen::ArrayXXd &vn,
                                            const Eigen::ArrayXXd &vt,
                                            const Eigen::ArrayXXd &weight,
                                            const Eigen::Array4d &prim)
        {

            std::array<double, 2> heat_flux; // heat flux in normal and tangential direction

            heat_flux[0] = 0.5 * ((weight * (vn - prim[1]) * ((vn - prim[1]) * (vn - prim[1]) + (vt - prim[2]) * (vt - prim[2])) * h).sum() +
                                  (weight * (vn - prim[1]) * b).sum());
            heat_flux[1] = 0.5 * ((weight * (vt - prim[2]) * ((vn - prim[1]) * (vn - prim[1]) + (vt - prim[2]) * (vt - prim[2])) * h).sum() +
                                  (weight * (vt - prim[2]) * b).sum());

            return heat_flux;
        }

        /// @brief set discrete velosity space using Newton�Cotes formulas
        /// @param num_u,num_v number of velosity points
        /// @param min_u,min_v smallest discrete velosity
        /// @param max_u,max_v largest discrete velosity
        std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, double, double>
        init_velosity_newton(const int &num_u, const double &min_u,
                             const double &max_u, const int &num_v, const double &min_v, const double &max_v)
        {

            // modify usize and vsize if not appropriate
            auto usize = (num_u / 4) * 4 + 1;
            auto vsize = (num_v / 4) * 4 + 1;

            Eigen::ArrayXXd uspace(vsize, usize);
            Eigen::ArrayXXd vspace(vsize, usize);
            Eigen::ArrayXXd weight(vsize, usize);

            // spacing in u and v velosity space
            const double du = (max_u - min_u) / (usize - 1);
            const double dv = (max_v - min_v) / (vsize - 1);

            /// Composite Boole's Rule
            auto newton_coeff = [](const int idx, const int num) -> double
            {
                if (idx == 0 || idx == num - 1)
                    return 14.0 / 45.0;
                else if ((idx + 1) % 2 == 0)
                    return 64.0 / 45.0;
                else if ((idx - 2) % 4 == 0)
                    return 24.0 / 45.0;
                else if ((idx - 4) % 4 == 0)
                    return 28.0 / 45.0;
                else
                    return 0; //! it has to be imposible
            };

            // velosity space
            for (int i = 0; i < vsize; ++i)
                for (int j = 0; j < usize; ++j)
                {
                    uspace(i, j) = min_u + i * du;
                    vspace(i, j) = min_v + j * dv;
                    weight(i, j) = (newton_coeff(i, vsize) * du) * (newton_coeff(j, usize) * dv);
                }

            // maximum micro velosity
            double umax = max_u;
            double vmax = max_v;
            return {uspace, vspace, weight, umax, vmax};
        }

        /// @brief set discrete velosity space using Gaussian-Hermite type quadrature
        /// @param umid,vmid middle value of the velosity space, zero or macroscopic velosity
        std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, double, double> init_velosity_gauss(const int &umid, const int &vmid)
        {

            Eigen::Array<double, 28, 1> vcoords, weights; // velosity points and weight for 28 points (symmetry)

            // set velosity points and weight
            vcoords = {-0.5392407922630e+01, -0.4628038787602e+01, -0.3997895360339e+01, -0.3438309154336e+01,
                       -0.2926155234545e+01, -0.2450765117455e+01, -0.2007226518418e+01, -0.1594180474269e+01,
                       -0.1213086106429e+01, -0.8681075880846e+00, -0.5662379126244e+00, -0.3172834649517e+00,
                       -0.1331473976273e+00, -0.2574593750171e-01, 0.2574593750171e-01, 0.1331473976273e+00,
                       0.3172834649517e+00, 0.5662379126244e+00, 0.8681075880846e+00, 0.1213086106429e+01,
                       0.1594180474269e+01, 0.2007226518418e+01, 0.2450765117455e+01, 0.2926155234545e+01,
                       0.3438309154336e+01, 0.3997895360339e+01, 0.4628038787602e+01, 0.5392407922630e+01};

            weights = {0.2070921821819e-12, 0.3391774320172e-09, 0.6744233894962e-07, 0.3916031412192e-05,
                       0.9416408715712e-04, 0.1130613659204e-02, 0.7620883072174e-02, 0.3130804321888e-01,
                       0.8355201801999e-01, 0.1528864568113e+00, 0.2012086859914e+00, 0.1976903952423e+00,
                       0.1450007948865e+00, 0.6573088665062e-01, 0.6573088665062e-01, 0.1450007948865e+00,
                       0.1976903952423e+00, 0.2012086859914e+00, 0.1528864568113e+00, 0.8355201801999e-01,
                       0.3130804321888e-01, 0.7620883072174e-02, 0.1130613659204e-02, 0.9416408715712e-04,
                       0.3916031412192e-05, 0.6744233894962e-07, 0.3391774320172e-09, 0.2070921821819e-12};

            // set grid number for u-velosity and v-velosity
            const auto usize = 28;
            const auto vsize = 28;

            Eigen::ArrayXXd uspace(vsize, usize);
            Eigen::ArrayXXd vspace(vsize, usize);
            Eigen::ArrayXXd weight(vsize, usize);

            // set velosity space and weight
            for (int i = 0; i < vsize; ++i)
                for (int j = 0; j < usize; ++j)
                {
                    vspace(i, j) = vmid + vcoords[i];
                    uspace(i, j) = umid + vcoords[j];
                    weight(i, j) = (weights[i] * std::exp(std::pow(vcoords[i], 2))) *
                                   (weights[j] * std::exp(std::pow(vcoords[j], 2)));
                }

            // determine maximum micro velosity
            double umax = vcoords.maxCoeff() + umid;
            double vmax = vcoords.maxCoeff() + vmid;

            return {uspace, vspace, weight, umax, vmax};
        }

        std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, double, double>
        get_velosity_space(const vel_space_param& param, integration integ)
        {
            switch (integ)
            {
            case integration::NEWTON_COTES:
                return init_velosity_newton(param.num_u, param.min_u, param.max_u, param.num_v, param.min_v, param.max_v);
            case integration::GAUSS:
                return init_velosity_gauss((param.min_u + param.max_u)*0.5,(param.min_v + param.max_v)*0.5);
            default:
                return init_velosity_gauss((param.min_u + param.max_u)*0.5,(param.min_v + param.max_v)*0.5);
            }
        }
    }

}