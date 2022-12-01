#include "solver.h"
#include "tools.h"

#include <iostream>
#include <fstream>
#include <float.h>
#include <limits>
#include <omp.h>

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

    template <typename T>
    static int __sgn(T val)
    {
        return (T(0) < val) - (val < T(0));
    }

    solver::solver(const size_t &rows, const size_t &cols) : ysize(rows), xsize(cols)
    {
        assert(rows > 2 && cols > 2);

        core.resize(ysize, xsize);      // cell centers
        vface.resize(ysize, xsize + 1); // vertical cell interface
        hface.resize(ysize + 1, xsize);

        associate_neighbors();
    }

    solver::solver(const size_t &rows, const size_t &cols, const physic_val &phys) : solver(rows, cols)
    {
        gamma = phys.gamma;
        DOF = phys.DOF;
        mu_ref = phys.mu_ref;
        omega = phys.omega;
        Pr = phys.Pr;
    }
    
    void solver::set_boundary(boundary_side side, const Eigen::Array4d bound, boundary_type type)
    {
        switch (side)
        {
        case boundary_side::LEFT:
            bc_L = bound;
            bc_typeL = type;
            break;
        case boundary_side::RIGHT:
            bc_R = bound;
            bc_typeR = type;
            break;
        case boundary_side::UP:
            bc_U = bound;
            bc_typeU = type;
            break;
        case boundary_side::DOWN:
            bc_D = bound;
            bc_typeD = type;
            break;
        default:
            break;
        }
    }

    void solver::timestep()
    {

        Eigen::Array4d prim; // primary variables

        double tmax = 0.0, sos;

        for (int i = 0; i < ysize; ++i)
            for (int j = 0; j < xsize; ++j)
            {

                // convert conservative variables to primary variables
                prim = tools::get_primary(core(i, j).w, gamma);

                // sound speed
                sos = tools::get_sos(prim, gamma);

                // maximum velocity
                prim[1] = std::max(umax, std::abs(prim[1])) + sos;
                prim[2] = std::max(vmax, std::abs(prim[2])) + sos;

                // maximum 1/dt allowed
                //TODO: make this better
                tmax = std::max(tmax,
                                (prim[1] + prim[2]) / std::sqrt(core(i, j).area));
            }

        // time step
        dt = CFL / tmax;
    //    dt = CFL*0.1e-2;
    }

    void solver::associate_neighbors(){
        // associate neighbors
        // boundaries
        // DOWN and UP
        for (int j = 1; j < xsize - 1; ++j)
        {
            // DOWN
            core(0, j).neighbors = {nullptr, nullptr, nullptr};
            core(0, j).neighbors[0] = &core(0, j - 1);
            core(0, j).neighbors[1] = &core(1, j);
            core(0, j).neighbors[2] = &core(0, j + 1);

            // UP
            core(ysize - 1, j).neighbors = {nullptr, nullptr, nullptr};
            core(ysize - 1, j).neighbors[0] = &core(ysize - 1, j - 1);
            core(ysize - 1, j).neighbors[1] = &core(ysize - 2, j);
            core(ysize - 1, j).neighbors[2] = &core(ysize - 1, j + 1);
        }

        // LEFT and RIGHT
        for (int i = 1; i < ysize - 1; ++i)
        {
            // LEFT
            core(i, 0).neighbors = {nullptr, nullptr, nullptr};
            core(i, 0).neighbors[0] = &core(i - 1, 0);
            core(i, 0).neighbors[1] = &core(i, 1);
            core(i, 0).neighbors[2] = &core(i + 1, 0);

            // RIGHT
            core(i, xsize - 1).neighbors = {nullptr, nullptr, nullptr};
            core(i, xsize - 1).neighbors[0] = &core(i - 1, xsize - 1);
            core(i, xsize - 1).neighbors[1] = &core(i, xsize - 2);
            core(i, xsize - 1).neighbors[2] = &core(i + 1, xsize - 1);
        }

        // core
        for (int i = 1; i < ysize - 1; ++i)
            for (int j = 1; j < xsize - 1; ++j)
            {   
                core(i, j).neighbors = {nullptr, nullptr, nullptr, nullptr};
                core(i, j).neighbors[0] = &core(i - 1, j);
                core(i, j).neighbors[1] = &core(i + 1, j);
                core(i, j).neighbors[2] = &core(i, j + 1);
                core(i, j).neighbors[3] = &core(i, j - 1);
            }

        //corners
        //LEFT DOWN
        core(0, 0).neighbors = {nullptr, nullptr, nullptr};
        core(0, 0).neighbors[0] = &core(1, 0);
        core(0, 0).neighbors[1] = &core(0, 1);
        core(0, 0).neighbors[2] = &core(1, 1);    

        //LEFT UP
        core(ysize-1, 0).neighbors = {nullptr, nullptr, nullptr};
        core(ysize-1, 0).neighbors[0] = &core(ysize - 2, 0);
        core(ysize-1, 0).neighbors[1] = &core(ysize - 1, 1);
        core(ysize-1, 0).neighbors[2] = &core(ysize - 2, 1);

        //RIGHT UP
        core(ysize-1, xsize-1).neighbors = {nullptr, nullptr, nullptr};
        core(ysize-1, xsize-1).neighbors[0] = &core(ysize - 2, xsize-1);
        core(ysize-1, xsize-1).neighbors[1] = &core(ysize - 1, xsize-2);
        core(ysize-1, xsize-1).neighbors[2] = &core(ysize - 2, xsize-2);

        //RIGHT DOWN
        core(ysize-1, 0).neighbors = {nullptr, nullptr, nullptr};
        core(ysize-1, 0).neighbors[0] = &core(1, xsize-1);
        core(ysize-1, 0).neighbors[1] = &core(0, xsize-2);
        core(ysize-1, 0).neighbors[2] = &core(1, xsize-2);


    }

    void solver::interpolation()
    {
        // no interpolation for first order
        if (siorder == precision::FIRST_ORDER)
            return;

        for (size_t i = 0; i < ysize; ++i)
            for (size_t j = 0; j < xsize; ++j)
                // solve LLS
                least_square_solver(core(i, j));
    }

    void solver::least_square_solver(cell &core)
    {

        const double TINY_VALUE = 1e-12; //for stabilize derivative

        //A matrix
        double A11 = 0, A12 = 0;
        double A21 = 0, A22 = 0;

        Eigen::ArrayXXd Bh1 = Eigen::ArrayXXd::Zero(vsize, usize);
        Eigen::ArrayXXd Bh2 = Eigen::ArrayXXd::Zero(vsize, usize);
        Eigen::ArrayXXd Bb1 = Eigen::ArrayXXd::Zero(vsize, usize);
        Eigen::ArrayXXd Bb2 = Eigen::ArrayXXd::Zero(vsize, usize);
        
        //for limiters
        Eigen::ArrayXXd MaxH = Eigen::ArrayXXd::Constant(vsize, usize, std::numeric_limits<double>::lowest());
        Eigen::ArrayXXd MaxB = Eigen::ArrayXXd::Constant(vsize, usize, std::numeric_limits<double>::lowest());
        Eigen::ArrayXXd MinH = Eigen::ArrayXXd::Constant(vsize, usize, std::numeric_limits<double>::max());
        Eigen::ArrayXXd MinB = Eigen::ArrayXXd::Constant(vsize, usize, std::numeric_limits<double>::max());
        
        Eigen::ArrayXXd CoeffH = Eigen::ArrayXXd::Constant(vsize, usize, 1.);
        Eigen::ArrayXXd CoeffB = Eigen::ArrayXXd::Constant(vsize, usize, 1.);

        for (auto &neighbor : core.neighbors)
        {
            A11 += std::pow(core.x - neighbor->x, 2);
            A12 += (core.x - neighbor->x) * (core.y - neighbor->y);
            A22 += std::pow(core.y - neighbor->y, 2);

            Bh1 += (core.x - neighbor->x) * (core.h - neighbor->h);
            Bh2 += (core.y - neighbor->y) * (core.h - neighbor->h);
            Bb1 += (core.x - neighbor->x) * (core.b - neighbor->b);
            Bb2 += (core.y - neighbor->y) * (core.b - neighbor->b);
        }

        A21 = A12;

        //LIMITERS:
        //for neighbors
        for (auto &neighbor : core.neighbors)
        {
            for (size_t i = 0; i < vsize; ++i)
                for (size_t j = 0; j < usize; ++j)
                {
                    MaxH(i, j) = std::max(neighbor->h(i, j), MaxH(i, j));
                    MaxB(i, j) = std::max(neighbor->b(i, j), MaxB(i, j));
                    
                    MinH(i, j) = std::min(neighbor->h(i, j), MinH(i, j));
                    MinB(i, j) = std::min(neighbor->b(i, j), MinB(i, j));
                }
        }

        //TODO: make the reference to the paper!
        auto limiter = [](double val)
        {
            return (val * val + 2 * val) / (val * val + val + 2);
        };

        //get limiter coefficient
        for (size_t i = 0; i < vsize; ++i)
            for (size_t j = 0; j < usize; ++j)
            {
                double minCoeffH = 1.;
                double minCoeffB = 1.;
                for (auto &neighbor : core.neighbors)
                {
                    double coH = 1, coB = 1;
                    if (neighbor->h(i, j) > core.h(i, j))
                    {   
                        double diff = neighbor->h(i, j) - core.h(i, j);
                        coH = (MaxH(i, j) - core.h(i, j)) / (diff + __sgn(diff)*TINY_VALUE);
                        coH = limiter(coH);
                    }
                    else if (neighbor->h(i, j) < core.h(i, j))
                    {   
                        double diff = neighbor->h(i, j) - core.h(i, j);
                        coH = (MinH(i, j) - core.h(i, j)) / (diff + __sgn(diff)*TINY_VALUE);
                        coH = limiter(coH);
                    }
                    minCoeffH = std::min(coH, minCoeffH);

                    if (neighbor->b(i, j) > core.b(i, j))
                    {
                        double diff = neighbor->b(i, j) - core.b(i, j);
                        coB = (MaxB(i, j) - core.b(i, j)) / (diff + __sgn(diff)*TINY_VALUE);
                        coB = limiter(coB);
                    }
                    else if (neighbor->b(i, j) < core.b(i, j))
                    {
                        double diff = neighbor->b(i, j) - core.b(i, j);
                        coB = (MinB(i, j) - core.b(i, j)) / (diff + __sgn(diff)*TINY_VALUE);
                        coB = limiter(coB);
                    }
                    minCoeffB = std::min(coB, minCoeffB);
                }

                CoeffH(i, j) = minCoeffH;
                CoeffB(i, j) = minCoeffB;
            }

        //solve system
        //create A matrix
        Eigen::Matrix2d A{{A11, A12},
                          {A21, A22}};

        for (size_t i = 0; i < vsize; ++i)
            for (size_t j = 0; j < usize; ++j)
            {
                Eigen::Vector2d bh{Bh1(i, j), Bh2(i, j)};
                Eigen::Vector2d bb{Bb1(i, j), Bb2(i, j)};

                Eigen::Vector2d h = A.colPivHouseholderQr().solve(bh);
                Eigen::Vector2d b = A.colPivHouseholderQr().solve(bb);
                
                core.sh[DX](i, j) = CoeffH(i, j) * h[0];
                core.sh[DY](i, j) = CoeffH(i, j) * h[1];

                core.sb[DX](i, j) = CoeffB(i, j) * b[0];
                core.sb[DY](i, j) = CoeffB(i, j) * b[1];
            }

    }
    

    void solver::flux_calculation()
    {      

        for (int j = 0; j < xsize; ++j)
        {
            calc_flux_boundary(bc_D, hface(0, j), core(0, j), bc_typeD);
            calc_flux_boundary(bc_U, hface(ysize, j), core(ysize - 1, j), bc_typeU);
        }
        
        for (int i = 1; i < ysize; ++i){
            #pragma omp parallel for
            for (int j = 0; j < xsize; ++j)
                calc_flux(core(i - 1, j), hface(i, j), core(i, j), direction::IDIR);
        }
        
        for (int i = 0; i < ysize; ++i)
        {
            calc_flux_boundary(bc_L, vface(i, 0), core(i, 0), bc_typeL);
            calc_flux_boundary(bc_R, vface(i, xsize), core(i, xsize - 1), bc_typeR);
        }

        for (int j = 1; j < xsize; ++j){
            #pragma omp parallel for
            for (int i = 0; i < ysize; ++i)
                calc_flux(core(i, j - 1), vface(i, j), core(i, j), direction::JDIR);
        }
    }

    void solver::update()
    {
        Eigen::ArrayXXd H_old(vsize, usize), B_old(vsize, usize);   // equilibrium distribution at t=t^n
        Eigen::ArrayXXd H(vsize, usize), B(vsize, usize);           // equilibrium distribution at t=t^{n+1}
        Eigen::ArrayXXd H_plus(vsize, usize), B_plus(vsize, usize); // Shakhov part
        Eigen::Array4d w_old;                                                                       // conservative variables at t^n
        Eigen::Array4d prim_old, prim;                                                              // primary variables at t^n and t^{n+1}
        Eigen::Array4d sum_res, sum_avg;
        std::array<double, 2> qf;
        double tau_old, tau; // collision time and t^n and t^{n+1}

        // set initial value
        res = {0.0};

        for (int i = 0; i < ysize; ++i)
            for (int j = 0; j < xsize; ++j)
            {
                // store W^n and calculate H^n,B^n,\tau^n
                w_old = core(i, j).w; // store W^n

                prim_old = tools::get_primary(w_old, gamma);                                              // convert to primary variables
                tools::maxwell_distribution(H_old, B_old, uspace, vspace, prim_old, DOF); // calculate Maxwellian
                tau_old = tools::get_tau(prim_old, mu_ref, omega);                                        // calculate collision time \tau^n

                // update W^{n+1} and calculate H^{n+1},B^{n+1},\tau^{n+1}
                core(i, j).w = core(i, j).w + (vface(i, j).flux - vface(i, j + 1).flux + hface(i, j).flux - hface(i + 1, j).flux) /
                                                  core(i, j).area; // update W^{n+1}

                prim = tools::get_primary(core(i, j).w, gamma);
                tools::maxwell_distribution(H, B, uspace, vspace, prim, DOF);
                tau = tools::get_tau(prim, mu_ref, omega);

                // record residual
                sum_res = sum_res + (w_old - core(i, j).w) * (w_old - core(i, j).w);
                sum_avg = sum_avg + core(i, j).w.abs();

                // Shakhov part
                // heat flux at t=t^n
                qf = tools::get_heat_flux(core(i, j).h, core(i, j).b, uspace, vspace, weight, prim_old);

                // h^+ = H+H^+ at t=t^n
                tools::shakhov_part(H_plus, B_plus, H_old, B_old, uspace, vspace, qf, prim_old, Pr, DOF); // H^+ and B^+
                H_old = H_old + H_plus;                                                                                   // h^+
                B_old = B_old + B_plus;                                                                                   // b^+

                // h^+ = H+H^+ at t=t^{n+1}
                tools::shakhov_part(H_plus, B_plus, H, B, uspace, vspace, qf, prim, Pr, DOF);
                H = H + H_plus;
                B = B + B_plus;

                // update distribution function
                core(i, j).h = (core(i, j).h + (vface(i, j).flux_h - vface(i, j + 1).flux_h + hface(i, j).flux_h - hface(i + 1, j).flux_h) / core(i, j).area +
                                0.5 * dt * (H / tau + (H_old - core(i, j).h) / tau_old)) /
                               (1.0 + 0.5 * dt / tau);
                core(i, j).b = (core(i, j).b + (vface(i, j).flux_b - vface(i, j + 1).flux_b + hface(i, j).flux_b - hface(i + 1, j).flux_b) / core(i, j).area +
                                0.5 * dt * (B / tau + (B_old - core(i, j).b) / tau_old)) /
                               (1.0 + 0.5 * dt / tau);
            }

        // final residual
        res = sqrt(xsize * ysize * sum_res) / (sum_avg + DBL_EPSILON);
    }

    void solver::set_geometry(const double &xlength, const double &ylength)
    {
        mesh.resize(ysize + 1, xsize + 1);

        // cell length and area
        const double dx = xlength / xsize;
        const double dy = ylength / ysize;
        const double area = dx * dy;

        for (int i = 0; i < ysize + 1; ++i)
            for (int j = 0; j < xsize + 1; ++j)
            { // mesh (node coordinate)
                mesh(i, j).y = i * dy;
                mesh(i, j).x = j * dx;
            }

        for (int i = 0; i < ysize; ++i)
            for (int j = 0; j < xsize; ++j)
            { // cell center
                core(i, j).y = (i + 0.5) * dy;
                core(i, j).x = (j + 0.5) * dx;
                core(i, j).area = area;
            }

        for (int i = 0; i < ysize; ++i)
            for (int j = 0; j < xsize + 1; ++j)
            { // vertical interface
                vface(i, j).length = dy;
                vface(i, j).cosa = 1.0;
                vface(i, j).sina = 0.0;
                vface(i, j).p = j*dx;
            }

        for (int i = 0; i < ysize + 1; ++i)
            for (int j = 0; j < xsize; ++j)
            { // horizontal interface
                hface(i, j).length = dx;
                hface(i, j).cosa = 0.0;
                hface(i, j).sina = 1.0;
                hface(i, j).p = i*dy;
            }
        
    }

    void solver::set_geometry(const point &ld, const point &lu, const point &ru, const point &rd)
    {

        mesh.resize(ysize + 1, xsize + 1);

        // create up wall
        // x
        Eigen::ArrayXd xupw(xsize + 1);
        xupw.setLinSpaced(lu.x, ru.x);
        // y
        Eigen::ArrayXd yupw(xsize + 1);
        yupw.setLinSpaced(lu.y, ru.y);

        // create down wall
        // x
        Eigen::ArrayXd xdownw(xsize + 1);
        xdownw.setLinSpaced(ld.x, rd.x);
        // y
        Eigen::ArrayXd ydownw(xsize + 1);
        ydownw.setLinSpaced(ld.y, rd.y);

        // fill mesh
        for (size_t j = 0; j < xsize + 1; ++j)
        {
            double dx = (xupw[j] - xdownw[j]) / ysize;
            double dy = (yupw[j] - ydownw[j]) / ysize;

            for (size_t i = 0; i < ysize + 1; ++i)
            {
                mesh(i, j).x = xdownw[j] + dx * i;
                mesh(i, j).y = ydownw[j] + dy * i;
            }
        }

        auto leng = [](point p1, point p2) -> double
        {
            return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p2.y - p1.y, 2));
        };

        // TODO: sure length dir dx dy
        for (size_t i = 0; i < ysize; ++i)
            for (size_t j = 0; j < xsize; ++j)
            { // cell center
                core(i, j).y = (mesh(i,j).y + mesh(i+1,j+1).y + mesh(i+1,j).y + mesh(i,j+1).y) * 0.25;
                core(i, j).x = (mesh(i,j).x + mesh(i+1,j+1).x + mesh(i+1,j).x + mesh(i,j+1).x) * 0.25;
                double area = 0.5 * ((mesh(i+1,j+1).x - mesh(i,j).x) * (mesh(i+1,j).y - mesh(i,j+1).y) -
                                         (mesh(i+1,j).x - mesh(i,j+1).x) * (mesh(i+1,j+1).y - mesh(i,j).y));
                core(i, j).area = area;
                //TODO: add throw here for area <= 0
            }

        // vertical interface
        for (size_t i = 0; i < ysize; ++i)
            for (size_t j = 0; j < xsize + 1; ++j)
            {
                double len = leng(mesh(i, j), mesh(i + 1, j));
                double x1 = mesh(i + 1, j).x;
                double x2 = mesh(i, j).x;
                double y1 = mesh(i + 1, j).y;
                double y2 = mesh(i, j).y;

                double a = x2 - x1;
                double b = y2 - y1;

                vface(i, j).length = len;

                assert(std::abs(a) > DBL_EPSILON || std::abs(b) > DBL_EPSILON);

                if (std::abs(a) < DBL_EPSILON && std::abs(b) > DBL_EPSILON)
                {
                    vface(i, j).cosa = 1.0;
                    vface(i, j).sina = 0.0;
                    vface(i, j).p = mesh(i, j).x;
                }
                else if (std::abs(b) < DBL_EPSILON && std::abs(a) > DBL_EPSILON)
                {
                    // TODO: hard mesh deformation, fix this case
                    assert(false);
                }
                else
                {   
                    double sqra_b = std::sqrt(a * a + b * b);
                    vface(i, j).cosa = -b / sqra_b;
                    vface(i, j).sina = a / sqra_b;
                    vface(i, j).p = -(x1 * b - y1 * a) / sqra_b;
                }
            }

        // horizontal interface
        for (size_t i = 0; i < ysize + 1; ++i)
            for (size_t j = 0; j < xsize; ++j)
            {
                double len = leng(mesh(i, j), mesh(i, j+1));
                double x1 = mesh(i, j).x;
                double x2 = mesh(i, j + 1).x;
                double y1 = mesh(i, j).y;
                double y2 = mesh(i, j + 1).y;

                double a = x2 - x1;
                double b = y2 - y1;
                hface(i, j).length = len;

                assert(std::abs(a) > DBL_EPSILON || std::abs(b) > DBL_EPSILON); // impossible mesh

                if (std::abs(b) < DBL_EPSILON && std::abs(a) > DBL_EPSILON)
                {
                    hface(i, j).cosa = 0.0;
                    hface(i, j).sina = 1.0;
                    hface(i, j).p = mesh(i, j).y;
                }
                else if (std::abs(a) < DBL_EPSILON && std::abs(b) > DBL_EPSILON)
                {
                    // TODO: hard mesh deformation, fix this case
                    assert(false);
                }
                else
                {
                    double sqra_b = std::sqrt(a * a + b * b);
                    hface(i, j).cosa = -b / sqra_b;
                    hface(i, j).sina = a / sqra_b;
                    hface(i, j).p = -(x1 * b - y1 * a) / sqra_b;
                }
            }
    }

    void solver::set_flow_field(const Eigen::Array4d &init_gas)
    {

        Eigen::ArrayXXd H(vsize, usize), B(vsize, usize); // reduced Maxwellian distribution functions

        // convert primary variables to conservative variables
        Eigen::Array4d w = tools::get_conserved(init_gas, gamma);

        // obtain discretized Maxwellian distribution H and B
        tools::maxwell_distribution(H, B, uspace, vspace, init_gas, DOF);

        // initial condition
        for (int i = 0; i < ysize; ++i)
            for (int j = 0; j < xsize; ++j)
            {
                core(i, j).w = w;
                core(i, j).h = H;
                core(i, j).b = B;
                core(i, j).sh[direction::IDIR] = 0.0;
                core(i, j).sh[direction::JDIR] = 0.0;
                core(i, j).sb[direction::IDIR] = 0.0;
                core(i, j).sb[direction::JDIR] = 0.0;
            }
    }

    void solver::set_velocity_space(const vel_space_param& param, integration integ)
    {
        // TODO: add ifdef c++17 block
        //!auto [uspace, vspace, weight, umax, vmax] = tools::get_velocity_space(integration::GAUSS);
        auto vel_spc = tools::get_velocity_space(param, integ);

        uspace = std::get<0>(vel_spc);
        vspace = std::get<1>(vel_spc);
        weight = std::get<2>(vel_spc);

        umax = std::get<3>(vel_spc);
        vmax = std::get<4>(vel_spc);

        usize = uspace.cols();
        vsize = uspace.rows();

        allocation_velocity_space();
    }

    void solver::allocation_velocity_space()
    {

        for (int i = 0; i < ysize; ++i)
            for (int j = 0; j < xsize; ++j)
            {

                core(i, j).h.resize(vsize, usize);
                core(i, j).b.resize(vsize, usize);
                core(i, j).sh[direction::IDIR].resize(vsize, usize);
                core(i, j).sh[direction::JDIR].resize(vsize, usize);
                core(i, j).sb[direction::IDIR].resize(vsize, usize);
                core(i, j).sb[direction::JDIR].resize(vsize, usize);
            }

        // cell interface
        for (int i = 0; i < ysize; ++i)
            for (int j = 0; j < xsize + 1; ++j)
            {

                vface(i, j).flux_h.resize(vsize, usize);
                vface(i, j).flux_b.resize(vsize, usize);
            }

        for (int i = 0; i < ysize + 1; ++i)
            for (int j = 0; j < xsize; ++j)
            {

                hface(i, j).flux_h.resize(vsize, usize);
                hface(i, j).flux_b.resize(vsize, usize);
            }
    }

    void solver::write_results() const
    {
        std::stringstream result;
        Eigen::ArrayXXd X(ysize, xsize);
        Eigen::ArrayXXd Y(ysize, xsize);
        Eigen::ArrayXXd RHO(ysize, xsize);
        Eigen::ArrayXXd U(ysize, xsize);
        Eigen::ArrayXXd V(ysize, xsize);
        Eigen::ArrayXXd T(ysize, xsize);
        Eigen::ArrayXXd P(ysize, xsize);
        Eigen::ArrayXXd QX(ysize, xsize);
        Eigen::ArrayXXd QY(ysize, xsize);

        // write header
        result << "VARIABLES = X\tY\tRHO\tU\tV\tT\tP\tQX\tQY\n";
        result << "ZONE  I = " << ysize << ", J = "<< xsize <<" DATAPACKING = BLOCK\n";

        for (int i = 0; i < ysize; ++i)
            for (int j = 0; j < xsize; ++j)
            {
                // primary variables
                auto prim = tools::get_primary(core(i, j).w, gamma);
                auto temp = tools::get_temperature(core(i, j).h, core(i, j).b, uspace, vspace, weight, prim, DOF);
                auto pressure = 0.5 * temp * prim[0];
                auto heat = tools::get_heat_flux(core(i, j).h, core(i, j).b, uspace, vspace, weight, prim);
                X(i, j) = core(i, j).x;
                Y(i, j) = core(i, j).y;
                RHO(i, j) = prim[0];
                U(i, j) = prim[1];
                V(i, j) = prim[2];
                T(i, j) = temp;
                P(i, j) = pressure;
                QX(i, j) = heat[0];
                QY(i, j) = heat[1];
            }

        result << X << '\n'
               << Y << '\n'
               << RHO << '\n'
               << U << '\n'
               << V << '\n'
               << T << '\n'
               << P << '\n'
               << QX << '\n'
               << QY;

        std::ofstream resfile;
        resfile.open ("cavity.dat");
        resfile << result.str().c_str();
        resfile.close();
    }

    void solver::calc_flux_boundary(const Eigen::Array4d &bc, cell_interface &face, const cell &cell, boundary_type type)
    {
        switch (type)
        {
        case boundary_type::WALL:
            calc_flux_boundary_wall(bc, face, cell);
            break;
        case boundary_type::INPUT:
            calc_flux_boundary_input(bc, face, cell);
            break;
        case boundary_type::OUTPUT:
            calc_flux_boundary_output(bc, face, cell);
            break;
        case boundary_type::MIRROR:
            calc_flux_boundary_mirror(bc, face, cell);
            break;
        default:
            break;
        }
    }

    void solver::calc_flux_boundary_wall(const Eigen::Array4d &bc, cell_interface &face, const cell &cell)
    {

        Eigen::ArrayXXd vn(vsize, usize), vt(vsize, usize); // normal and tangential micro velosity
        Eigen::ArrayXXd h(vsize, usize), b(vsize, usize);   // distribution function at the interface
        Eigen::ArrayXXd H0(vsize, usize), B0(vsize, usize); // Maxwellian distribution function
        Eigen::ArrayXXd delta(vsize, usize);                // Heaviside step function

        Eigen::Array4d prim; // boundary condition in local frame

        // convert the micro velocity to local frame
        vn = uspace * face.cosa + vspace * face.sina;
        vt = vspace * face.cosa - uspace * face.sina;
        
        // boundary condition in local frame
        prim = tools::frame_local(bc, face.cosa, face.sina);

        //take from normal equation of a line
        double H = cell.x*face.cosa + cell.y*face.sina - face.p;
        double dx = H * face.cosa;
        double dy = H * face.sina;

        // Heaviside step function. The rotation accounts for the right wall
        delta = (Eigen::sign(vn) * __sgn(H) + 1) / 2; // for REVERSE order H is negative

        // obtain h^{in} and b^{in}, rotation accounts for the right wall
        h = cell.h - (dx*cell.sh[DX] + dy*cell.sh[DY]);
        b = cell.b - (dx*cell.sb[DX] + dy*cell.sb[DY]);

        // calculate wall density and Maxwellian distribution
        double SF = (weight * vn * h * (1 - delta)).sum();
        double SG = (prim[3] / M_PI) * (weight * vn * exp(-prim[3] * ((vn - prim[1]) * (vn - prim[1]) + (vt - prim[2]) * (vt - prim[2]))) * delta).sum();

        prim[0] = -SF / SG;
        
        //get H0, B0
        tools::maxwell_distribution(H0, B0, vn, vt, prim, DOF);

        // distribution function at the boundary interface
        h = H0 * delta + h * (1 - delta);
        b = B0 * delta + b * (1 - delta);

        // calculate flux
        face.flux[0] = (weight * vn * h).sum();
        face.flux[1] = (weight * vn * vn * h).sum();
        face.flux[2] = (weight * vn * vt * h).sum();
        face.flux[3] = 0.5 * (weight * vn * ((vn * vn + vt * vt) * h + b)).sum();

        face.flux_h = vn * h;
        face.flux_b = vn * b;

        face.flux = tools::frame_global(face.flux, face.cosa, face.sina);

        // total flux
        face.flux = dt * face.length * face.flux;
        face.flux_h = dt * face.length * face.flux_h;
        face.flux_b = dt * face.length * face.flux_b;
    }


    Eigen::Array4d solver::micro_slope(const Eigen::Array4d &prim, const Eigen::Array4d &sw)
    {

        Eigen::Array4d micro_slope;

        micro_slope[3] = 4.0 * std::pow(prim[3], 2) / (DOF + 2) / prim[0] *
                         (2.0 * sw[3] - 2.0 * prim[1] * sw[1] - 2.0 * prim[2] * sw[2] + sw[0] * (std::pow(prim[1], 2) + std::pow(prim[2], 2) - 0.5 * (DOF + 2) / prim[3]));

        micro_slope[2] = 2.0 * prim[3] / prim[0] * (sw[2] - prim[2] * sw[0]) - prim[2] * micro_slope[3];
        micro_slope[1] = 2.0 * prim[3] / prim[0] * (sw[1] - prim[1] * sw[0]) - prim[1] * micro_slope[3];
        micro_slope[0] = sw[0] / prim[0] - prim[1] * micro_slope[1] - prim[2] * micro_slope[2] -
                         0.5 * (std::pow(prim[1], 2) + std::pow(prim[2], 2) + 0.5 * (DOF + 2) / prim[3]) * micro_slope[3];

        return micro_slope;
    }

    void solver::calc_moment_u(const Eigen::Array4d &prim,
                               Eigen::Array<double, MNUM, 1> &Mu, Eigen::Array<double, MTUM, 1> &Mv, Eigen::Array<double, 3, 1> &Mxi,
                               Eigen::Array<double, MNUM, 1> &Mu_L, Eigen::Array<double, MNUM, 1> &Mu_R)
    {

        // moments of normal velocity
        Mu_L[0] = 0.5 * erfc(-sqrt(prim[3]) * prim[1]);
        Mu_L[1] = prim[1] * Mu_L[0] + 0.5 * exp(-prim[3] * std::pow(prim[1], 2)) / sqrt(M_PI * prim[3]);
        Mu_R[0] = 0.5 * erfc(sqrt(prim[3]) * prim[1]);
        Mu_R[1] = prim[1] * Mu_R[0] - 0.5 * exp(-prim[3] * std::pow(prim[1], 2)) / sqrt(M_PI * prim[3]);

        for (int i = 2; i < MNUM; ++i)
        {
            Mu_L[i] = prim[1] * Mu_L[i - 1] + 0.5 * (i - 1) * Mu_L[i - 2] / prim[3];
            Mu_R[i] = prim[1] * Mu_R[i - 1] + 0.5 * (i - 1) * Mu_R[i - 2] / prim[3];
        }

        Mu = Mu_L + Mu_R;

        // moments of tangential velocity
        Mv[0] = 1.0;
        Mv[1] = prim[2];

        for (int i = 2; i < MTUM; ++i)
        {
            Mv[i] = prim[2] * Mv[i - 1] + 0.5 * (i - 1) * Mv[i - 2] / prim[3];
        }

        // moments of \xi
        Mxi[0] = 1.0;                                                    //<\xi^0>
        Mxi[1] = 0.5 * DOF / prim[3];                                    //<\xi^2>
        Mxi[2] = (DOF * DOF + 2.0 * DOF) / (4.0 * std::pow(prim[3], 2)); //<\xi^4>
    }

    [[nodiscard]] Eigen::Array4d solver::moment_uv(Eigen::Array<double, MNUM, 1> &Mu, Eigen::Array<double, MTUM, 1> &Mv, Eigen::Array<double, 3, 1> &Mxi,
                                                   const int alpha, const int beta, const int delta)
    {

        Eigen::Array4d moment_uv;

        moment_uv[0] = Mu[alpha] * Mv[beta] * Mxi[delta / 2];
        moment_uv[1] = Mu[alpha + 1] * Mv[beta] * Mxi[delta / 2];
        moment_uv[2] = Mu[alpha] * Mv[beta + 1] * Mxi[delta / 2];
        moment_uv[3] = 0.5 * (Mu[alpha + 2] * Mv[beta] * Mxi[delta / 2] + Mu[alpha] * Mv[beta + 2] * Mxi[delta / 2] + Mu[alpha] * Mv[beta] * Mxi[(delta + 2) / 2]);

        return moment_uv;
    }

    [[nodiscard]] Eigen::Array4d solver::moment_au(const Eigen::Array4d &a,
                                                   Eigen::Array<double, MNUM, 1> &Mu, Eigen::Array<double, MTUM, 1> &Mv, Eigen::Array<double, 3, 1> &Mxi,
                                                   const int alpha, const int beta)
    {

        Eigen::Array4d moment_au;

        moment_au = a[0] * moment_uv(Mu, Mv, Mxi, alpha + 0, beta + 0, 0) +
                    a[1] * moment_uv(Mu, Mv, Mxi, alpha + 1, beta + 0, 0) +
                    a[2] * moment_uv(Mu, Mv, Mxi, alpha + 0, beta + 1, 0) +
                    0.5 * a[3] * moment_uv(Mu, Mv, Mxi, alpha + 2, beta + 0, 0) +
                    0.5 * a[3] * moment_uv(Mu, Mv, Mxi, alpha + 0, beta + 2, 0) +
                    0.5 * a[3] * moment_uv(Mu, Mv, Mxi, alpha + 0, beta + 0, 2);

        return moment_au;
    }

    void solver::calc_flux(cell &cell_L, cell_interface &face, cell &cell_R, direction dir)
    {

        Eigen::ArrayXXd vn(vsize, usize), vt(vsize, usize);         // normal and tangential micro velosity
        Eigen::ArrayXXd h(vsize, usize), b(vsize, usize);           // distribution function at the interface
        Eigen::ArrayXXd H0(vsize, usize), B0(vsize, usize);         // Maxwellian distribution function
        Eigen::ArrayXXd H_plus(vsize, usize), B_plus(vsize, usize); // Shakhov part of the equilibrium distribution
        Eigen::ArrayXXd sh(vsize, usize), sb(vsize, usize);         // slope of distribution function at the interface
        Eigen::ArrayXXd delta(vsize, usize);                                        // Heaviside step function


        Eigen::Array4d w, prim; // conservative and primary variables at the interface

        std::array<double, 2> qf;  // heat flux in normal and tangential direction
        Eigen::Array4d sw;         // slope of W
        Eigen::Array4d aL, aR, aT; // micro slope of Maxwellian distribution, left,right and time.
        Eigen::Array<double, MNUM, 1> Mu, Mu_L, Mu_R;
        Eigen::Array<double, MTUM, 1> Mv;
        Eigen::Array3d Mxi;
        Eigen::Array4d Mau_0, Mau_L, Mau_R, Mau_T; //<u\psi>,<aL*u^n*\psi>,<aR*u^n*\psi>,<A*u*\psi>
        double tau;                                // collision time
        Eigen::Array<double, 5, 1> Mt;             // some time integration terms

        // convert the micro velocity to local frame
        vn = uspace * face.cosa + vspace * face.sina;
        vt = vspace * face.cosa - uspace * face.sina;

        // Heaviside step function
        delta = (Eigen::sign(vn) + 1) / 2;

        // reconstruct initial distribution
        //take from normal equation of a line
        double HL = cell_L.x*face.cosa + cell_L.y*face.sina - face.p;
        double HR = cell_R.x*face.cosa + cell_R.y*face.sina - face.p;
        double dx_L = HL*face.cosa;
        double dy_L = HL*face.sina;
        double dx_R = HR*face.cosa;
        double dy_R = HR*face.sina;        

        //TODO: input latex comment
        h = (cell_L.h + dx_L * cell_L.sh[DX] + dy_L * cell_L.sh[DY]) * delta +
            (cell_R.h + dx_R * cell_R.sh[DX] + dy_R * cell_R.sh[DY]) * (1 - delta);
        b = (cell_L.b + dx_L * cell_L.sb[DX] + dy_L * cell_L.sb[DY]) * delta +
            (cell_R.b + dx_R * cell_R.sb[DX] + dy_R * cell_R.sb[DY]) * (1 - delta);

        sh = cell_L.sh[dir] * delta + cell_R.sh[dir] * (1 - delta);
        sb = cell_L.sb[dir] * delta + cell_R.sb[dir] * (1 - delta);

        // obtain macroscopic variables (local frame)
        // conservative variables w_0
        w[0] = (weight * h).sum();
        w[1] = (weight * vn * h).sum();
        w[2] = (weight * vt * h).sum();
        w[3] = 0.5 * ((weight * (vn * vn + vt * vt) * h).sum() + (weight * b).sum());

        // convert to primary variables
        prim = tools::get_primary(w, gamma);

        // heat flux
        qf = tools::get_heat_flux(h, b, vn, vt, weight, prim);

        // calculate a^L,a^R
        sw = (w - tools::frame_local(cell_L.w, face.cosa, face.sina)) / HL; // left slope of W
        aL = micro_slope(prim, sw);                                                             // calculate a^L

        sw = (tools::frame_local(cell_R.w, face.cosa, face.sina) - w) / HR; // right slope of W
        aR = micro_slope(prim, sw);                                                             // calculate a^R

        // calculate time slope of W and A
        //<u^n>,<v^m>,<\xi^l>,<u^n>_{>0},<u^n>_{<0}
        calc_moment_u(prim, Mu, Mv, Mxi, Mu_L, Mu_R);

        Mau_L = moment_au(aL, Mu_L, Mv, Mxi, 1, 0); //<aL*u*\psi>_{>0}
        Mau_R = moment_au(aR, Mu_R, Mv, Mxi, 1, 0); //<aR*u*\psi>_{<0}

        sw = -prim[0] * (Mau_L + Mau_R); // time slope of W
        aT = micro_slope(prim, sw);      // calculate A

        // calculate collision time and some time integration terms
        tau = tools::get_tau(prim, mu_ref, omega);

        Mt[3] = tau * (1.0 - exp(-dt / tau));
        Mt[4] = -tau * dt * exp(-dt / tau) + tau * Mt[3];
        Mt[0] = dt - Mt[3];
        Mt[1] = -tau * Mt[0] + Mt[4];
        Mt[2] = dt * dt / 2.0 - tau * Mt[0];

        // calculate the flux of conservative variables related to g0
        Mau_0 = moment_uv(Mu, Mv, Mxi, 1, 0, 0);    //<u*\psi>
        Mau_L = moment_au(aL, Mu_L, Mv, Mxi, 2, 0); //<aL*u^2*\psi>_{>0}
        Mau_R = moment_au(aR, Mu_R, Mv, Mxi, 2, 0); //<aR*u^2*\psi>_{<0}
        Mau_T = moment_au(aT, Mu, Mv, Mxi, 1, 0);   //<A*u*\psi>

        face.flux = Mt[0] * prim[0] * Mau_0 + Mt[1] * prim[0] * (Mau_L + Mau_R) + Mt[2] * prim[0] * Mau_T;

        // calculate the flux of conservative variables related to g+ and f0
        // Maxwellian distribution H0 and B0
        tools::maxwell_distribution(H0, B0, vn, vt, prim, DOF);

        // Shakhov part H+ and B+
        tools::shakhov_part(H_plus, B_plus, H0, B0, vn, vt, qf, prim, Pr, DOF);

        // macro flux related to g+ and f0
        face.flux[0] = face.flux[0] +
                       Mt[0] * (weight * vn * H_plus).sum() +
                       Mt[3] * (weight * vn * h).sum() - Mt[4] * (weight * vn * vn * sh).sum();

        face.flux[1] = face.flux[1] +
                       Mt[0] * (weight * vn * vn * H_plus).sum() +
                       Mt[3] * (weight * vn * vn * h).sum() - Mt[4] * (weight * vn * vn * vn * sh).sum();

        face.flux[2] = face.flux[2] +
                       Mt[0] * (weight * vt * vn * H_plus).sum() +
                       Mt[3] * (weight * vt * vn * h).sum() - Mt[4] * (weight * vt * vn * vn * sh).sum();

        face.flux[3] = face.flux[3] +
                       Mt[0] * 0.5 * ((weight * vn * (vn * vn + vt * vt) * H_plus).sum() + (weight * vn * B_plus).sum()) +
                       Mt[3] * 0.5 * ((weight * vn * (vn * vn + vt * vt) * h).sum() + (weight * vn * b).sum()) -
                       Mt[4] * 0.5 * ((weight * vn * vn * (vn * vn + vt * vt) * sh).sum() + (weight * vn * vn * sb).sum());

        // calculate flux of distribution function
        face.flux_h = Mt[0] * vn * (H0 + H_plus) +
                      Mt[1] * vn * vn * (aL[0] * H0 + aL[1] * vn * H0 + aL[2] * vt * H0 + 0.5 * aL[3] * ((vn * vn + vt * vt) * H0 + B0)) * delta +
                      Mt[1] * vn * vn * (aR[0] * H0 + aR[1] * vn * H0 + aR[2] * vt * H0 + 0.5 * aR[3] * ((vn * vn + vt * vt) * H0 + B0)) * (1 - delta) +
                      Mt[2] * vn * (aT[0] * H0 + aT[1] * vn * H0 + aT[2] * vt * H0 + 0.5 * aT[3] * ((vn * vn + vt * vt) * H0 + B0)) +
                      Mt[3] * vn * h - Mt[4] * vn * vn * sh;

        face.flux_b = Mt[0] * vn * (B0 + B_plus) +
                      Mt[1] * vn * vn * (aL[0] * B0 + aL[1] * vn * B0 + aL[2] * vt * B0 + 0.5 * aL[3] * ((vn * vn + vt * vt) * B0 + Mxi[1] * H0)) * delta +
                      Mt[1] * vn * vn * (aR[0] * B0 + aR[1] * vn * B0 + aR[2] * vt * B0 + 0.5 * aR[3] * ((vn * vn + vt * vt) * B0 + Mxi[1] * H0)) * (1 - delta) +
                      Mt[2] * vn * (aT[0] * B0 + aT[1] * vn * B0 + aT[2] * vt * B0 + 0.5 * aT[3] * ((vn * vn + vt * vt) * B0 + Mxi[1] * H0)) +
                      Mt[3] * vn * b - Mt[4] * vn * vn * sb;

        face.flux = tools::frame_global(face.flux, face.cosa, face.sina);

        // total flux
        face.flux = face.length * face.flux;
        face.flux_h = face.length * face.flux_h;
        face.flux_b = face.length * face.flux_b;
    }

}
