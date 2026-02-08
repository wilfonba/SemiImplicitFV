#ifndef SIMULATION_CONFIG_HPP
#define SIMULATION_CONFIG_HPP

namespace SemiImplicitFV {

struct SimulationConfig {
    int dim = 3;               // Spatial dimension (1, 2, or 3)
    int nGhost = 2;            // Ghost cell layers
    int RKOrder = 1;           // Runge-Kutta time stepper order (1, 2, or 3)
    bool useIGR = false;       // Use information geometric regularization
    int step = 0;              // Current time step (for output purposes)
    bool semiImplicit = false; // Whether to use semi-implicit time stepping
};

} // namespace SemiImplicitFV

#endif // SIMULATION_CONFIG_HPP
