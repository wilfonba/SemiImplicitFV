#ifndef SIMULATION_CONFIG_HPP
#define SIMULATION_CONFIG_HPP

namespace SemiImplicitFV {

struct SimulationConfig {
    int dim = 3;       // Spatial dimension (1, 2, or 3)
    int nGhost = 2;    // Ghost cell layers
};

} // namespace SemiImplicitFV

#endif // SIMULATION_CONFIG_HPP
