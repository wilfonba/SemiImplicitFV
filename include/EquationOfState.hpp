#ifndef EQUATION_OF_STATE_HPP
#define EQUATION_OF_STATE_HPP

#include "State.hpp"
#include "SimulationConfig.hpp"

enum EOSType {
    EOS_IDEAL_GAS,
    EOS_STIFFENED_GAS
};

struct EOSData {
    enum EOSType type;
    double gamma;
    double pInf;
    double R;
    int dim;
};

EOSData eos_create_ideal_gas(double gamma, double R, int dim);
EOSData eos_create_stiffened_gas(double gamma, double pInf, double R, int dim);

#pragma omp declare target
double eos_pressure(const EOSData* eos, const ConservativeState* U);
double eos_temperature(const EOSData* eos, const PrimitiveState* W);
double eos_sound_speed(const EOSData* eos, const PrimitiveState* W);
double eos_internal_energy(const EOSData* eos, const PrimitiveState* W);
double eos_total_energy(const EOSData* eos, const PrimitiveState* W);
PrimitiveState eos_to_primitive(const EOSData* eos, const ConservativeState* U);
ConservativeState eos_to_conservative(const EOSData* eos, const PrimitiveState* W);
#pragma omp end declare target

#endif /* EQUATION_OF_STATE_HPP */
