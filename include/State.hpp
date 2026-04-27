#ifndef STATE_HPP
#define STATE_HPP

#define MAX_PHASES 8

struct ConservativeState {
    double rho;
    double rhoU[3];
    double rhoE;
};

struct PrimitiveState {
    double rho;
    double u[3];
    double p;
    double sigma;
    double gammaEff;
    double piInfEff;
    double alpha[MAX_PHASES];
};

#endif /* STATE_HPP */
