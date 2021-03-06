/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#ifndef GPFPREDICTION_H
#define GPFPREDICTION_H

#include <BayesFilters/ExogenousModel.h>
#include <BayesFilters/GaussianPrediction.h>
#include <BayesFilters/ParticleSet.h>
#include <BayesFilters/PFPrediction.h>
#include <BayesFilters/StateModel.h>

#include <memory>

namespace bfl {
    class GPFPrediction;
}


class bfl::GPFPrediction : public bfl::PFPrediction
{
public:
    GPFPrediction(std::unique_ptr<bfl::GaussianPrediction> gauss_pred) noexcept;

    GPFPrediction(GPFPrediction&& gpf_prediction) noexcept;

    virtual ~GPFPrediction() noexcept { };

    void setStateModel(std::unique_ptr<bfl::StateModel> state_model) override;

    bfl::StateModel& getStateModel() override;

protected:
    void predictStep(const bfl::ParticleSet& previous_particles, bfl::ParticleSet& predicted_particles) override;

    std::unique_ptr<bfl::GaussianPrediction> gaussian_prediction_;
};

#endif /* GPFPREDICTION_H */
