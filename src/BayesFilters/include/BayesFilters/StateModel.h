/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

/*
StateModel represents a generic x_k = f(x_{k-1}, u_{k-1}, w_{k-1})
Implemented AdditiveStateModel representing x_k = f(x_{k-1}, u_{k-1}) + w_{k-1} with w white Gaussian noise
Implemented LinearStateModel inheriting from AdditiveStateModel, representing x_k = F_{k-1} x_{k-1} + w_{k-1}
Implemented LTIStateModel inheriting from LinearStateModel, representing x_k = F x_{k-1} + w_{k-1}
*/

#ifndef STATEMODEL_H
#define STATEMODEL_H

#include <Eigen/Dense>

namespace bfl {
    class StateModel;
}


class bfl::StateModel
{
public:
    virtual ~StateModel() noexcept { };

    virtual void propagate(const Eigen::Ref<const Eigen::MatrixXd>& cur_states, Eigen::Ref<Eigen::MatrixXd> prop_states) = 0;

    virtual void motion(const Eigen::Ref<const Eigen::MatrixXd>& cur_states, Eigen::Ref<Eigen::MatrixXd> mot_states) = 0;

    virtual Eigen::MatrixXd getJacobian();

    virtual Eigen::VectorXd getTransitionProbability(const Eigen::Ref<const Eigen::MatrixXd>& prev_states, Eigen::Ref<Eigen::MatrixXd> cur_states);

    virtual Eigen::MatrixXd getNoiseCovarianceMatrix();

    virtual Eigen::MatrixXd getNoiseSample(const std::size_t num);

    virtual bool setProperty(const std::string& property) = 0;

    /**
     * Returns the linear and circular size of the output of the state equation.
     */
    virtual std::pair<std::size_t, std::size_t> getOutputSize() const = 0;
};

#endif /* STATEMODEL_H */
