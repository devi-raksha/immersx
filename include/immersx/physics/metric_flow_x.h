// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#ifndef immersx_metric_flow_x_h
#define immersx_metric_flow_x_h

#include <immersx/config.h>

#ifdef IMMERSX_WITH_METRIC_FLOW_X

#  include <immersx/algebra/local_preconditioner.h>
#  include <immersx/core/contributor.h>
#  include <immersx/core/fe_space.h>
#  include <metric_flow_x/blood_flow_system.h>

#  include <memory>
#  include <numeric>
#  include <type_traits>
#  include <utility>

namespace ImmersX
{
  namespace detail
  {
    template <typename Problem, typename = void>
    struct metric_flow_x_has_system_dof_index : std::false_type
    {};

    template <typename Problem>
    struct metric_flow_x_has_system_dof_index<
      Problem,
      std::void_t<decltype(std::declval<const Problem &>().system_dof_index(
        dealii::types::global_dof_index{}))>> : std::true_type
    {};
  } // namespace detail

  /** Non-owning description of a MetricFlowX Problem contribution.
   *
   * BloodFlowSystem owns the mesh, physics, and native vectors.  This value
   * only supplies the mutable access needed by native assembly while allowing
   * the public adapter::add() API to retain its logically-const contributor
   * contract.
   */
  template <int dim, int spacedim = dim>
  struct MetricFlowXProblem
  {
    using Problem = ::MetricFlowX::BloodFlowSystem<dim, spacedim>;

    Problem *problem = nullptr;
  };

  /** Semantic fields and native component views for one BloodFlowSystem. */
  struct MetricFlowXFields
  {
    using Space       = FiniteElementSpaceView<1, 3>;
    using ScalarField = Field<1, 3, dealii::FEValuesExtractors::Scalar>;

    FieldId                      state;
    std::shared_ptr<const Space> space;
    ScalarField                  area;
    ScalarField                  velocity;
  };

  /** Make a non-owning MetricFlowX contribution descriptor. */
  template <int dim, int spacedim>
  MetricFlowXProblem<dim, spacedim>
  metric_flow_x(::MetricFlowX::BloodFlowSystem<dim, spacedim> &problem)
  {
    return {&problem};
  }

  /** Register BloodFlowSystem as one mixed differential/algebraic Field. */
  template <typename Builder, int dim, int spacedim>
  MetricFlowXFields
  contribute(Builder                                 &builder,
             const MetricFlowXProblem<dim, spacedim> &description)
  {
    using Problem = typename MetricFlowXProblem<dim, spacedim>::Problem;
    using Vector  = ::MetricFlowX::VectorType;
    using Matrix  = ::MetricFlowX::MatrixType;

    static_assert(
      std::is_same_v<Vector, ImmersXLA::MPI::Vector>,
      "MetricFlowX and ImmersX must use the same linear algebra backend.");

    AssertThrow(description.problem != nullptr,
                dealii::ExcMessage("A MetricFlowX Problem cannot be null."));
    Problem &problem = *description.problem;

    const FieldId state = builder.field("state",
                                        problem.locally_owned_dofs(),
                                        problem.locally_relevant_dofs(),
                                        problem.differential_dofs());

    // Coupled Schur preconditioning needs an inverse of the mixed flow block.
    builder.preconditioner(
      state, [](const auto &matrix, const auto &prototype) {
        using Preconditioner = ::MetricFlowX::LA::MPI::PreconditionILU;
        typename Preconditioner::AdditionalData data;
#  ifdef METRIC_FLOW_X_WITH_TRILINOS
        // Algebraic face traces have zero diagonal entries in the native
        // matrix.
        data.ilu_atol = 1.e-5;
#  endif
        return make_local_preconditioner<Vector, Matrix, Preconditioner>(
          matrix, prototype, data);
      });

    auto term = builder.term(state, "blood-flow");
    term.residual([description, state](const auto &context) {
      const auto *y    = &context.state(state);
      const auto *ydot = &context.derivative(state);
      const auto  time = context.time();

      dealii::PackagedOperation<Vector> result;
      result.reinit_vector = [description](Vector &vector, const bool) {
        description.problem->reinit_state(vector);
      };
      result.apply = [description, y, ydot, time](Vector &destination) {
        description.problem->assemble_residual(time, *y, *ydot, destination);
      };
      result.apply_add = [description, y, ydot, time](Vector &destination) {
        Vector contribution;
        description.problem->reinit_state(contribution);
        description.problem->assemble_residual(time, *y, *ydot, contribution);
        destination += contribution;
      };
      return result;
    });

    typename Builder::Model::MatrixOperatorFactory state_factory =
      [description, state](const auto &context) {
        AssertThrow(context.has_state_derivative(),
                    dealii::ExcMessage(
                      "MetricFlowX Jacobians require a state derivative."));
        description.problem->assemble_state_jacobian(context.time(),
                                                     context.state(state),
                                                     context.derivative(state));
        return matrix_operator<Vector, Matrix>(
          description.problem->state_jacobian_matrix());
      };
    term.state(state, std::move(state_factory));

    typename Builder::Model::MatrixOperatorFactory derivative_factory =
      [description, state](const auto &context) {
        AssertThrow(context.has_state_derivative(),
                    dealii::ExcMessage(
                      "MetricFlowX Jacobians require a state derivative."));
        description.problem->assemble_derivative_jacobian(
          context.time(), context.state(state), context.derivative(state));
        return matrix_operator<Vector, Matrix>(
          description.problem->derivative_jacobian_matrix());
      };
    term.derivative(state, std::move(derivative_factory));

    auto space = std::make_shared<MetricFlowXFields::Space>(
      problem.dof_handler(),
      dealii::StaticMappingQ1<1, 3>::mapping,
      problem.constraints());
    auto execution_constraints =
      std::make_shared<dealii::AffineConstraints<double>>(
        problem.constraints());
    std::vector<dealii::types::global_dof_index> fe_to_state(
      problem.dof_handler().n_dofs());
    std::iota(fe_to_state.begin(), fe_to_state.end(), 0);
    // Rank-local RCR pressures interrupt the FE numbering between ranks.
    // Older MetricFlowX packages kept FE indices unchanged in the full state.
    if constexpr (detail::metric_flow_x_has_system_dof_index<Problem>::value)
      for (auto &index : fe_to_state)
        index = problem.system_dof_index(index);
    const auto make_state_field = [&](const auto        &extractor,
                                      const std::string &name) {
      return space->field(state, name, extractor)
        .reindexed(name,
                   problem.locally_owned_dofs(),
                   problem.locally_relevant_dofs(),
                   fe_to_state,
                   execution_constraints);
    };
    return {state,
            space,
            make_state_field(problem.area_extractor(), "area"),
            make_state_field(problem.velocity_extractor(), "velocity")};
  }
} // namespace ImmersX

#endif // IMMERSX_WITH_METRIC_FLOW_X

#endif // immersx_metric_flow_x_h
