// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of ImmersX, based on the deal.II library.
// ---------------------------------------------------------------------
#ifndef immersx_fiber_reinforced_elasticity_freq_h
#define immersx_fiber_reinforced_elasticity_freq_h

#include <deal.II/fe/fe_tools.h>

#include <immersx/core/constraint.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/output_handler.h>
#include <immersx/physics/elasticity_freq_contributor.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

namespace ImmersX
{
  /** Matrix-plus-excess-fiber parameters for a harmonic frequency sweep. */
  template <int dim>
  class FiberReinforcedElasticityFreqParameters
    : public dealii::ParameterAcceptor
  {
  public:
    std::vector<double> frequencies{1.};

    explicit FiberReinforcedElasticityFreqParameters(
      const std::string &subsection = "/Fiber Reinforced Elasticity Frequency/")
      : ParameterAcceptor(MaterialProperties::normalize_subsection(subsection))
      , matrix_parameters(MaterialProperties::normalize_subsection(subsection) +
                            "Matrix/",
                          &frequencies)
      , fiber_parameters(MaterialProperties::normalize_subsection(subsection) +
                           "Fiber/",
                         &frequencies)
      , multiplier_parameters(MaterialProperties::normalize_subsection(
                                subsection) +
                                "Multiplier space/",
                              "auto")
      , solver_parameters(MaterialProperties::normalize_subsection(subsection) +
                          "Solver/")
    {
      add_parameter("Frequencies", frequencies);
      add_parameter("Output directory", output_directory);
      add_parameter("Multiplier FE degree",
                    multiplier_degree,
                    "Zero follows the fiber degree.",
                    prm,
                    dealii::Patterns::Integer(0));
      solver_parameters.solver = LinearSolver::direct;
      parse_parameters_call_back.connect([this]() {
        matrix_parameters.output_directory = output_directory + "/matrix";
        fiber_parameters.output_directory  = output_directory + "/fiber";
        matrix_parameters.output_name      = "matrix";
        fiber_parameters.output_name       = "fiber";
        AssertThrow(!frequencies.empty(),
                    dealii::ExcMessage("Provide at least one frequency."));
        for (const auto frequency : frequencies)
          AssertThrow(std::isfinite(frequency) && frequency >= 0.,
                      dealii::ExcMessage(
                        "Frequencies must be finite and nonnegative."));
      });
    }

    ElasticityFreqProblemParameters<dim, dim> matrix_parameters;
    ElasticityFreqProblemParameters<1, dim>   fiber_parameters;
    FiniteElementSpaceParameters<1, dim>      multiplier_parameters;
    LinearAdapterParameters                   solver_parameters;
    std::string  output_directory = "./output/fiber_reinforced_elasticity_freq";
    unsigned int multiplier_degree = 0;
  };

  /** Coupled harmonic response of a bulk matrix and an independently meshed
   * additive/excess fiber. Separate real and imaginary displacement-continuity
   * constraints carry the complex reaction. No positivity assumption is made
   * about the frequency-dependent displacement blocks.
   */
  template <int dim>
  class FiberReinforcedElasticityFreq
  {
    static_assert(dim == 2 || dim == 3,
                  "The matrix must be two- or three-dimensional.");

  public:
    using Parameters       = FiberReinforcedElasticityFreqParameters<dim>;
    using MatrixProblem    = ElasticityFreqProblem<dim, dim>;
    using FiberProblem     = ElasticityFreqProblem<1, dim>;
    using VectorType       = typename MatrixProblem::VectorType;
    using GlobalVectorType = ImmersXLA::MPI::BlockVector;
    using Adapter          = LinearAdapter<VectorType, GlobalVectorType>;

    explicit FiberReinforcedElasticityFreq(const Parameters &parameters)
      : parameters(parameters)
      , matrix_storage(parameters.matrix_parameters)
      , fiber_storage(parameters.fiber_parameters)
    {}

    void
    setup()
    {
      AssertThrow(!multiplier_space,
                  dealii::ExcMessage(
                    "The harmonic driver is already initialized."));
      AssertThrow(parameters.matrix_parameters.n_refinement_cycles == 1 &&
                    parameters.fiber_parameters.n_refinement_cycles == 1,
                  dealii::ExcMessage(
                    "The harmonic fiber driver uses one mesh per Problem; "
                    "set Number of refinement cycles to one."));
      matrix_storage.make_grid();
      matrix_storage.setup_fe();
      matrix_storage.setup_dofs();
      fiber_storage.make_grid();
      fiber_storage.setup_fe();
      fiber_storage.setup_dofs();
      const unsigned int degree =
        parameters.multiplier_degree == 0 ?
          fiber_storage.dof_handler().get_fe().degree :
          parameters.multiplier_degree;
      multiplier_space = std::make_unique<FiniteElementSpace<1, dim>>(
        fiber_storage.triangulation(),
        parameters.multiplier_parameters.finite_element == "auto" ?
          std::make_unique<dealii::FESystem<1, dim>>(
            dealii::FE_Q<1, dim>(degree), dim) :
          dealii::FETools::get_fe_by_name<1, dim>(
            parameters.multiplier_parameters.finite_element));
      AssertThrow(multiplier_space->finite_element().n_components() == dim,
                  dealii::ExcMessage(
                    "The multiplier FE must have dim components."));
      auto &constraints = multiplier_space->constraints();
      constraints.reinit(multiplier_space->locally_owned_dofs(),
                         multiplier_space->locally_relevant_dofs());
      dealii::DoFTools::make_hanging_node_constraints(
        multiplier_space->dof_handler(), constraints);
      constraints.close();
      multiplier_view = std::make_unique<FiniteElementSpaceView<1, dim>>(
        multiplier_space->view());
    }

    void
    solve_frequency(const double frequency, const unsigned int index = 0)
    {
      AssertThrow(multiplier_space != nullptr, dealii::ExcNotInitialized());
      matrix_storage.prepare_frequency(frequency);
      fiber_storage.prepare_frequency(frequency);
      Adapter    adapter(parameters.solver_parameters,
                         matrix_storage.communicator());
      const auto matrix = adapter.add(matrix_storage, "matrix").fields();
      const auto fiber  = adapter.add(fiber_storage, "fiber").fields();
      const auto lambda_real =
        adapter
          .add(make_continuity_constraint(
                 matrix.real,
                 fiber.real,
                 multiplier_view->field("lambda_real",
                                        dealii::FEValuesExtractors::Vector(0))),
               "real-continuity")
          .fields()
          .multiplier;
      const auto lambda_imaginary =
        adapter
          .add(make_continuity_constraint(
                 matrix.imaginary,
                 fiber.imaginary,
                 multiplier_view->field("lambda_imaginary",
                                        dealii::FEValuesExtractors::Vector(0))),
               "imaginary-continuity")
          .fields()
          .multiplier;
      auto state = adapter.make_state();
      adapter.solve(state);
      matrix_storage.accept_frequency_solution(
        adapter.field(state, matrix.real),
        adapter.field(state, matrix.imaginary));
      fiber_storage.accept_frequency_solution(adapter.field(state, fiber.real),
                                              adapter.field(state,
                                                            fiber.imaginary));
      real_multiplier_storage      = adapter.field(state, lambda_real);
      imaginary_multiplier_storage = adapter.field(state, lambda_imaginary);
      auto residual                = adapter.make_state();
      adapter.evaluate_residual(state, residual);
      residual_norm_storage = residual.l2_norm();
      continuity_norm_storage =
        std::hypot(adapter.field(residual, lambda_real).l2_norm(),
                   adapter.field(residual, lambda_imaginary).l2_norm());
      matrix_storage.output_results(index);
      fiber_storage.output_results(index);
      const auto directory =
        std::filesystem::path(parameters.output_directory) / "interaction" /
        ("frequency_" + std::to_string(index));
      OutputHandler<1, dim, VectorType> real_output(lambda_real,
                                                    directory,
                                                    "lambda_real");
      OutputHandler<1, dim, VectorType> imaginary_output(lambda_imaginary,
                                                         directory,
                                                         "lambda_imaginary");
      adapter.write_output(real_output, state, frequency);
      adapter.write_output(imaginary_output, state, frequency);
    }

    void
    run()
    {
      setup();
      std::filesystem::create_directories(parameters.output_directory);
      unsigned int metadata_written = 1;
      if (dealii::Utilities::MPI::this_mpi_process(
            matrix_storage.communicator()) == 0)
        {
          std::ofstream metadata(parameters.output_directory +
                                 "/frequencies.csv");
          metadata << "index,frequency_hz\n";
          for (unsigned int i = 0; i < parameters.frequencies.size(); ++i)
            metadata << i << ',' << std::setprecision(17)
                     << parameters.frequencies[i] << '\n';
          metadata.close();
          metadata_written = metadata.good();
        }
      AssertThrow(
        dealii::Utilities::MPI::min(metadata_written,
                                    matrix_storage.communicator()) == 1,
        dealii::ExcMessage("Could not write harmonic frequency metadata."));
      for (unsigned int i = 0; i < parameters.frequencies.size(); ++i)
        solve_frequency(parameters.frequencies[i], i);
    }

    const MatrixProblem &
    matrix_problem() const
    {
      return matrix_storage;
    }
    const FiberProblem &
    fiber_problem() const
    {
      return fiber_storage;
    }
    const VectorType &
    real_multiplier() const
    {
      return real_multiplier_storage;
    }
    const VectorType &
    imaginary_multiplier() const
    {
      return imaginary_multiplier_storage;
    }
    double
    residual_norm() const
    {
      return residual_norm_storage;
    }
    double
    continuity_norm() const
    {
      return continuity_norm_storage;
    }

  private:
    const Parameters                               &parameters;
    MatrixProblem                                   matrix_storage;
    FiberProblem                                    fiber_storage;
    std::unique_ptr<FiniteElementSpace<1, dim>>     multiplier_space;
    std::unique_ptr<FiniteElementSpaceView<1, dim>> multiplier_view;
    VectorType real_multiplier_storage, imaginary_multiplier_storage;
    double     residual_norm_storage   = 0.;
    double     continuity_norm_storage = 0.;
  };
} // namespace ImmersX
#endif
