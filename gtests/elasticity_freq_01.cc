// ---------------------------------------------------------------------
// Copyright (C) 2026 by Luca Heltai
// This file is part of ImmersX, based on the deal.II library.
// ---------------------------------------------------------------------
#include <gtest/gtest.h>
#include <immersx/io/utils.h>
#include <immersx/physics/elasticity_freq.h>

#include <cmath>
#include <complex>

#include "test_paths.h"

namespace
{
  template <int dim, int spacedim>
  void
  check_dimension()
  {
    dealii::ParameterAcceptor::clear();
    ImmersX::ElasticityFreqProblemParameters<dim, spacedim> parameters;
    parameters.output_directory =
      ImmersX::TestPaths::output_directory("harmonic-dimensions");
    parameters.initial_refinement = 0;
    parameters.dirichlet_ids.clear();
    ImmersX::initialize_parameters();
    ImmersX::ElasticityFreqProblem<dim, spacedim> problem(parameters);
    problem.make_grid();
    problem.setup_fe();
    problem.setup_dofs();
    EXPECT_EQ(problem.dof_handler().get_triangulation().n_global_active_cells(),
              1u);
    EXPECT_EQ(problem.dof_handler().n_dofs(), (1u << dim) * spacedim);
  }
} // namespace

TEST(ElasticityFreq, BOTH_Dimensions)
{
  check_dimension<1, 1>();
  check_dimension<1, 2>();
  check_dimension<1, 3>();
  check_dimension<2, 2>();
  check_dimension<2, 3>();
  check_dimension<3, 3>();
}

TEST(ElasticityFreq, BOTH_IndependentParameterSubsections)
{
  dealii::ParameterAcceptor::clear();
  ImmersX::ElasticityFreqProblemParameters<2>    matrix("/Harmonic/Matrix/");
  ImmersX::ElasticityFreqProblemParameters<1, 2> fiber("/Harmonic/Fiber/");
  ImmersX::initialize_parameters_from_string(
    "subsection Harmonic\n"
    " subsection Matrix\n"
    "  subsection Material properties\n"
    "   subsection default\n"
    "    set Lame mu = 7\n"
    "   end\n"
    "  end\n"
    " end\n"
    " subsection Fiber\n"
    "  subsection Material properties\n"
    "   subsection default\n"
    "    set Lame mu = 13\n"
    "   end\n"
    "  end\n"
    " end\nend\n");
  EXPECT_DOUBLE_EQ(matrix.default_material_properties.Lame_mu, 7.);
  EXPECT_DOUBLE_EQ(fiber.default_material_properties.Lame_mu, 13.);
}

namespace
{
  template <int dim, int spacedim>
  void
  check_translation()
  {
    dealii::ParameterAcceptor::clear();
    ImmersX::ElasticityFreqProblemParameters<dim, spacedim> parameters;
    parameters.output_directory =
      ImmersX::TestPaths::output_directory("harmonic-translation");
    parameters.output_name =
      "translation_" + std::to_string(dim) + std::to_string(spacedim);
    parameters.initial_refinement = 1;
    parameters.dirichlet_ids.clear();
    parameters.frequencies                                = {0.25};
    parameters.default_material_properties.rho            = 2.;
    parameters.default_material_properties.rayleigh_alpha = 0.4;
    std::string force                                     = "1";
    for (unsigned int d = 1; d < spacedim; ++d)
      force += "; 1";
    ImmersX::initialize_parameters_from_string(
      "subsection Functions\n subsection Right hand side\n set Function expression = " +
      force + "\n end\nend\n");
    ImmersX::ElasticityFreqProblem<dim, spacedim> problem(parameters);
    problem.run();
    const double omega = 0.5 * std::acos(-1.);
    const auto   expected =
      1. / std::complex<double>(-2. * omega * omega, 0.8 * omega);
    for (const auto i : problem.locally_owned_dofs())
      {
        EXPECT_NEAR(problem.real_displacement()[i], expected.real(), 1.e-8);
        EXPECT_NEAR(problem.imaginary_displacement()[i],
                    expected.imag(),
                    1.e-8);
      }
  }
} // namespace

TEST(ElasticityFreq, BOTH_DampedRigidTranslation)
{
  check_translation<1, 1>();
  check_translation<1, 2>();
  check_translation<1, 3>();
  check_translation<2, 2>();
  check_translation<2, 3>();
  check_translation<3, 3>();
}
