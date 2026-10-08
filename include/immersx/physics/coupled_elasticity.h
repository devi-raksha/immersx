#  ifndef dealii_distributed_lagrange_multiplier_coupled_elasticity_h
#    define dealii_distributed_lagrange_multiplier_coupled_elasticity_h

#    include <immersx/physics/elasticity.h>



namespace ImmersX
{
  template <int dim, int spacedim = dim>
  class CoupledElasticityProblem : public ElasticityProblem<dim, spacedim>
  {
  public:
    CoupledElasticityProblem(
      const ElasticityProblemParameters<dim, spacedim> &par);

    void
    run();
    void
    run_timestep0();
    void
    run_timestep();
    void
    compute_pressure_at_immersed_surface();

    void
    update_vessel_radius_data(std::vector<double> new_data);
    void
    update_vessel_radius_data(std::vector<double> new_data,
                           std::vector<int>    cells_per_vessel);

    std::vector<std::vector<double>>
      split_pressure_over_inclusions(std::vector<int>, Vector<double>) const;

    unsigned int
    n_vessels() const
    {
      return this->inclusions.get_n_vessels();
    };

    TrilinosWrappers::MPI::Vector pressure_at_immersed_surface; ///<
    TrilinosWrappers::MPI::Vector
      pressure_at_immersed_surface_by_inclusion; ///<
    /// @}

  private:
    void
    reassemble_coupling_rhs();

    void
    refine_and_transfer_around_inclusions();

    void
    output_pressure_at_immersed_surface(bool) const;
  };

#  endif

} // namespace ImmersX

