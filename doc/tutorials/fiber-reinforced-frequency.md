# Fiber-reinforced harmonic elasticity

This example couples a two- or three-dimensional matrix to an independently
meshed one-dimensional fiber. Both displacements have `spacedim` components.
The fiber contributes excess stiffness, mass, and damping: its coefficients
are additive to the matrix response already present along the fiber.

The input is
`tutorials/fiber_reinforced_elasticity_freq/parameters.prm.in`:

```{literalinclude} ../../tutorials/fiber_reinforced_elasticity_freq/parameters.prm.in
:language: ini
```

Configure and run the Debug application with:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=mpicxx -DDEAL_II_DIR=/path/to/deal.II \
  -DENABLE_COUPLED_PROBLEMS=OFF -DIMMERSX_BUILD_CORAL_PLUGINS=OFF
cmake --build build-debug --target fiber_reinforced_elasticity_freq_debug -j 4
mpirun -np 2 build-debug/fiber_reinforced_elasticity_freq_debug \
  build-debug/tutorials/fiber_reinforced_elasticity_freq/parameters.prm
```

The parameter template is configured into the build tree. Its output directory
is absolute and lies below the build tree, so the command also works from
another working directory when the executable and input paths are absolute.
The Release target is `fiber_reinforced_elasticity_freq`.

## Equations and phase convention

For a real signal reconstructed as
$u(t)=\operatorname{Re}(\widehat u\exp(i\omega t))$, the harmonic displacement
operator is

$$
H(\omega)=K-\omega^2M+i\omega D,\qquad \omega=2\pi f.
$$

The real and imaginary equations are

$$
\begin{pmatrix}K-\omega^2M&-\omega D\\
\omega D&K-\omega^2M\end{pmatrix}
\begin{pmatrix}u_R\\u_I\end{pmatrix}
=
\begin{pmatrix}f_R\\f_I\end{pmatrix}.
$$

The volume material forms match those in `elasticity_freq`:

$$
\begin{aligned}
K(u,v)&=\int 2\mu\,\varepsilon(u):\varepsilon(v)
                   +\lambda\,\operatorname{div}u\operatorname{div}v,\\
M(u,v)&=\int\rho\,u\cdot v,\\
D(u,v)&=\beta K(u,v)+\alpha M(u,v)
                  +\int\eta\,\varepsilon(u):\varepsilon(v).
\end{aligned}
$$

Gradients on an embedded mesh are the tangential gradients supplied by
deal.II's `FEValuesViews`. The fiber uses the same vector-valued small-strain
form as the existing fiber-reinforced elastodynamics application. It is not a
nonlinear fiber law or a beam model with rotational degrees of freedom.
Choose the excess coefficients, including any cross-sectional scaling,
consistently with the intended one-dimensional model.

Let $C_m$ integrate the matrix trace against the line multiplier, and let
$C_f$ integrate the fiber displacement against that multiplier. The coupled
complex equations are

$$
\begin{pmatrix}
H_m&0&C_m^T\\
0&H_f&-C_f^T\\
C_m&-C_f&0
\end{pmatrix}
\begin{pmatrix}\widehat u_m\\\widehat u_f\\\widehat\lambda\end{pmatrix}
=
\begin{pmatrix}\widehat f_m\\\widehat f_f\\0\end{pmatrix}.
$$

Separate real and imaginary continuity constraints enforce both components of
$C_m\widehat u_m-C_f\widehat u_f=0$. The multiplier is the complex coupling
reaction. The direct solver treats the complete saddle-point system; it does
not assume that either displacement block is positive definite or invertible
on its own. A frequency at an undamped resonance can still make the complete
system singular.

## Parameters and output

`Frequencies` in `Fiber Reinforced Elasticity Frequency` is the common sweep in
Hz. Both Problems share this list. The matrix and fiber have independent
`Grid generation`, `Material properties`, and `Functions` subsections. Loads
and essential data are spatial amplitudes, evaluated at `t=0`; the frequency
solver combines them with their configured phase angles in degrees.

The example clamps the left side of the matrix and applies a harmonic traction
to its right side. The fiber lies inside the matrix, has free endpoints, and
receives its load through the continuity constraint. The damping coefficients
are nonzero. Set both root dimensions to three to select a 3D matrix and an
embedded 3D line; update every vector-valued expression to three components.

By default the multiplier uses continuous `FE_Q` with the fiber degree.
`Multiplier FE degree` can choose another degree, and `Multiplier space/Finite
element` can select an explicit vector element. It must have `dim` components.
The mesh and multiplier choice must yield independent continuity equations;
an unstable or overconstrained choice can produce a singular system.

The driver writes:

- `frequencies.csv`, mapping output indices to frequencies in Hz;
- matrix and fiber VTU/PVTU datasets containing real and imaginary displacement,
  component amplitudes, and component phase angles in degrees;
- the two multiplier components under `interaction/frequency_<index>`.

`residual_norm()` reports the complete algebraic residual, and
`continuity_norm()` reports the combined real and imaginary multiplier-row
residual. These norms do not estimate the discretization error.

The coupled driver uses one prepared mesh per Problem. Set `Number of refinement
cycles = 1`; choose the initial refinements independently before the sweep.
Its adapter path supports strong Dirichlet and Neumann data. Weak Dirichlet data
remain supported by the standalone `elasticity_freq` application. Normal-flux
constraints are available only on full-dimensional 2D and 3D Problems.

## Compose a harmonic Problem

An uncoupled harmonic Problem can also be registered directly with
`LinearAdapter`:

```cpp
using namespace ImmersX;
ElasticityFreqProblemParameters<2> parameters;
initialize_parameters(parameter_file);
ElasticityFreqProblem<2> problem(parameters);
problem.make_grid();
problem.setup_fe();
problem.setup_dofs();
problem.prepare_frequency(1.0);

LinearAdapterParameters solver_parameters;
solver_parameters.solver = LinearSolver::direct;
LinearAdapter<ImmersXLA::MPI::Vector, ImmersXLA::MPI::BlockVector>
  adapter(solver_parameters, problem.communicator());
const auto fields = adapter.add(problem, "solid").fields();
auto state = adapter.make_state();
adapter.solve(state);
problem.accept_frequency_solution(adapter.field(state, fields.real),
                                  adapter.field(state, fields.imaginary));
problem.output_results(0);
```

Include `immersx/physics/elasticity_freq_contributor.h` and
`immersx/core/linear_adapter.h`. The two algebraic Fields can be used in ordinary
FE expressions and continuity constraints. A new Interaction can contribute to
either residual row or introduce its own multiplier Field. Mesh ownership,
material lookup, and essential constraints remain with the Problem; execution
block numbers remain inside the adapter.

Each frequency has constant coefficients during one linear solve. Prepare the
new frequency and create a fresh adapter composition for the next solve so its
cached weak terms use the new coefficients. For a new constitutive term, add
its real and imaginary weak contributions with this same phase convention;
frequency-dependent solver policy belongs in the execution adapter.
