# Fiber-reinforced harmonic elasticity

The canonical input is `parameters.prm.in`. CMake writes the runnable file to
`<build>/tutorials/fiber_reinforced_elasticity_freq/parameters.prm`.

```bash
cmake --build build-debug --target fiber_reinforced_elasticity_freq_debug -j 4
mpirun -np 2 build-debug/fiber_reinforced_elasticity_freq_debug \
  build-debug/tutorials/fiber_reinforced_elasticity_freq/parameters.prm
```

The example solves a matrix with an embedded excess fiber at 0, 0.1, and 0.2 Hz.
It writes complex displacements, component amplitudes and phases, multiplier
reactions, and a frequency-index CSV below the configured build output tree.
See `doc/tutorials/fiber-reinforced-frequency.md` for the equations, parameters,
extension API, and solver limitations.
