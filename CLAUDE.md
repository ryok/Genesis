# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Genesis Physics Platform

Genesis is a universal physics engine and robotics simulation platform that integrates multiple physics solvers (Rigid body, MPM, SPH, FEM, PBD, Stable Fluid) into a unified framework. It's designed for general-purpose Robotics/Embodied AI/Physical AI applications with photo-realistic rendering capabilities.

## Installation and Setup

### Standard Installation
```bash
pip install genesis-world  # Requires Python>=3.10,<3.13
```

### Development Installation  
```bash
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install -e ".[dev]"
```

### Docker Installation
```bash
docker build -t genesis -f docker/Dockerfile docker
```

## Core Architecture

### Main Components Structure
- **genesis/engine/**: Core physics engine containing solvers, entities, and scene management
  - **simulator.py**: Scene-level simulation manager coordinating multiple solvers
  - **scene.py**: Wraps all simulation components (simulator, entities, visualizer)
  - **entities/**: Different physics entity types (RigidEntity, MPMEntity, PBDEntity, etc.)
  - **solvers/**: Physics solvers (RigidSolver, MPMSolver, SPHSolver, FEMSolver, etc.)
  - **materials/**: Material definitions for different physics types
  - **states/**: State management and caching system
- **genesis/vis/**: Visualization and rendering system
- **genesis/utils/**: Utility functions for geometry, mesh processing, file I/O
- **genesis/options/**: Configuration classes for various simulation components
- **genesis/ext/**: External dependencies and extensions

### Key Physics Solvers
- **Rigid Body**: For solid objects and robotic systems
- **MPM (Material Point Method)**: For granular materials, fluids, and elastoplastic materials  
- **SPH (Smoothed Particle Hydrodynamics)**: For liquid simulation
- **FEM (Finite Element Method)**: For deformable solid simulation
- **PBD (Position Based Dynamics)**: For cloth, soft bodies, and particle systems
- **Stable Fluid**: For smoke and gas simulation

### Entity Types
Each physics solver has corresponding entity types that can be added to scenes:
- RigidEntity, MPMEntity, SPHEntity, PBDEntity, SFParticleEntity, etc.
- HybridEntity for coupling multiple physics types
- ToolEntity for non-simulated objects
- AvatarEntity for character animation

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not benchmarks"  # Skip benchmark tests
pytest -m "required"        # Run only required tests
pytest tests/test_rigid_physics.py  # Run specific test file

# Run benchmarks separately  
pytest -m benchmarks
```

### Code Quality
```bash
# Format code with Black
black genesis/ examples/ tests/

# Run type checking (if mypy is configured)
mypy genesis/
```

### Building Extensions
The project includes Cython extensions that are built automatically during installation:
```bash
python setup.py build_ext --inplace  # Build extensions in place
```

## Common Usage Patterns

### Basic Scene Setup
```python
import genesis as gs

# Initialize Genesis
gs.init(backend='gpu')  # or 'cpu', 'vulkan', 'metal'

# Create scene
scene = gs.Scene()

# Add entities
rigid_entity = scene.add_entity(gs.RigidEntity(...))
mpm_entity = scene.add_entity(gs.MPMEntity(...))

# Build and run simulation
scene.build()
for i in range(1000):
    scene.step()
```

### Loading Assets
Genesis supports multiple file formats:
- URDF files for robotic systems: `assets/urdf/`
- MJCF (.xml) files: `assets/xml/`  
- Mesh files (.obj, .glb, .ply, .stl): `assets/meshes/`

### Example Scripts
The `examples/` directory contains 80+ example scripts organized by category:
- `rigid/`: Rigid body examples (robotic arms, manipulation, etc.)
- `coupling/`: Multi-physics coupling examples
- `locomotion/`: Legged robot locomotion
- `manipulation/`: Robotic manipulation tasks
- `tutorials/`: Step-by-step learning examples

## Testing Strategy

Tests are organized in the `tests/` directory:
- `test_rigid_physics.py`: Rigid body simulation tests
- `test_deformable_physics.py`: Soft body and deformable material tests  
- `test_render.py`: Rendering system tests
- `test_utils.py`: Utility function tests
- `run_benchmarks.py`: Performance benchmarking

Use pytest markers:
- `@pytest.mark.benchmarks`: Performance tests (run separately)
- `@pytest.mark.required`: Critical tests that must pass

## Architecture Notes

### Multi-Physics Coupling
Genesis uses a coupler system to handle interactions between different physics solvers:
- LegacyCoupler: Basic coupling implementation
- SAPCoupler: Sweep and Prune based spatial partitioning for efficient collision detection

### Taichi Integration
Genesis is built on Taichi for high-performance cross-platform computation:
- GPU acceleration on CUDA, Vulkan, Metal
- CPU fallback support
- JIT compilation for optimal performance

### Rendering Backends
- **Rasterizer**: Fast OpenGL-based rendering via PyRender
- **Raytracer**: Photo-realistic rendering via LuisaRender (optional)

### State Management
The simulation state system provides:
- Efficient state caching and retrieval
- Automatic memory management
- Support for batched simulations

## File Organization Notes

- Configuration in `pyproject.toml` (modern Python packaging)
- Legacy `setup.py` only for Cython extension building
- Assets organized by type in `genesis/assets/`
- External dependencies in `genesis/ext/`
- Black code formatting with 120 character line length
- Pytest configuration with parallel test execution