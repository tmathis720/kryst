# Kryst: PETSc-Style Options Database & Explicit Setup Implementation

## Summary

We have successfully implemented a complete PETSc-style options database and explicit setup phase for the Kryst linear solver library. This implementation provides both runtime configuration via command-line options and efficient workspace reuse for repeated solves.

## ✅ Completed Features

### 1. PETSc-Style Options Database

#### ✅ Options Structs Created
- **`KspOptions`** - Krylov solver configuration options
- **`PcOptions`** - Preconditioner configuration options  
- **`PcSide`** - Preconditioning side enum (Left/Right/Symmetric)

#### ✅ Supported Command-Line Options

**KSP (Krylov Solver) Options:**
- `-ksp_type <solver>` - Solver type: cg, pcg, gmres, bicgstab, cgs, qmr, tfqmr, minres, cgnr, preonly
- `-ksp_rtol <float>` - Relative tolerance (default: 1e-6)
- `-ksp_atol <float>` - Absolute tolerance (default: 1e-12)
- `-ksp_dtol <float>` - Divergence tolerance (default: 1e3)
- `-ksp_max_it <int>` - Maximum iterations (default: 1000)
- `-ksp_gmres_restart <int>` - GMRES restart parameter (default: 50)
- `-ksp_pc_side <side>` - Preconditioning side: left, right, symmetric

**PC (Preconditioner) Options:**
- `-pc_type <pc>` - Preconditioner type: jacobi, ilu0, none
- `-pc_ilu_levels <int>` - ILU fill levels (default: 0)
- `-pc_chebyshev_degree <int>` - Chebyshev polynomial degree (default: 3)
- `-pc_ilut_drop_tol <float>` - ILUT drop tolerance (default: 1e-3)
- `-pc_ilut_max_fill <int>` - ILUT maximum fill per row (default: 10)

#### ✅ Argument Parsing Implementation
- Manual argument parsing with proper error handling
- Support for command-line and environment variable precedence
- Help system with `-help`, `--help`, `-h` flags
- Robust error messages for invalid options

#### ✅ Integration with KspContext
- `set_from_options()` method to configure from parsed options
- `set_from_all_options()` for combined KSP/PC configuration
- Seamless integration with existing builder pattern

### 2. Explicit Setup Phase & Workspace Reuse

#### ✅ Workspace Structure
- **`Workspace`** struct containing all Krylov solver buffers:
  - Krylov basis vectors (`q`)
  - Preconditioned basis vectors (`z`)
  - Hessenberg matrix (`h`)
  - Givens rotation arrays (`cs`, `sn`)
  - Residual vector (`g`)
  - Temporary vectors (`tmp1`, `tmp2`)

#### ✅ Two-Phase API
- **Setup Phase**: `setup()` - Prepare preconditioner and allocate workspaces
- **Solve Phase**: `solve()` - Efficiently solve using cached data

#### ✅ Workspace Management
- Automatic workspace allocation and sizing
- Compatibility checking for problem dimensions
- Invalidation when solver parameters change
- Manual invalidation via `invalidate_setup()`

#### ✅ KspContext Enhancements
- `work: Option<Workspace>` field for cached workspace
- `setup_called: bool` flag for state tracking
- `is_setup()` method to check readiness
- Auto-setup on first solve if not explicitly called

### 3. Error Handling & Validation

#### ✅ Extended Error Types
- `UnrecognizedSolverType` for invalid solver names
- `UnrecognizedPcType` for invalid preconditioner names
- Comprehensive error messages with context

#### ✅ Robust Validation
- Option value parsing with clear error messages
- Workspace compatibility checking
- Solver configuration validation

### 4. Testing & Examples

#### ✅ Comprehensive Test Suite
**Options Tests (`options_integration.rs`):**
- ✅ Parsing KSP and PC options from arguments
- ✅ Mixed option parsing (KSP + PC in same command)
- ✅ Invalid option handling
- ✅ Missing value detection
- ✅ Numeric value validation
- ✅ Enum parsing (SolverType, PcType, PcSide)
- ✅ Empty argument handling
- ✅ Non-option argument filtering

**Workspace Tests (`workspace_tests.rs`):**
- ✅ Explicit setup functionality
- ✅ Auto-setup on first solve
- ✅ Workspace reuse across multiple solves
- ✅ Restart parameter invalidation
- ✅ Manual setup invalidation
- ✅ Size compatibility validation
- ✅ Error handling for unconfigured solvers

#### ✅ Example Programs
**Options Demo (`options_demo.rs`):**
- Demonstrates command-line option parsing
- Shows KSP context configuration from options
- Solves a simple test system
- Provides usage examples

**Setup & Reuse Demo (`setup_reuse_demo.rs`):**
- Compares auto-setup vs explicit setup performance
- Demonstrates workspace reuse efficiency
- Shows workspace management features
- Illustrates best practices

### 5. Documentation

#### ✅ Updated README
- Complete options reference table
- Usage examples for all features
- Command-line option examples
- API usage patterns

#### ✅ Comprehensive Code Documentation
- Detailed docstrings for all public methods
- Examples in documentation
- Clear parameter descriptions
- Usage patterns and best practices

## Usage Examples

### Basic PETSc-Style Usage

```rust
use kryst::context::ksp_context::KspContext;
use kryst::config::options::parse_all_options;

// Parse command-line options
let args: Vec<String> = std::env::args().collect();
let (ksp_opts, pc_opts) = parse_all_options(&args)?;

// Configure and solve
let mut ksp = KspContext::new();
ksp.set_from_all_options(&ksp_opts, &pc_opts)?;
let stats = ksp.solve(&A, &b, &mut x)?;
```

### Explicit Setup for Efficiency

```rust
let mut ksp = KspContext::new();
ksp.set_type(SolverType::Gmres)?
   .set_pc_type(PcType::Jacobi)?;

// Setup once
ksp.setup(&A, n)?;

// Solve multiple times efficiently
for b in right_hand_sides {
    let stats = ksp.solve(&A, &b, &mut x)?;
}
```

### Command-Line Usage

```bash
# Configure CG with Jacobi preconditioning
./my_program -ksp_type cg -ksp_rtol 1e-8 -pc_type jacobi

# Configure GMRES with custom restart
./my_program -ksp_type gmres -ksp_gmres_restart 100 -pc_type ilu0

# Show help
./my_program -help
```

## Performance Benefits

1. **Setup Reuse**: Preconditioner factorization done once, reused for multiple solves
2. **Workspace Reuse**: All Krylov vectors allocated once, eliminating repeated allocations
3. **Memory Efficiency**: Pre-allocated buffers reduce GC pressure
4. **Cache Locality**: Reused workspace improves cache performance

## Architecture Highlights

1. **Type Safety**: Enum-based configuration prevents runtime errors
2. **Builder Pattern**: Fluent API for configuration
3. **Automatic Management**: Auto-setup with manual override capability
4. **Compatibility Checking**: Runtime validation of workspace dimensions
5. **Error Propagation**: Clear error messages with context

## Status: ✅ Complete

All checklist items from the original recipe have been implemented and tested:

- ✅ **Module scaffolding**: Options structs and KSP context integration
- ✅ **Enums defined**: SolverType, PcType, PcSide with string parsing
- ✅ **Workspace struct**: Complete with all Krylov solver buffers
- ✅ **Setup methods**: Both explicit and automatic setup functionality
- ✅ **Workspace reuse**: Efficient buffer management and compatibility checking
- ✅ **Options integration**: set_from_options() methods with precedence
- ✅ **Error handling**: Comprehensive error types and validation
- ✅ **Testing**: Unit and integration tests for all functionality
- ✅ **Examples**: Demonstration programs showing best practices
- ✅ **Documentation**: Complete API docs and usage guide

The implementation provides a production-ready PETSc-style interface with efficient workspace management, making Kryst suitable for high-performance scientific computing applications requiring repeated solves.
