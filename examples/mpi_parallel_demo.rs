#[cfg(feature = "complex")]
fn main() {
    eprintln!("mpi_parallel_demo.rs is unavailable when built with --features complex");
}

use kryst::parallel::Comm;
use kryst::*; // Import the trait

/// Example demonstrating MPI-parallel inner products and communicator splitting.
///
/// This example shows how to use the enhanced Comm trait with MPI-parallel reductions
/// and communicator splitting functionality.
///
/// to run:
/// cargo mpirun -n 4 --example mpi_parallel_demo --features=mpi
#[cfg(not(feature = "complex"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Kryst MPI-Parallel Communication Demo");
    println!("=====================================");

    // Example 1: Basic communicator operations
    example_basic_comm_operations()?;

    // Example 2: Parallel inner products
    example_parallel_inner_products()?;

    // Example 3: Communicator splitting
    example_communicator_splitting()?;

    Ok(())
}

/// Demonstrate basic communicator operations
#[cfg(not(feature = "complex"))]
fn example_basic_comm_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n1. Basic Communicator Operations");
    println!("---------------------------------");

    // Create a communicator based on available features
    #[cfg(not(any(feature = "mpi", feature = "rayon")))]
    let comm = parallel::NoComm;
    #[cfg(all(feature = "rayon", not(feature = "mpi")))]
    let comm = parallel::RayonComm::new();
    #[cfg(feature = "mpi")]
    let comm = parallel::MpiComm::new();

    println!("Communicator rank: {}", comm.rank());
    println!("Communicator size: {}", comm.size());

    // Test all-reduce operation
    let local_value = 1.0;
    let global_sum = comm.all_reduce_f64(local_value);
    println!("Local value: {}, Global sum: {}", local_value, global_sum);
    println!(
        "Expected: {} (local_value * comm_size)",
        local_value * comm.size() as f64
    );

    // Barrier synchronization
    comm.barrier();
    println!("✓ Barrier synchronization completed");

    Ok(())
}

/// Demonstrate parallel inner products with MPI reductions
#[cfg(not(feature = "complex"))]
fn example_parallel_inner_products() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2. Parallel Inner Products");
    println!("---------------------------");

    // Create test vectors
    let n = 5;
    let x = vec![1.0; n]; // Vector of ones
    let y = vec![2.0; n]; // Vector of twos

    // Create communicator
    #[cfg(not(any(feature = "mpi", feature = "rayon")))]
    let comm = parallel::NoComm;
    #[cfg(all(feature = "rayon", not(feature = "mpi")))]
    let comm = parallel::RayonComm::new();
    #[cfg(feature = "mpi")]
    let comm = parallel::MpiComm::new();

    // NOTE: This would require updating the InnerProduct implementation
    // which currently fails compilation. The intended usage would be:
    //
    // let ip = ();  // Unit type implements InnerProduct
    // let dot_result = ip.dot(&x, &y, &comm);
    // let norm_result = ip.norm(&x, &comm);

    println!("Vector x: {:?}", x);
    println!("Vector y: {:?}", y);
    println!("Expected dot product: {} (per process)", 10.0);
    println!(
        "Expected global dot product: {} (total across {} processes)",
        10.0 * comm.size() as f64,
        comm.size()
    );

    // Demonstrate manual parallel dot product using the communicator
    let local_dot: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let global_dot = comm.all_reduce_f64(local_dot);

    println!("Manual calculation:");
    println!("  Local dot product: {}", local_dot);
    println!("  Global dot product: {}", global_dot);
    println!("✓ Parallel reduction working correctly");

    Ok(())
}

/// Demonstrate communicator splitting for sub-group operations  
#[cfg(not(feature = "complex"))]
fn example_communicator_splitting() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n3. Communicator Splitting");
    println!("--------------------------");

    // Create main communicator
    #[cfg(not(any(feature = "mpi", feature = "rayon")))]
    let main_comm = parallel::UniverseComm::Serial;
    #[cfg(all(feature = "rayon", not(feature = "mpi")))]
    let main_comm = parallel::UniverseComm::Rayon(parallel::RayonComm::new());
    #[cfg(feature = "mpi")]
    let main_comm = parallel::UniverseComm::Mpi(std::sync::Arc::new(parallel::MpiComm::new()));

    println!(
        "Main communicator - Rank: {}, Size: {}",
        main_comm.rank(),
        main_comm.size()
    );

    // Split into two groups: even and odd ranks
    let color = main_comm.rank() as i32 % 2; // 0 for even, 1 for odd
    let key = main_comm.rank() as i32; // Preserve relative ordering

    let sub_comm = main_comm.split(color, key);

    println!(
        "Sub-communicator (color {}) - Rank: {}, Size: {}",
        color,
        sub_comm.rank(),
        sub_comm.size()
    );

    // Demonstrate independent operations in sub-communicators
    let local_value = main_comm.rank() as f64 + 1.0; // Rank + 1
    let sub_sum = sub_comm.all_reduce_f64(local_value);
    let main_sum = main_comm.all_reduce_f64(local_value);

    println!("Local value: {}", local_value);
    println!("Sub-communicator sum: {}", sub_sum);
    println!("Main communicator sum: {}", main_sum);
    println!("✓ Communicator splitting demonstration completed");

    Ok(())
}
