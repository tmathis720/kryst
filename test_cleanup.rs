// Test file to verify the cleaned up API works correctly
use kryst::context::{KspContext, PcType, PcFactory};
use kryst::context::ksp_context::SolverType;
use faer::Mat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test that all the moved types are accessible and work correctly
    let mut ksp = KspContext::new();
    
    // Test solver type setting (should still work)
    ksp.set_type(SolverType::Cg)?;
    
    // Test preconditioner type setting (now uses pc_context)
    ksp.set_pc_type(PcType::Jacobi)?;
    
    // Test PC factory functionality
    let pc = PcFactory::create_preconditioner(PcType::Ilu0, None)?;
    println!("Created preconditioner successfully");
    
    // Test deferred PC creation
    let deferred = PcFactory::create_deferred_pc(PcType::Chebyshev, None)?;
    println!("Created deferred PC info successfully");
    
    // Create a test matrix for deferred PC construction
    let matrix = Mat::<f64>::identity(3, 3);
    let constructed_pc = PcFactory::construct_deferred_preconditioner(deferred, &matrix)?;
    println!("Constructed deferred preconditioner successfully");
    
    println!("All API tests passed - cleanup was successful!");
    
    Ok(())
}
