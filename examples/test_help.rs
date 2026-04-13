#[cfg(feature = "complex")]
fn main() {
    eprintln!("test_help.rs is unavailable when built with --features complex");
}

// To run:
// ```sh
// cargo run --example test_help
// ```

use kryst::config::options::PcOptions;

#[cfg(not(feature = "complex"))]
fn main() {
    let args = vec!["-help"];

    match PcOptions::from_args(&args) {
        Ok(_) => println!("Unexpected success"),
        Err(_) => println!("Help should be displayed via help function"),
    }

    // Check if help is requested
    if kryst::config::options_core::is_help_requested(&args) {
        kryst::config::print_help();
    }
}
