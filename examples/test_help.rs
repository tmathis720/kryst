use kryst::PcOptions;

fn main() {
    let args = vec!["-help"];

    match PcOptions::from_args(&args) {
        Ok(_) => println!("Unexpected success"),
        Err(_) => println!("Help should be displayed via help function"),
    }

    // Check if help is requested
    if kryst::config::options_core::is_help_requested(&args) {
        kryst::print_help();
    }
}
