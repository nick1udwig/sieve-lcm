#![forbid(unsafe_code)]

use sieve_lcm::cli::{
    execute_command, parse_command, serialize_error_json, serialize_success_json,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let result = parse_command(&args).and_then(execute_command);
    match result {
        Ok(success) => match serialize_success_json(&success) {
            Ok(encoded) => {
                println!("{encoded}");
                std::process::exit(0);
            }
            Err(err) => {
                eprintln!("{}", serialize_error_json(&err));
                std::process::exit(1);
            }
        },
        Err(err) => {
            eprintln!("{}", serialize_error_json(&err));
            std::process::exit(1);
        }
    }
}
