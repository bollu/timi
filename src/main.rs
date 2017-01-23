
use std::io; //for IO

mod frontend;
mod ir;
mod machine;

use frontend::*;
use machine::*;

fn main() {
    use std::io::Write;
    let mut pause_per_step = true;
    let mut m : Machine = Machine::new_minimal();

    loop {
        print!("\n>>>");
        io::stdout().flush().unwrap();

        let input = {
            let mut input : String = String::new();
            match io::stdin().read_line(&mut input) {
                Ok(_) => {}
                Err(error) => panic!("error in read_line: {}", error)
            };
            input.trim().to_string()
        };

        if input == "" {
            continue;
        }
        if input == "exit".to_string() {
            break;
        }
        else if input == "step" {
            pause_per_step = true;
            continue;
        }
        else if input =="nostep" {
            pause_per_step = false;
            continue;

        }
        else if input == "stack" {
            panic!("unimplemented");
            //continue;

        }
        else if input == "dump" {
            panic!("unimplemented");
            //continue;

        }
        else if input == "globals" {
            panic!("unimplemented");
            //continue;

        }
        else if input == "heap" {
            panic!("unimplemented");
            //continue;

        }


        //someone is defining a binding
        if input.starts_with("let ") {
            let sc_defn_str = input.trim_left_matches("let ").to_string();
            let sc_defn = match string_to_sc_defn(sc_defn_str) {
                Result::Ok(defn) => defn,
                Result::Err(e) => {
                    println!("PARSE ERROR: {:#?}", e);
                    continue;
                }
            };

            m.add_supercombinator(sc_defn); 
        }
        else {
            let expr = match string_to_expr(input) {
                Result::Ok(expr) => expr,
                Result::Err(e) => {
                    println!("PARSE ERROR: {:#?}", e);
                    continue;
                }
            };
            m.set_main_expr(&expr);

            let mut i = 0;
            loop {
                match m.step() {
                    Result::Ok(env) => {
                        println!("*** ITERATION: {}", i);
                        print_machine(&m, &env);
                    },
                    Result::Err(e) => {
                        print!("step error: {}\n", e);
                        break;
                    }
                };

                if machine_is_final_state(&m) { break; }

                i += 1;

                if pause_per_step {
                    let mut discard = String::new();
                    let _ = io::stdin().read_line(&mut discard);
                }
            }

            print!("=== MACHINE ENDED ===");
        }
    }
}
