#[macro_use]
extern crate prettytable;

use std::io; //for IO

mod frontend;
mod ir;
mod machine;

use frontend::*;
use machine::*;


fn run_step_interaction(iter_count: i32, m: &Machine) {
    use std::io::Write;
    loop {
        print!("{}>> ", iter_count);
        io::stdout().flush().unwrap();

        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input).unwrap();

        input = input.trim().to_string();
        if input == "" {
            continue;
        }
        else if input == ":help" {
            println!("*** HELP ***\n\
                     :help - bring up help\n\
                     :stack - show current stack
                     ")
       }
        else if input == ":stack" {
            print_machine_stack(m);
        }
        else if input == "s" || input == "step" {
            return;
        }
        else {
            println!("unrecognized: |{}|, type :help for help.", input)

        }

    }
}
fn run_machine(m: &mut Machine, pause_per_step: bool) {

    let mut i = 0;
    loop {
        match m.step() {
            Result::Ok(()) => {
                println!("*** ITERATION: {}", i);
                i += 1;
                print_machine(&m);
            },
            Result::Err(e) => {
                print!("step error: {}\n", e);
                break;
            }
        };

        if machine_is_final_state(&m) {
            println!("=== FINAL: {:#?} ===", machine_get_final_val(&m));
            break;
        }


        if pause_per_step {
            run_step_interaction(i, m);
        }
    }
}

fn main() {
    use std::io::Write;
    let mut pause_per_step = false;
    let mut m : Machine = Machine::new_minimal();

    loop {
        print!(">");
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
            run_machine(&mut m, pause_per_step);
        }
    }
}
