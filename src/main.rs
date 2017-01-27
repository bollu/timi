#[macro_use]
extern crate prettytable;
extern crate rustyline;

use rustyline::error::ReadlineError;
use rustyline::Editor;
use rustyline::completion::Completer;

mod frontend;
mod ir;
mod machine;

use frontend::*;
use machine::*;


fn run_step_interaction<C>(rl: &mut Editor<C>,
                           iter_count: i32,
                           m: &Machine,
                           pause_per_step: &mut bool) ->
    Result<(), ReadlineError>
    where C: Completer {
    loop {
        let input = {
            let prompt = format!("{}>>", iter_count);
            let raw_input : String = try!(rl.readline(&prompt));
            raw_input.trim().to_string()
        };
        rl.add_history_entry(&input);

        if input == "" {
            continue;
        }
        else if input == ":help" {
            println!("*** HELP ***");
            println!(":help - bring up help");
            println!(":stack - show current state");
            println!(":nostep - disable stepping and run through the code");
            println!("n - go to next state");
       }
        else if input == ":stack" {
            print_machine_stack(m);
        }
        else if input == ":nostep" {
            *pause_per_step = false;
            return Result::Ok(());
        }
        else if input == "n" {
            return Result::Ok(());
        }
        else {
            println!("unrecognized: |{}|, type :help for help.", input)
        }

    }
}
fn run_machine<C>(rl: &mut Editor<C>, m: &mut Machine, pause_per_step: &mut bool) ->
    Result<(), ReadlineError>
    where C: Completer {

    let mut i = 1;
    loop {
        println!("*** ITERATION: {}", i);

        if let Result::Err(e) = m.step() {
            print!("step error: {}\n", e);
            break;

        }

        print_machine(&m);
        if m.is_final_state() {
            println!("=== FINAL: {} ===", machine_get_final_val(&m));
            break;
        }

        if *pause_per_step {
            try!(run_step_interaction(rl, i, m, pause_per_step));
        }
        i += 1;
    }

    Result::Ok(())
}

fn main() {
    let mut pause_per_step = false;
    let mut m : Machine = Machine::new_minimal();
    let mut rl = Editor::<()>::new();

    loop {

        let input = match rl.readline(">") {
            Ok(line) => {
                rl.add_history_entry(&line);
                line
            }
            Err(ReadlineError::Interrupted) |
            Err(ReadlineError::Eof) => {
                break
            }
            Err(err) => {
                panic!("readline error: {}", err)
            }

        };
        if input == "" {
            continue;
        }
        if input == ":exit".to_string() {
            break;
        }
        else if input == ":step" {
            pause_per_step = true;
            println!("enabled stepping through execution");
            continue;
        }
        else if input ==":nostep" {
            pause_per_step = false;
            println!("disabled stepping through execution");
            continue;

        }
        else if input == ":help" {
            println!(":help - bring up help");
            println!(":step - enable stepping through execution");
            println!(":nostep - disable stepping through execution");
            print!("use 'define <name> [<param>]* = <expr>' to define a toplevel function");
            print!("write out an expression to have it evaluated");
        }


        //someone is defining a binding
        if input.starts_with("define ") {
            let sc_defn_str = input.trim_left_matches("define ").to_string();
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
            
            match run_machine(&mut rl, &mut m, &mut pause_per_step) {
                Result::Ok(()) => {},
                Err(ReadlineError::Interrupted) |
                Err(ReadlineError::Eof) => {
                    break
                }
                Err(err) => {
                    panic!("readline error: {}", err)
                } 
            }
        }
    }
}
