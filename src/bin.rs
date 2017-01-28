#[macro_use]
extern crate prettytable;
extern crate rustyline;

use std::fs::File;


use rustyline::error::ReadlineError;
use rustyline::Editor;
use rustyline::completion::Completer;

mod frontend;
mod ir;
mod machine;

use frontend::*;
use machine::*;


fn run_step_interaction<C>(rl: &mut Editor<C>,
                           iter_count: usize,
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

fn run_machine_step(m: &mut Machine, iteration: usize) -> Result<(), MachineError> {
    println!("*** ITERATION: {}", iteration);

    if let Result::Err(e) = m.step() {
        return Result::Err(e)
    }

    print_machine(&m);
    if m.is_final_state() {
        println!("=== FINAL: {} ===", machine_get_final_val(&m));
    };

    Result::Ok(())
}

fn run_machine<C>(rl: &mut Editor<C>, m: &mut Machine, pause_per_step: &mut bool) ->
Result<(), ReadlineError>
where C: Completer {

    let mut iteration = 1;
    loop {

        match run_machine_step(m, iteration) {
            Ok(()) => {},
            Err(e) => { println!("step error: {}", e); break; }
        }

        if *pause_per_step {
            try!(run_step_interaction(rl, iteration, m, pause_per_step));
        }

        if m.is_final_state() {
            break;
        }

        iteration += 1;
    }

    Result::Ok(())
}

fn interpreter() {

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
            let sc_defn = match string_to_sc_defn(&sc_defn_str) {
                Result::Ok(defn) => defn,
                Result::Err(e) => {
                    println!("Parse Error:\n{}", e.pretty_print(&sc_defn_str));
                    continue;
                }
            };

            m.add_supercombinator(sc_defn); 
        }
        else {
            let expr = match string_to_expr(&input) {
                Result::Ok(expr) => expr,
                Result::Err(e) => {
                    println!("PARSE ERROR:\n{}", e.pretty_print(&input));
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
                    panic!("Readline failed: {}", err)
                } 
            }
        }
    }
}


fn exit_with_err(err: &str) -> ! {
    print!("{}", err);
    std::process::exit(1);
}


fn create_machine_from_file_path(path: &str) -> Machine {
    use std::io::Read;

    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) => exit_with_err(&format!(
                    "unable to open file: {}\n{}", path, e))
    };

    let mut program_string = String::new();
    if let Err(e) = file.read_to_string(&mut program_string) {
        exit_with_err(&format!(
            "unable to read from file: {}\n{}", path, e))
    }


    let program = match string_to_program(&program_string) {
        Ok(p) => p,
        Err(e) => exit_with_err(&format!(
                    "parse error:\n{}", e.pretty_print(&program_string)))
    };

    match Machine::new_with_main(program) {
        Ok(m) => return m,
        Err(e) => panic!("unable to create machine:\n{}", e)
    };

}

fn main() {
    use std::env;
    if env::args().len() == 1 {
        interpreter();
        return;
    }
    else if env::args().len() > 2 {
       println!("usage: timi [file-to-execute] [--help]");
       println!("timi is an interpreter for the template instantiation language");
       return;
    }
    
    let arg = env::args().nth(1).expect("expected 1 command line argument");

    if arg == "--help" {
       println!("usage: timi [file-to-execute] [--help]");
       println!("timi is an interpreter for the template instantiation language") ;
       println!("github: http://github.com/bollu/timi");
       return;
    }

    let mut m = create_machine_from_file_path(&arg);
    
    let mut iteration = 1;
    while !m.is_final_state() {
        match run_machine_step(&mut m, iteration) {
            Ok(()) => {}
            Err(e) => {
                println!("step error: {}", e);
                break;
            }
        }
        iteration += 1;
    }
}
