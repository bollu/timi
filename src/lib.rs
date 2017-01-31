//! a Template Instantiation Machine (TIM) interpreter written in Rust.
//!
//! TIM is a particular kind of machine used
//! to implement a lazily evaluated functional programming language.
//!
//! This implementation comes with a parser for the language called as `Core`,
//! along with an interpreter for `Core`.
//! This is based on [Implementing Functional Languages, a tutorial](FIXME: add link)
//!
//!
//! ### Example - Creating [`Machine`](machine/struct.Machine.html) instance from a string
//!
//! ```
//! use timi::machine;
//! use timi::frontend;
//!
//! let program_str = "main = 1 + 1";
//! let parsed_program = frontend::string_to_program(program_str).unwrap();
//!
//! let mut m = machine::Machine::new_from_program(parsed_program).unwrap();
//!
//! while !m.is_final_state().unwrap() {
//!   m.step();   
//! }
//!
//! //check that the top node on the stack is 1 + 1 == 2
//! assert_eq!(machine::HeapNode::Num(2), m.heap.get(&m.stack.peek().unwrap()));
//! ```
#[macro_use]
#[warn(missing_docs)]

extern crate prettytable;
extern crate rustyline;

pub mod machine;
pub mod frontend;
pub mod ir; 
pub mod pretty_print;
