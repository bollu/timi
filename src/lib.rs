//! a Template Instantiation Machine (TIM) interpreter written in Rust.
//!
//! TIM is a particular kind of machine used
//! to implement a lazily evaluated functional programming language.
//!
//! This implementation comes with a parser for the language called as `Core`,
//! along with an interpreter for `Core`.
//! This is based on [Implementing Functional Languages, a tutorial](FIXME: add link)

#[macro_use]
extern crate prettytable;
extern crate rustyline;

/// Machine that performs interpretation. 
pub mod machine;
/// Frontend of the interpreter. Tokenization & Parsing is handled here
pub mod frontend;
/// Internal Representation (IR) of the machine. Contains the data
/// representation used by the machine.
pub mod ir; 
