//! The expression type that is parsed by the Parser is held in `CoreExpr`
//!
//! a Core program ([`CoreProgram`](type.CoreProgram.html)) is a collection
//! of supercombinator(top-level) definitions
//! top-level consists of `SupercombDefn`.
//!
#[warn(missing_docs)]
extern crate ansi_term;

use std::fmt;

/// A Heap address.
pub type Addr = i32;

/// A variable name.
pub type Name = String;

#[derive(Clone, PartialEq, Eq, Debug)]
/// Core Expression struct to hold let-bindings.
/// 
/// a binding is a mapping from a variable name to the bound expression
pub struct CoreLet {
    ///bindings in the `let` expression
    pub bindings: Vec<(Name, Box<CoreExpr>)>,
    /// Expression that is at the `in ...` part of the let
    pub expr: Box<CoreExpr>
}

#[derive(Clone, PartialEq, Eq)]
/// A Core language expression
pub enum CoreExpr {
    //change this?
    Variable(Name),
    Num(i32),
    Application(Box<CoreExpr>, Box<CoreExpr>),
    Pack{tag: u32, arity: u32},
    Let(CoreLet),
}

impl fmt::Debug for CoreExpr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &CoreExpr::Variable(ref name) => write!(fmt, "v_{}", name),
            &CoreExpr::Num(ref num) => write!(fmt, "n_{}", num),
            &CoreExpr::Application(ref e1, ref e2) =>
                write!(fmt, "({:#?} $ {:#?})", *e1, *e2),
            &CoreExpr::Let(CoreLet{ref bindings, ref expr}) => {
                try!(write!(fmt, "let"));
                try!(write!(fmt, " {{\n"));
                for &(ref name, ref expr) in bindings {
                    try!(write!(fmt, "{} = {:#?}\n", name, expr));
                }
                try!(write!(fmt, "in\n"));
                try!(write!(fmt, "{:#?}", expr));
                write!(fmt, "}}")
            }
            &CoreExpr::Pack{ref tag, ref arity} => {
                write!(fmt, "Pack(tag: {} arity: {})", tag, arity)
            }
        }
    }
}


#[derive(Clone, PartialEq, Eq)]
/// A supercombinator definition, consisting of the name, arguments,
/// and body.
pub struct SupercombDefn {
    /// name of the supercombinator
    pub name: String,
    /// name of the arguments (formal parameters)
    pub args: Vec<String>,
    /// body of the supercombinator
    pub body: CoreExpr
}


impl fmt::Debug for SupercombDefn {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(fmt, "{} ", &self.name));
        for arg in self.args.iter() {
            try!(write!(fmt, "{} ", &arg));
        }

        try!(write!(fmt, "= {{ {:#?} }}", self.body));
        Ok(())

    }

}

/// A core program is a list of top-level supercombinator
/// definitions
pub type CoreProgram = Vec<SupercombDefn>;
