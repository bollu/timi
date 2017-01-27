extern crate ansi_term;

use std::fmt;

pub type Addr = i32;
pub type Name = String;

pub type CoreVariable = Name;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CoreLet {
    pub bindings: Vec<(Name, Box<CoreExpr>)>,
    pub expr: Box<CoreExpr>
}


#[derive(Clone, PartialEq, Eq)]
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
            &CoreExpr::Variable(ref name) => write!(fmt, "{}", name),
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
pub struct SupercombDefn {
    pub name: String,
    pub args: Vec<String>,
    pub body: CoreExpr
}


impl fmt::Debug for SupercombDefn {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(fmt, "{} ", &self.name));
        for arg in self.args.iter() {
            try!(write!(fmt, "{} ", &arg));
        }

        try!(write!(fmt, "= {{ {:#?} }}", self.body));
        Result::Ok(())

    }

}
//a core program is a list of supercombinator
//definitions
pub type CoreProgram = Vec<SupercombDefn>;



use self::ansi_term::Colour::{Green};

pub fn format_addr_string(addr: &Addr)  -> String {
    format!("{}{}", Green.paint("0x"), Green.underline().paint(format!("{:X}", addr)))
}

/*
pub fn format_name_string(name: &str) -> String {
    format!("{}", Cyan.paint(name))
}
*/


