//!  
//! The machine state is formed out of 4 components: `(Stack, Heap, Dump, Globals)`
extern crate ansi_term;

use std::fmt;
use std::collections::HashMap;
use std::collections::HashSet;

use std;
use std::fmt::Write;

use ir::*;
use frontend;


use self::ansi_term::Colour::{Blue, Red, Black};
use self::ansi_term::Style;

use prettytable::Table;
use prettytable::format::consts::FORMAT_CLEAN;


/// Primitive operations that are implemented directly in the machine
#[derive(Clone, PartialEq, Eq)]
pub enum MachinePrimOp {
    /// Add two machine integers
    Add,
    /// Subtract two machine integers
    Sub,
    /// Multiply two machine integers
    Mul,
    /// Divide two machine integers
    Div,
    /// Negate a machine integer
    Negate,
    /// Compare two integers and return if LHS > RHS
    G,
    /// Compare two integers and return if LHS >= RHS
    GEQ,
    /// Compare two integers and return if LHS < RHS
    L,
    /// Compare two integers and return if LHS >= RHS
    LEQ,
    /// Compare two integers and return if LHS == RHS
    EQ,
    /// Compare two integers and return if LHS != RHS
    NEQ,
    /// Construct a complex object which is tagged with `DataTag`
    /// and takes `arity` number of components.
    Construct {
        /// Tag used to disambiguate between different data objects
        tag: DataTag,
        /// number of components that the complex object has.
        arity: u32
    },
    /// Check a predicate and run the `then` or `else` clause
    If,
    /// Explode a tuple and pass the `left` and `right` components of a tuple to a function
    CasePair,
    /// Perform case analysis on a list.
    CaseList,
    /// Undefined. Machine will quit on reaching this address. Useful for testing code that
    /// should never run
    Undef,
}

impl fmt::Debug for MachinePrimOp {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &MachinePrimOp::Negate => write!(fmt, "Negate"),
            &MachinePrimOp::Add => write!(fmt, "+"),
            &MachinePrimOp::Sub => write!(fmt, "-"),
            &MachinePrimOp::Mul => write!(fmt, "*"),
            &MachinePrimOp::Div => write!(fmt, "/"),
            &MachinePrimOp::G => write!(fmt, ">"),
            &MachinePrimOp::L => write!(fmt, "<"),
            &MachinePrimOp::GEQ => write!(fmt, ">="),
            &MachinePrimOp::LEQ => write!(fmt, "<="),
            &MachinePrimOp::EQ => write!(fmt, "=="),
            &MachinePrimOp::NEQ => write!(fmt, "!="),
            &MachinePrimOp::If => write!(fmt, "if"),
            &MachinePrimOp::CasePair => write!(fmt, "casePair"),
            &MachinePrimOp::CaseList => write!(fmt, "caseList"),
            &MachinePrimOp::Undef => write!(fmt, "undef"),
            &MachinePrimOp::Construct{ref tag, ref arity} => {
                write!(fmt, "Construct(tag:{:#?} | arity: {})", tag, arity)
            }
        }
    }
}


#[derive(Clone,PartialEq,Eq,Debug)]
/// Used to tag data in `HeapNode::Data`.
pub enum DataTag {
    TagFalse = 0,
    TagTrue = 1,
    TagPair = 2,
    TagListNil = 3,
    TagListCons = 4,
}

fn raw_tag_to_data_tag (raw_tag: u32) -> Result<DataTag, MachineError> {
    match raw_tag {
        0 => Result::Ok(DataTag::TagFalse),
        1 => Result::Ok(DataTag::TagTrue),
        2 => Result::Ok(DataTag::TagPair),
        3 => Result::Ok(DataTag::TagListNil),
        4 => Result::Ok(DataTag::TagListCons),
        other @ _ => Result::Err(format!(
                "expected False(0), \
                 True(1), or Pair(2). \
                 found: {}",
                 other))
    }
} 


#[derive(Clone, PartialEq, Eq)]
/// an element on the [`Heap`](struct.HeapNode) of the machine.
pub enum HeapNode {
    /// Function application of function at `fn_addr` to function at `arg_addr`
    Application {
        fn_addr: Addr,
        arg_addr: Addr
    },
    /// a Supercombinator (top level function) that is created on the heap
    Supercombinator(SupercombDefn),
    /// Raw integer
    Num(i32),
    /// An indirection from the current heap address to another heap address.
    /// Used to reduce work done by machine by preventing re-computation.
    Indirection(Addr),
    /// A primitive that needs to be executed.
    Primitive(MachinePrimOp),
    /// Complex data on the Heap with tag `tag`. Components are at the
    /// `component_addrs`.
    Data{tag: DataTag, component_addrs: Vec<Addr>}
}


/// makes the heap tag bold
fn format_heap_tag(s: &str) -> String {
    format!("{}", Style::new().bold().paint(s))
}

impl fmt::Debug for HeapNode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {


        match self {
            &HeapNode::Application{ref fn_addr, ref arg_addr} => {
                write!(fmt, "{}({} $ {})", 
                       format_heap_tag("H-Ap"),
                       format_addr_string(fn_addr),
                       format_addr_string(arg_addr))
            }
            &HeapNode::Supercombinator(ref sc_defn) => {
                write!(fmt, "{}({:#?})", 
                       format_heap_tag("H-Supercombinator"),
                       sc_defn)
            },
            &HeapNode::Num(ref num)  => {
                write!(fmt, "{}({})", 
                       format_heap_tag("H-Num"),
                       num)
            }
            &HeapNode::Indirection(ref addr)  => {
                write!(fmt, "{}({})",
                format_heap_tag("H-Indirection"),
                format_addr_string(addr))
            }
            &HeapNode::Primitive(ref primop)  => {
                write!(fmt, "{}({:#?})",
                format_heap_tag("H-Primitive"),
                primop)
            },
            &HeapNode::Data{ref tag, ref component_addrs} => {
                try!(write!(fmt, "{}(tag: {:#?}",
                            format_heap_tag("H-Data"),
                            tag));

                if component_addrs.len() == 0 {
                    try!(write!(fmt, ")"))
                }
                else {
                    try!(write!(fmt, " | data: "));
                    for addr in component_addrs.iter() {
                        try!(write!(fmt, "{} ", format_addr_string(addr)));
                    }
                    try!(write!(fmt, ")"));
                }
                Result::Ok(())

            }
        }
    }
}

impl HeapNode {
    fn is_data_node(&self) -> bool {
        match self {
            &HeapNode::Num(_) => true,
            &HeapNode::Data{..} => true,
            _ => false
        }
    }
}


/// format a heap node, by pretty printing the node.
/// If the node contains recursive structure, this will handle it and
/// print `<<recursive_defn>>`
fn format_heap_node(heap: &Heap, addr: &Addr) -> String {
    
    if is_heap_node_cyclic(heap, addr, &mut HashSet::new(), &mut HashSet::new()) {
        return "<<recursive defn>>".to_string();
    }
    

    match heap.get(addr) {
        HeapNode::Indirection(addr) => format!("indirection({})", format_heap_node(heap, &addr)),
        HeapNode::Num(num) => format!("{}", num),
        HeapNode::Primitive(ref primop) => format!("{:#?}", primop),
        HeapNode::Application{ref fn_addr, ref arg_addr} =>
            format!("({} {})",
            format_heap_node(heap, fn_addr),
            format_heap_node(heap, arg_addr)),
            HeapNode::Supercombinator(ref sc_defn) =>  {
                let mut sc_str = String::new();
                write!(&mut sc_str, "{}", sc_defn.name).unwrap();
                sc_str
            }
        HeapNode::Data{tag: DataTag::TagTrue, ..} => {
            format!("True")
        }
        HeapNode::Data{tag: DataTag::TagFalse, ..} => {
            format!("False")
        }
        HeapNode::Data{tag: DataTag::TagPair, ref component_addrs} => {
            let left_addr = component_addrs
                                .get(0)
                                .expect("left component of tuple expected");
            let right_addr = component_addrs
                                 .get(1)
                                 .expect("right component of tuple expected");
            format!("({}, {})",
            format_heap_node(heap, &left_addr),
            format_heap_node(heap, &right_addr))
        }
        HeapNode::Data{tag: DataTag::TagListNil, ..} => {
            format!("[]")
        }
        HeapNode::Data{tag: DataTag::TagListCons, ref component_addrs} => {
            let left_addr = component_addrs
                                .get(0)
                                .expect("expected left component \
                                        of list constructor");
            let right_addr = component_addrs
                                 .get(1)
                                 .expect("expected right component of list\
                                         constructor");

            format!("{}:{}", format_heap_node(heap, &left_addr),
            format_heap_node(heap, &right_addr))

        }
    }
}


/// returns if the Heap node at address `addr`
/// contains a cyclic definition or not.
fn is_heap_node_cyclic(heap: &Heap, addr: &Addr,
                               mut processed: &mut HashSet<Addr>,
                               mut rec_stack: &mut HashSet<Addr>) -> bool {

    fn get_node_neighbours(node: &HeapNode) -> Vec<Addr> {
        match node {
            &HeapNode::Indirection(ref addr) => {
                vec![*addr]
            }
            &HeapNode::Application{ref fn_addr, ref arg_addr} => {
                vec![*fn_addr, *arg_addr]
            }
            &HeapNode::Data{ref component_addrs, ..} => {
                component_addrs.clone()
            }
            &HeapNode::Supercombinator(_) | 
                &HeapNode::Primitive(..) |
                &HeapNode::Num(_) => {Vec::new()}
        }
    };

    if is_addr_phantom(addr) {
        return false;
    }

    //we have reached a processed node, so quit
    if processed.contains(addr) {
        rec_stack.remove(&addr);
        return false;
    } else {
        processed.insert(*addr);
        rec_stack.insert(*addr);

        let neighbours = get_node_neighbours(&heap.get(addr));
        for n in neighbours {
            if rec_stack.contains(&n) {
                return true;
            }
            else if is_heap_node_cyclic(heap, &n, processed, rec_stack) {
                return true;
            }
        }
    }

    false
}


/// Gives addresses pointed to in the heap node at address `addr`.
/// This is recursive and produces a list of _all_ addresses by walking the
/// graph of address dependencies.
///
/// # Use Case
/// Collect all addresses that are currently being used.
/// That way, we don't need to print all of the heap nodes during
/// pretty-printing of machine state.
/// 
/// This is useful since the heap grows quite large very quickly.
fn collect_addrs_from_heap_node(heap: &Heap,
                                addr: &Addr,
                                mut collection: &mut HashSet<Addr>) {

    if collection.contains(addr) {
        return;
    }
    if is_addr_phantom(addr) {
        return;
    }

    collection.insert(*addr);

    match heap.get(addr) {
        HeapNode::Indirection(ref addr) => {
            collection.insert(*addr);
            collect_addrs_from_heap_node(heap, addr, collection);
        }
        HeapNode::Application{ref fn_addr, ref arg_addr} => {
            collection.insert(*fn_addr);
            collection.insert(*arg_addr);
            collect_addrs_from_heap_node(heap, fn_addr, &mut collection);
            collect_addrs_from_heap_node(heap, arg_addr, &mut collection);
        }
        HeapNode::Data{ref component_addrs, ..} => {
            for addr in component_addrs.iter() {
                collection.insert(*addr);
                collect_addrs_from_heap_node(heap, addr, &mut collection);
            }
        }
        HeapNode::Supercombinator(_) | 
            HeapNode::Primitive(..) |
            HeapNode::Num(_) => {}
    };
}


/// Tries to unwrap a heap node to an application node,
/// fails if the heap node is not application.
///
/// # Use Case
/// unwrapping application nodes is a very common process
/// when implementing primitives. This abstracts out the process
/// and reduces code duplication
fn unwrap_heap_node_to_ap(node: HeapNode) -> 
Result<(Addr, Addr), MachineError> {

    match node {
        HeapNode::Application{fn_addr, arg_addr} => 
            Result::Ok((fn_addr, arg_addr)),
            other @ _ => Result::Err(format!(
                    "expected application node, \
                                         found: {:#?}", other))
    }
}

/// Dump is a stack of [machine `Stack`](struct.Stack.html).
///
/// # Use Case
/// It is used to store intermediate execution of primitives.
/// That way, a complex computation can be stored on a dump. 
/// A sub computation can be run, after which the complex computation can be
/// brought back into scope.
pub type Dump = Vec<Stack>;

/// Stack is a stack of [`Addr`](../ir/type.Addr.html).
/// 
/// # Use Case
/// Function application is "unwound" on the stack.
#[derive(Clone,PartialEq,Eq,Debug)]
pub struct Stack {
    stack: Vec<Addr>
}

impl Stack {
    pub fn new() -> Stack {
        Stack {
            stack: Vec::new(),
        }
    }

    /// return number of elements on the stack.
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    /// push an address on top of the stack.
    pub fn push(&mut self, addr: Addr) {
        self.stack.push(addr)
    }

    /// pops the top of the stack.
    ///
    /// # Errors
    /// returns an error if the stack is empty
    pub fn pop(&mut self) -> Result<Addr, MachineError> {
        self.stack.pop().ok_or("top of stack is empty".to_string())
    }

    /// peeks the top of the stack.
    ///
    /// *NOTE:*  does _not_ remove the element on the top of the stack.
    pub fn peek(&self) -> Result<Addr, MachineError> {
        self.stack.last().cloned().ok_or("top of stack is empty to peek".to_string())
    }

    /// returns an iterator to the stack elements. 
    ///
    /// *NOTE:* top of the stack is returned first, bottom of the stack
    /// is returned last.
    pub fn iter(&self) -> std::iter::Rev<std::slice::Iter<Addr>> {
        self.stack.iter().rev()
    }

}

/// Mapping from [variable names](../ir/type.Name.html) to
/// [addresses](type.Addr.html).
pub type Environment = HashMap<Name, Addr>;

#[derive(Clone)]
/// maps Address to Heap nodes. 
pub struct Heap {
    heap: HashMap<Addr, HeapNode>,
    next_addr: Addr
}

impl fmt::Debug for Heap {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut keyvals : Vec<(&Addr, &HeapNode)> = self.heap.iter().collect();
        keyvals.sort_by(|a, b| a.0.cmp(b.0));

        for &(key, val) in keyvals.iter().rev() {
            try!(write!(fmt, "\t{} => {:#?}\n", format_addr_string(key), val));
        }

        return Result::Ok(())

    }
}

impl Heap {
    pub fn new()  -> Heap {
        Heap {
            heap: HashMap::new(),
            next_addr: 0
        }
    }

    /// Allocate the HeapNode on the heap. returns the address at which
    /// the node was allocated.
    pub fn alloc(&mut self, node: HeapNode) -> Addr {
        let addr = self.next_addr;
        self.next_addr += 1;

        self.heap.insert(addr, node);
        addr
    }

    /// returns the heap node at address Addr
    ///
    /// ### Panics
    ///
    /// `get` panics if the address does not exist on the heap. To check
    /// if an address is on the heap, use [contains](struct.Heap.html#method.contains)
    pub fn get(&self, addr: &Addr) -> HeapNode {
        self.heap
            .get(&addr)
            .cloned()
            .expect(&format!("expected heap node at addess: {}", addr))
    }

    /// rewrites the address `addr` with new heap node `node`
    ///
    /// ### Panics
    /// if the heap node at `addr` does not exist, this function will panic
    pub fn rewrite(&mut self, addr: &Addr, node: HeapNode) {
        assert!(self.heap.contains_key(addr),
        "asked to rewrite (address: {}) with \
        (node: {:#?}) which does not exist on heap",
        addr, node);
        self.heap.insert(*addr, node);
    }

    /// returns the number of elements in the heap
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// returns whether the heap contains an element at address `addr`.
    pub fn contains(&self, addr: &Addr) -> bool {
        match self.heap.get(&addr) {
            Some(_) => true,
            None => false
        }
    }

}

/// Options used to configure the machine
#[derive(Clone)]
pub struct MachineOptions {
    /// controls whether a function application node should be rewritten
    /// with the final value after evaluation.
    ///
    /// Enable this to see the use of `Indirection` nodes.
    pub update_heap_on_sc_eval: bool,
}

#[derive(Clone)]
/// the core machine that runs our program.
pub struct Machine {
    pub stack : Stack,
    pub heap : Heap,
    pub globals: Environment,
    pub dump: Dump,
    pub options: MachineOptions,
}

/// represents an error during the execution of the machine
pub type MachineError = String;


/// converts a boolean into a HeapNode for True and False
fn bool_to_heap_node(b: bool) -> HeapNode {
    if b {
        HeapNode::Primitive(MachinePrimOp::Construct{tag: DataTag::TagTrue,
            arity: 0})
    }
    else {
        HeapNode::Primitive(MachinePrimOp::Construct{tag: DataTag::TagFalse,
            arity: 0})
    }
}


/// constructs the prelude program for the machine.
fn get_prelude() -> CoreProgram {

    let program_str = "I x = x;\n\
                        K x y = x;\n\
                        K1 x y = y;\n\
                        S f g x = f x (g x);\n\
                        compose f g x = f (g x);\n\
                        twice f = compose f f;\n\
                        False = Pack{0, 0};\n\
                        True = Pack{1, 0};\n\
                        MkPair = Pack{2, 2};\n\
                        Nil = Pack{3, 0};\n\
                        Cons = Pack{4, 2};\n\
                        Y f = f (Y f);\n\
                        facrec f n = if (n == 0) 1 (n * f (n - 1));\n\
                        fac n = (Y facrec) n\n\
                        ";
    match frontend::string_to_program(program_str) {
        Result::Ok(program) => program,
        Result::Err(e) => panic!("prelude compilation failed:\n{}", e.pretty_print(program_str))
    }
}

/// mapping between strings for primitives and the primitive operation
fn get_primitives() -> Vec<(Name, MachinePrimOp)> {
    [("+".to_string(), MachinePrimOp::Add),
    ("-".to_string(), MachinePrimOp::Sub),
    ("*".to_string(), MachinePrimOp::Mul),
    ("/".to_string(), MachinePrimOp::Div),
    (">".to_string(), MachinePrimOp::G),
    ("<".to_string(), MachinePrimOp::L),
    (">=".to_string(), MachinePrimOp::GEQ),
    ("<=".to_string(), MachinePrimOp::LEQ),
    ("!=".to_string(), MachinePrimOp::NEQ),
    ("==".to_string(), MachinePrimOp::EQ),
    ("negate".to_string(), MachinePrimOp::Negate),
    ("if".to_string(), MachinePrimOp::If),
    ("casePair".to_string(), MachinePrimOp::CasePair),
    ("caseList".to_string(), MachinePrimOp::CaseList),
    ("undef".to_string(), MachinePrimOp::Undef)
    ].iter().cloned().collect()
}


/// builds the initial heap and environment corresponding to the
/// program given.
///
/// This allocates heap nodes for supercombinators in `sc_defs` and
/// primitives in `prims` and maps their names
/// to the allocated addresses in `Environment`.
fn build_heap_and_env_for_program(sc_defs: CoreProgram,
                                  prims: Vec<(Name, MachinePrimOp)>) 
    -> (Heap, Environment) {

        let mut heap = Heap::new();
        let mut globals = HashMap::new();

        for sc_def in sc_defs.iter() {
            //create a heap node for the Supercombinator definition
            //and insert it
            let node = HeapNode::Supercombinator(sc_def.clone());
            let addr = heap.alloc(node);

            //insert it into the globals, binding the name to the
            //heap address
            globals.insert(sc_def.name.clone(), addr);
        }

        for (name, prim_op) in prims.into_iter() {
            let addr = heap.alloc(HeapNode::Primitive(prim_op));
            globals.insert(name, addr);
        }

        (heap, globals)
    }

impl Machine {
    /// Create a minimal machine that has the prelude and primitives instantiated.
    pub fn new_minimal() -> Machine {
        let (initial_heap, globals) = build_heap_and_env_for_program(get_prelude(),
        get_primitives());

        Machine {
            dump: Vec::new(),
            stack:  Stack::new(),
            globals: globals,
            heap: initial_heap,
            options: MachineOptions {
                update_heap_on_sc_eval: true
            }
        }

    }

    /// Create a machine with the given program, which is assured to have `main` as a
    /// supercombinator.
    /// This creates the prelude, primitives, as well as all the supercombinators in `program`.
    /// It sets up the stack to have `main` on the top of the stack.
    ///
    /// ### Errors
    /// If `program` does not have `main`, a `MachineError` is returned.
    pub fn new_with_main(program: CoreProgram) -> Result<Machine, MachineError> {
        let mut m = Machine::new_minimal();
        for sc in program.into_iter() {
            m.add_supercombinator(sc);
        }

        //get main out of the heap
        let main_addr : Addr = match m.globals.get("main") {
            Some(main) => main,
            None => return Result::Err("no main found in given program".to_string())
        }.clone();

        m.stack.push(main_addr);
        Result::Ok(m)
    }

    /// Add the supercombinator to the machine. 
    ///
    /// This will allocate the supercombinator on the heap and create a binding
    /// in the environment to the name of the supercombinator.
    /// 
    /// Returns the address of allocation of the supercombinator
    pub fn add_supercombinator(&mut self, sc_defn: SupercombDefn) -> Addr{
        let name = sc_defn.name.clone();
        let node = HeapNode::Supercombinator(sc_defn);
        let addr = self.heap.alloc(node);

        //insert it into the globals, binding the name to the
        //heap address
        self.globals.insert(name, addr);
        addr
    }

    /// Creates a supercombinator, names it `main` and sets its body to the
    /// given expression. Then starts execution of the given expression by
    /// putting main on top of the stack.
    pub fn run_expr_as_main(&mut self, expr: &CoreExpr) {
        let main_defn = SupercombDefn {
            name: "main".to_string(),
            body: expr.clone(),
            args: Vec::new()
        };

        let main_addr = self.add_supercombinator(main_defn);
        self.stack = Stack { stack: vec![main_addr] };
    }

    /// returns whether the machine is in final state or not.
    /// ### Panics
    ///
    /// This panics if the stack is empty.
    pub fn is_final_state(&self) -> bool {
        assert!(self.stack.len() > 0, "expect stack to have at least 1 node");

        if self.stack.len() > 1 {
            false
        } else {
            let dump_empty = self.dump.len() == 0;
            self.heap.get(&self.stack.peek().unwrap()).is_data_node() &&
                dump_empty
        }
    }

    /// dump the current stack into the dump, and create a fresh stack
    fn dump_stack(&mut self, stack: Stack) {
        self.dump.push(stack);
        self.stack = Stack::new();
    }

    /// inspect the top of the stack and take a step in interpretation if
    /// we are not in the final state
    pub fn step(&mut self) -> Result<(), MachineError> { 
        //top of stack
        let tos_addr : Addr = try!(self.stack.peek());
        let heap_val = self.heap.get(&tos_addr);


        //there is something on the dump that wants to use this
        //data node, so pop it back.
        if heap_val.is_data_node() && self.dump.len() > 0 {
            self.stack = self.dump
                .pop()
                .expect("dump should have at least 1 element");
            Result::Ok(())
        } else {
            self.heap_node_step(&heap_val)
        }
    }

    /// perform an interpretation step by case analysis of the given heap node
    /// (which is the top of the stack)
    fn heap_node_step(&mut self, tos_node: &HeapNode) -> Result<(), MachineError> {
        match tos_node {
            &HeapNode::Num(n) => {
                return Result::Err(format!("number applied as a function: {}", n));
            }

            data @ &HeapNode::Data{..} => {
                return Result::Err(format!(
                        "data node applied as function: {:#?}", data));
            }
            &HeapNode::Application{fn_addr, ..} => {
                //push function address over the function
                self.stack.push(fn_addr);
            }
            &HeapNode::Indirection(ref addr) => {
                //simply ignore an indirection during execution, and
                //push the indirected value on the stack
                try!(self.stack.pop());
                self.stack.push(*addr);
            }
            //expand supercombinator
            &HeapNode::Supercombinator(ref sc_defn) => {
                try!(run_supercombinator(self, sc_defn));
            }

            &HeapNode::Primitive(MachinePrimOp::Negate) => {
                try!(run_primitive_negate(self));
            }
            &HeapNode::Primitive(MachinePrimOp::Add) => {
                try!(run_primitive_num_binop(self,
                    |x, y| HeapNode::Num(x + y)));
            }
            &HeapNode::Primitive(MachinePrimOp::Sub) => {
                try!(run_primitive_num_binop(self,
                    |x, y| HeapNode::Num(x - y)));
            }
            &HeapNode::Primitive(MachinePrimOp::Mul) => {
                try!(run_primitive_num_binop(self,
                    |x, y| HeapNode::Num(x * y)));
            }
            &HeapNode::Primitive(MachinePrimOp::Div) => {
                try!(run_primitive_num_binop(self,
                    |x, y| HeapNode::Num(x / y)));
            }
            //construct a complex type
            &HeapNode::Primitive(MachinePrimOp::Construct {ref tag, arity}) => {
                try!(run_constructor(self, tag, arity));
            }
            //boolean ops
            &HeapNode::Primitive(MachinePrimOp::G) => {
                try!(run_primitive_num_binop(self,
                    |x, y| bool_to_heap_node(x > y)));
            }
            &HeapNode::Primitive(MachinePrimOp::GEQ) => {
                try!(run_primitive_num_binop(self,
                        |x, y| bool_to_heap_node(x >= y)));
            }
            &HeapNode::Primitive(MachinePrimOp::L) => {
                try!(run_primitive_num_binop(self,
                    |x, y| bool_to_heap_node(x < y)));
            }
            &HeapNode::Primitive(MachinePrimOp::LEQ) => {
                try!(run_primitive_num_binop(self,
                    |x, y| bool_to_heap_node(x <= y)));
            }
            &HeapNode::Primitive(MachinePrimOp::EQ) => {
                try!(run_primitive_num_binop(self,
                    |x, y| bool_to_heap_node(x == y)));
            }
            &HeapNode::Primitive(MachinePrimOp::NEQ) => {
                try!(run_primitive_num_binop(self,
                    |x, y| bool_to_heap_node(x != y)));
            }
            //run if condition
            &HeapNode::Primitive(MachinePrimOp::If) => {
                try!(run_primitive_if(self));
            }
            &HeapNode::Primitive(MachinePrimOp::CasePair) => {
                try!(run_primitive_case_pair(self));
            }
            &HeapNode::Primitive(MachinePrimOp::CaseList) => {
                try!(run_primitive_case_list(self));
            }
            &HeapNode::Primitive(MachinePrimOp::Undef) => {
                return Result::Err("hit undefined operation".to_string())
            }
        };
        Result::Ok(())
    }


    /// given a supercombinator, realize it on the heap
    /// by recursively instantiating its body, with respect to the environment `env.
    /// The environment is used to find variable names.
    fn instantiate(&mut self, expr: CoreExpr, env: &Environment) -> Result<Addr, MachineError> {
        match expr {
            CoreExpr::Let(CoreLet{expr: let_rhs, bindings, ..}) => {
                let let_env = try!(instantiate_let_bindings(self, env, bindings));
                self.instantiate(*let_rhs, &let_env)
            }
            CoreExpr::Num(x) => Result::Ok(self.heap.alloc(HeapNode::Num(x))),
            CoreExpr::Application(fn_expr, arg_expr) => {
                let fn_addr = try!(self.instantiate(*fn_expr, env));
                let arg_addr = try!(self.instantiate(*arg_expr, env));

                Result::Ok(self.heap.alloc(HeapNode::Application {
                    fn_addr: fn_addr,
                    arg_addr: arg_addr
                }))

            }
            CoreExpr::Variable(vname) => {
                match env.get(&vname) {
                    Some(addr) => Result::Ok(*addr),
                    None => Result::Err(format!("unable to find variable in heap: |{}|", vname))
                }

            }
            CoreExpr::Pack{tag, arity} => {
                let prim_for_pack = 
                    HeapNode::Primitive(MachinePrimOp::Construct{
                        tag: try!(raw_tag_to_data_tag(tag)),
                        arity: arity
                    });

                Result::Ok(self.heap.alloc(prim_for_pack))

            } 
        }
    }
}

///this will return a node that the address is bound to on the heap.
///if there is no instantiation (which is possible in cases like)
///let y = x; x = y in 10
///this will return no address.
fn find_root_heap_node_for_addr(fake_addr: &Addr,
                                fake_to_instantiated_addr: &HashMap<Addr, Addr>,
                                heap: &Heap) -> Option<Addr> {
    let mut visited : HashSet<Addr> = HashSet::new();
    let mut cur_addr = *fake_addr;

    loop {
        if visited.contains(&cur_addr) {
            return None;
        }

        visited.insert(cur_addr);

        if heap.contains(&cur_addr) {
            return Some(cur_addr)
        } else {
            cur_addr = *fake_to_instantiated_addr
                        .get(&cur_addr)
                        .expect(&format!("expected to find address
                                that is not in heap as a 
                                let-created address: {}", cur_addr));

        }
    }




}

/// let bindings first bind to "phantom" addresses, after which
/// they relink addresses to their correct locations. This
/// lets us check if an address is phantom to prevent us from trying to access
/// these 
/// TODO: create an alegbraic data type for Addr to represent this, and not
/// just using negative numbers. This is a hack.
fn is_addr_phantom(addr: &Addr) -> bool {
    *addr < 0

}

/// instantiate let bindings
///
/// This tricky case needs to be handled:
/// ```
/// let y = x; x = 10 in y + y
/// ```
/// here, if we instantiate `y`, then `x`, and then continue
/// replacing addresses, `y` will get the temporary address of `x`.
/// so, we need to use the "real" addresses of values
fn instantiate_let_bindings(m: &mut Machine,
                            orig_env: &Environment,
                            bindings: Vec<(Name, Box<CoreExpr>)>) 
    -> Result<Environment, MachineError> {

        let mut env : Environment = orig_env.clone();

        for (&(ref name, _), addr) in bindings.iter().zip(1..(bindings.len()+2))  {
            env.insert(name.clone(), -(addr as i32));
        }

        let mut fake_to_instantiated_addr: HashMap<Addr, Addr> = HashMap::new();
        let mut fake_addr_to_name: HashMap<Addr, String> = HashMap::new();

        //instantiate RHS, while storing legit LHS addresses
        for (bind_name, bind_expr) in bindings.into_iter() {
            let inst_addr = try!(m.instantiate(*bind_expr.clone(), &env));
            let fake_addr = try!(env.get(&bind_name)
                                .ok_or(format!("unable to find |{}| in env", bind_name))).clone();


            fake_to_instantiated_addr.insert(fake_addr, inst_addr);
            fake_addr_to_name.insert(fake_addr, bind_name);
        }


        for  (&fake_addr, _) in fake_to_instantiated_addr.iter() {
            let name = fake_addr_to_name
                                .get(&fake_addr)
                                .expect("environment must have let-binding name")
                                .clone();

            let new_addr = match find_root_heap_node_for_addr(&fake_addr,
                                                              &fake_to_instantiated_addr,
                                                              &m.heap) {
                Some(addr) => addr,
                None => return Err(format!(
                        "variable contains cyclic definition: {}",
                         name))
                        
            };

            //replace address in globals
            env.insert(name, new_addr);
            //change all the "instantiation addresses" to the actual
            //root of the heap node
            for &inst_addr in fake_to_instantiated_addr.values() {

                //something was actually instantiated: that is, it wasn't a variable
                //pointing to another variable
                //TODO: make this an algebraic data type rather than using negatives
                //let x = 10; y = x in y will work for this
                if !is_addr_phantom(&inst_addr) {
                    change_addr_in_heap_node(fake_addr,
                                             new_addr,
                                             inst_addr,
                                             &mut HashSet::new(),
                                             &mut m.heap)
                }
            }

        }

        Result::Ok(env)

    }


///edits the address recursively in the given heap node to
///replace the `fake_addr` with `new_addr` **at** `edit_addr` node.
///edited_addrs is a bookkeeping device used to make sure we don't recursively
///edit things ad infitum.
///
/// Example of recursively editing addresses:
/// let x = y x; y = x y
/// 0 -> Ap (1 $ 0)
/// 1 -> Ap (0 $ 1)
fn change_addr_in_heap_node(fake_addr: Addr,
                            new_addr: Addr,
                            edit_addr: Addr,
                            mut edited_addrs: &mut HashSet<Addr>,
                            mut heap: &mut Heap) {

    if is_addr_phantom(&edit_addr) {
        return;
    }

    if edited_addrs.contains(&edit_addr) {
        return;
    } else { 
        edited_addrs.insert(edit_addr);
    }

    match heap.get(&edit_addr) {
        HeapNode::Data{component_addrs, tag} => {

            let mut new_addrs = Vec::new();
            for i in 0..component_addrs.len() {
                if component_addrs[i] == fake_addr {
                    new_addrs[i] = new_addr;
                }
                else {
                    new_addrs[i] = component_addrs[i];
                    change_addr_in_heap_node(fake_addr,
                                             new_addr,
                                             new_addrs[i],
                                             &mut edited_addrs,
                                             heap);
                }
            };

            heap.rewrite(&edit_addr, 
                         HeapNode::Data{component_addrs: new_addrs,
                         tag:tag})
        },
        HeapNode::Application{fn_addr, arg_addr} => {
            let new_fn_addr = if fn_addr == fake_addr {
                new_addr
            } else {
                fn_addr
            };


            let new_arg_addr = if arg_addr == fake_addr {
                new_addr
            } else {
                arg_addr
            };

            heap.rewrite(&edit_addr,
                         HeapNode::Application{
                             fn_addr: new_fn_addr,
                             arg_addr: new_arg_addr
                         });

            //if we have not replaced, then recurse
            //into the application calls
            if fn_addr != fake_addr {
                change_addr_in_heap_node(fake_addr,
                                         new_addr,
                                         fn_addr,
                                         &mut edited_addrs,
                                         &mut heap);

            };

            if arg_addr != fake_addr {
                change_addr_in_heap_node(fake_addr,
                                         new_addr,
                                         arg_addr,
                                         &mut edited_addrs,
                                         &mut heap);
            };


        },
        HeapNode::Indirection(ref addr) =>
            change_addr_in_heap_node(fake_addr,
                                     new_addr,
                                     *addr,
                                     &mut edited_addrs,
                                     &mut heap),

                                     HeapNode::Primitive(_) => {}
        HeapNode::Supercombinator(_) => {}
        HeapNode::Num(_) => {},
    }

}





//TODO: rewrite code to follow the new style of peeking all of them
//and then popping off all of them
fn run_primitive_negate(m: &mut Machine) -> Result<(), MachineError> {
    //we need a copy of the stack to push into the dump
    let stack_copy = m.stack.clone();

    //pop the primitive off
    try!(m.stack.pop());

    //we rewrite this addres in case of
    //a raw number
    let neg_ap_addr = try!(m.stack.peek());

    //Apply <negprim> <argument>
    //look at what argument is and dispatch work
    let to_negate_val = 
        match try!(setup_heap_node_access(m,
                                          stack_copy, 
                                          neg_ap_addr,
                                          heap_try_num_access)) {
            HeapAccessValue::Found(val) => val,
            HeapAccessValue::SetupExecution => return Result::Ok(())
        };

    m.heap.rewrite(&neg_ap_addr, HeapNode::Num(-to_negate_val));
    Result::Ok(())
}

    //0: if 
    //1: if $ <cond> <- if_ap_addr
    //2: if <cond> $ <then>
    //3: if <cond> <then> $ <else>
    fn run_primitive_if(m: &mut Machine) -> Result<(), MachineError> {
        let stack_copy = m.stack.clone();



        let if_ap_addr = try!(m.stack
                              .iter()
                              .nth(1)
                              .cloned()
                              .ok_or("expected condition, was not found on stack"));

        let then_ap_addr = try!(m.stack
                                .iter()
                                .nth(2)
                                .ok_or("expected then application, was not found on stack".to_string())).clone();

        let else_ap_addr = try!(m.stack
                                .iter()
                                .nth(3)
                                .ok_or("expected else application, was not found on stack".to_string())).clone();

        let cond : bool = {
            match try!(setup_heap_node_access(m,
                                              stack_copy,
                                              if_ap_addr,
                                              heap_try_bool_access)) {
                HeapAccessValue::Found(b) => b,
                HeapAccessValue::SetupExecution => {
                    return Result::Ok(())
                }
            }
        };

        //pop off 0:, 1:, 2:
        try!(m.stack.pop()); 
        try!(m.stack.pop()); 
        try!(m.stack.pop()); 

        if cond {
            let (_, then_addr) = try!(unwrap_heap_node_to_ap(m.heap.get(&then_ap_addr)));
            let then_node = m.heap.get(&then_addr);
            m.heap.rewrite(&else_ap_addr, then_node);
        }
        else {
            let (_, else_addr) = try!(unwrap_heap_node_to_ap(m.heap.get(&else_ap_addr)));
            let else_node = m.heap.get(&else_addr);
            m.heap.rewrite(&else_ap_addr, else_node);
        }
        Result::Ok(())
    }

    //0: casePair
    //1: casePair $ (left, right)
    //2: casePair (left, right) $ <case_handler>
    //on rewrite
    //2: (<case_handler> $ left) $ right
    fn run_primitive_case_pair(m: &mut Machine) -> Result<(), MachineError> {
        let stack_copy = m.stack.clone();


        let pair_ap_addr = try!(m.stack
                                .iter()
                                .nth(1)
                                .cloned()
                                .ok_or("expected pair parameter, was not found on stack"));

        let case_handler_ap_addr = try!(m.stack
                                        .iter()
                                        .nth(2)
                                        .cloned()
                                        .ok_or("expected handler parameter, was not found on stack"));

        let (_, case_handler_addr) = try!(unwrap_heap_node_to_ap(m.heap.get(&case_handler_ap_addr)));
        let (left_addr, right_addr) =  
            match try!(setup_heap_node_access(m,
                                              stack_copy,
                                              pair_ap_addr,
                                              heap_try_pair_access)) {
                HeapAccessValue::Found(pair) => { 
                    pair
                },
                HeapAccessValue::SetupExecution => {
                    return Result::Ok(())
                }
            };


        //(f left) $ right

        //(f left)
        let ap_f_left = HeapNode::Application{
            fn_addr: case_handler_addr, 
            arg_addr: left_addr};

        let ap_f_left_addr = m.heap.alloc(ap_f_left);

        //(f left) $ right
        let ap_outer = HeapNode::Application{
            fn_addr: ap_f_left_addr,
            arg_addr: right_addr};

        //pop out 0:, 1:
        try!(m.stack.pop());
        try!(m.stack.pop());

        //rewrite 2: with value
        m.heap.rewrite(&case_handler_ap_addr, ap_outer);
        Result::Ok(())
    }

    //Calling Convention
    //==================
    //caseList <list-param> <nil-handler> <cons-handler>
    //
    //Reduction Rules
    //===============
    //
    //Rule 1. (for Cons)
    //0: caseList
    //1: caseList $ (Cons x xs)
    //2: caseList Cons x xs $ <nil-handler>
    //3: caseList Cons x xs $ <nil-handler> $ <cons-handler>
    //on rewrite
    //3: <cons-handler> x xs

    //Rule 2. (for Nil)
    //0: caseList
    //1: caseList $ (Nil)
    //2: caseList Nil $ <nil-handler>
    //3: caseList Nil $ <nil-handler> $ <cons-handler>
    //on rewrite
    //3: <nil-handler>
    fn run_primitive_case_list(m: &mut Machine) -> Result<(), MachineError> {
        let stack_copy = m.stack.clone();


        //caseList $ <list-param>
        let param_ap_addr = try!(m.stack
                                 .iter()
                                 .nth(1)
                                 .cloned()
                                 .ok_or("expected list parameter, was not found on stack"));

        let nil_handler_ap_addr = try!(m.stack
                                       .iter()
                                       .nth(2)
                                       .cloned()
                                       .ok_or("expected nil handler, was not found on stack"));

        let cons_handler_ap_addr = try!(m.stack
                                        .iter()
                                        .nth(3)
                                        .cloned()
                                        .ok_or("expected cons handler, was not found on stack"));


        let list_data =  
            match try!(setup_heap_node_access(m,
                                              stack_copy,
                                              param_ap_addr,
                                              heap_try_list_access)) {
                HeapAccessValue::Found(list_data) => list_data, 
                HeapAccessValue::SetupExecution => return Result::Ok(())

            };

        //pop off 0: 1:, 2:
        try!(m.stack.pop());
        try!(m.stack.pop());
        try!(m.stack.pop());

        match list_data {
            ListAccess::Nil => {
                let (_, nil_handler_addr) = try!(unwrap_heap_node_to_ap(m.heap.get(&nil_handler_ap_addr)));
                let nil_handler = m.heap.get(&nil_handler_addr);

                m.heap.rewrite(&cons_handler_ap_addr, nil_handler)
            } 
            ListAccess::Cons(x_addr, xs_addr) => {
                // (<cons_handler> $ x) $ xs
                let (_, cons_handler_addr) = try!(unwrap_heap_node_to_ap(m.heap.get(&cons_handler_ap_addr)));
                let ap_x = HeapNode::Application{fn_addr: cons_handler_addr,
                arg_addr: x_addr};
                let ap_x_addr = m.heap.alloc(ap_x);

                let ap_xs = HeapNode::Application{fn_addr: ap_x_addr,
                arg_addr: xs_addr};

                m.heap.rewrite(&cons_handler_ap_addr, ap_xs);
            }
        };

        Result::Ok(())
    }



//extractor should return an error if a node cannot have data
//extracted from. It should return None
//0: +
//1: (+ a)
//2: (+ a) b
//bottom-^
fn run_primitive_num_binop<F>(m: &mut Machine, handler: F) -> Result<(), MachineError> 
where F: Fn(i32, i32) -> HeapNode {

    let stack_copy = m.stack.clone();

    try!(m.stack.pop());


    let left_value = {
        //pop off left value
        let left_ap_addr = try!(m.stack.pop());
        match try!(setup_heap_node_access(m,
                                          stack_copy.clone(),
                                          left_ap_addr,
                                          heap_try_num_access)) {
            HeapAccessValue::Found(val) => val,
            HeapAccessValue::SetupExecution => return Result::Ok(())
        }
    };

    //do the same process for right argument
    //peek (+ a) b
    //we peek, since in the case where (+ a) b can be reduced,
    //we simply rewrite the node (+ a b) with the final value
    //(instead of creating a fresh node)
    let binop_ap_addr = try!(m.stack.peek());
    let right_value = 
        match try!(setup_heap_node_access(m, 
                                          stack_copy,
                                          binop_ap_addr,
                                          heap_try_num_access)) {
            HeapAccessValue::Found(val) => val,
            HeapAccessValue::SetupExecution => return Result::Ok(())
        };

    m.heap.rewrite(&binop_ap_addr, handler(left_value,
                                           right_value));

    Result::Ok(())
}

//NOTE: rewrite_addr will point to the address of the full constructor call.
//That way, when we dump the stack and use it to run something more complex,
//it will point to the correct location
//eg:

//I
//stack: if (<complex expr) 1 2
//dump: []

//II
//stack: <complex_expr>
//dump: if (<complex_expr> 1 2

//III
//stack:
//..
//...
//<complex_expr> <- rewrite rule
//dump: if (<complex_expr> 1 2

//IV
//stack: <rewritten complex expr>
//dump: if <rewritten complex expr> 1 2
fn run_constructor(m: &mut Machine,
                   tag: &DataTag,
                   arity: u32) -> Result<(), MachineError> {

    //pop out constructor
    let mut rewrite_addr = try!(m.stack.pop());

    if m.stack.len() < arity as usize {
        return Result::Err(format!("expected to have \
                                       {} arguments to {:#?} \
                                       constructor, found {}",
                                       arity, 
                                       tag,
                                       m.stack.len()));
    }

    let mut arg_addrs : Vec<Addr> = Vec::new();

    //This will be rewritten with the data
    //since the fn call would have been something like:
    //##top##
    //(Prim (Constructor tag arity))
    //(Prim (Constructor tag arity) $ a)
    //(Prim (Constructor tag arity) a $ b)
    //( Prim (Constructor tag arity) a b $ c) <- to rewrite
    //##bottom##

    for _ in 0..arity {
        let arg_ap_addr = try!(m.stack.pop());
        rewrite_addr = arg_ap_addr;
        let (_, arg_addr) = try!(unwrap_heap_node_to_ap(m.heap.get(&arg_ap_addr)));
        arg_addrs.push(arg_addr);

    };

    m.heap.rewrite(&rewrite_addr, 
                   HeapNode::Data{
                       component_addrs: arg_addrs,
                       tag: tag.clone()
                   });

    m.stack.push(rewrite_addr);
    Result::Ok(())
}

/// Runs the given supercombinator by instantiating it on top of the stack.
fn run_supercombinator(m: &mut Machine, sc_defn: &SupercombDefn) -> Result<(), MachineError> {

    //pop the supercombinator
    let sc_addr = try!(m.stack.pop());

    //the arguments are the stack
    //values below the supercombinator. There
    //are (n = arity of supercombinator) arguments
    let arg_addrs = {
        let mut addrs = Vec::new();
        for _ in 0..sc_defn.args.len() {
            addrs.push(try!(m.stack.pop()));
        }
        addrs
    };

    let env = try!(make_supercombinator_env(&sc_defn,
                                            &m.heap,
                                            &arg_addrs,
                                            &m.globals));

    let new_alloc_addr = try!(m.instantiate(sc_defn.body.clone(), &env));

    m.stack.push(new_alloc_addr);

    if m.options.update_heap_on_sc_eval {
        //if the function call was (f x y), the stack will be
        //f
        //f x
        //(f x) y  <- final address in arg_addrs
        //we need to rewrite this heap value
        let full_call_addr = {
            //if the SC has 0 parameters (a constant), then eval the SC
            //and replace the SC itself
            if sc_defn.args.len() == 0 {
                sc_addr
            }
            else {
                *arg_addrs.last()
                    .expect(concat!("arguments has no final value ",
                                    "even though the Supercombinator ",
                                    "has >= 1 parameter"))
            }
        };
        m.heap.rewrite(&full_call_addr, HeapNode::Indirection(new_alloc_addr));
    }

    Result::Ok(())
}


/// Make an environment for the execution of the Supercombinator.
/// let f a b c = <body>
///
/// if a function call of the form `(f x y z)` was made,
/// the stack will look like
/// ```haskell
/// ---top---
/// f
/// f $ x <- first parameter x
/// (f $ x) $ y <- second parameter y
/// ((f $ x) $ y) $ z <- third parameter z
/// --------
/// ```
fn make_supercombinator_env(sc_defn: &SupercombDefn,
                            heap: &Heap,
                            stack_args:&Vec<Addr>,
                            globals: &Environment) -> Result<Environment, MachineError> {

    assert!(stack_args.len() == sc_defn.args.len());

    let mut env = globals.clone();

    for (arg_name, application_addr) in
        sc_defn.args.iter().zip(stack_args.iter()) {

            let application = heap.get(application_addr);
            let (_, param_addr) = try!(unwrap_heap_node_to_ap(application));
            env.insert(arg_name.clone(), param_addr);

        }
    Result::Ok(env)
}



/// represents what happens when you try to access a heap node for a 
/// primitive run. Either you found the required heap node,
/// or you ask to setup execution since there is a frozen supercombinator
/// node or something else that needs to be evaluated
enum HeapAccessValue<T> {
    Found(T),
    SetupExecution
}

type HeapAccessResult<T> = Result<HeapAccessValue<T>, MachineError>;

/// get a heap node of the kind that handler wants to get,
/// otherwise setup the heap so that unevaluated code
/// is evaluated to get something of this type
/// TODO: check if we can change semantics so it does not need to take the
/// application node as the parameter that's a little awkward
fn setup_heap_node_access<F, T>(m: &mut Machine,
                                stack_to_dump: Stack,
                                ap_addr: Addr,
                                access_handler: F ) -> HeapAccessResult<T>
where F: Fn(HeapNode) -> Result<T, MachineError> {

    let (fn_addr, arg_addr) = try!(unwrap_heap_node_to_ap(m.heap.get(&ap_addr))); 
    let arg = m.heap.get(&arg_addr);

    //setup indirection
    if let HeapNode::Indirection(ind_addr) = arg {
        //rewrite the indirection node directly with the application node
        //application that does into the indirection address
        m.heap.rewrite(&ap_addr, 
                       HeapNode::Application {
                           fn_addr: fn_addr,
                           arg_addr: ind_addr
                       });
        return Result::Ok(HeapAccessValue::SetupExecution)
    };


    //it's not a data node, so this is something we need to still execute
    if !arg.is_data_node() {
        m.dump_stack(stack_to_dump);
        m.stack.push(arg_addr);
        return Result::Ok(HeapAccessValue::SetupExecution)
    }

    //give the node the access handler. it will either return the value
    //or fail to do so
    let access_result = try!(access_handler(arg));
    Result::Ok(HeapAccessValue::Found(access_result))
}


/// try to access the heap node as a number.
///
/// returns an error if the heap node is not a `Num` node
fn heap_try_num_access(h: HeapNode) -> Result<i32, MachineError> {
    match h {
        HeapNode::Num(i) => Result::Ok(i),
        other @ _ => Result::Err(format!(
                "expected number, found: {:#?}", other))
    }
}


/// try to access the heap node as a boolean
///
/// returns an error if the heap node is not data with `TagTrue` or `TagFalse`.
fn heap_try_bool_access(h: HeapNode) -> Result<bool, MachineError> {
    match h {
        HeapNode::Data{tag: DataTag::TagFalse, ..} => Result::Ok(false),
        HeapNode::Data{tag: DataTag::TagTrue, ..} => Result::Ok(true),
        other @ _ => Result::Err(format!(
                "expected true / false, found: {:#?}", other))
    }
}

/// try to access the heap node as a pair
fn heap_try_pair_access(h: HeapNode) -> Result<(Addr, Addr), MachineError> {
    match h {
        HeapNode::Data{tag: DataTag::TagPair, ref component_addrs} => {
            let left = try!(component_addrs.get(0).ok_or(format!(
                        "expected left component, of pair {:#?}, was not found", h)));
            let right = try!(component_addrs.get(1).ok_or(format!(
                        "expected right component, of pair {:#?}, was not found", h)));

            Result::Ok((*left, *right))
        }
        other @ _ => 
            Result::Err(format!(
                    "expected Pair tag, found: {:#?}", other))
    }
}

/// Represents the result of accessing a heap node as a list.
/// either a `Nil` is found, or a `Cons` of two addresses is found
enum ListAccess { Nil, Cons (Addr, Addr) }

/// try to access the heap node as list.
fn heap_try_list_access(h: HeapNode) -> Result<ListAccess, MachineError> {
    match h {
        HeapNode::Data{tag: DataTag::TagListNil,..} => {
            Result::Ok(ListAccess::Nil)
        }
        HeapNode::Data{tag: DataTag::TagListCons, ref component_addrs} => {
            let x_addr = try!(component_addrs.get(0).cloned().ok_or(format!(
                        "expected first component of list, found {:#?}", h)));
            let xs_addr = try!(component_addrs.get(1).cloned().ok_or(format!(
                        "expected second component of list, found {:#?}", h)));

            Result::Ok((ListAccess::Cons(x_addr, xs_addr)))
        }
        other @ _ => 
            Result::Err(format!(
                    "expected Pair tag, found: {:#?}", other))
    }
}

/// returns the pretty-printed version of the final heap node on top
/// of the stack.
///
/// ### Panics
/// this function panics if the machine is not in the final state
pub fn machine_get_final_val_string(m: &Machine) -> String {
    assert!(m.is_final_state());
    format_heap_node(&m.heap, &m.stack.peek().unwrap())
}



/// pretty print the given stack `s`
fn print_stack(heap: &Heap, s: &Stack) {
    if s.len() == 0 {

    }
    else {
        println!("{}", Black.underline().paint("## top ##"));

        let mut table = Table::new();

        for addr in s.iter() {
            let node = heap.get(addr);
            table.add_row(row![format_addr_string(addr),
            "->",
            format_heap_node(&heap, addr),
            format!("{:#?}", node)]);

        }
        table.set_format(*FORMAT_CLEAN);
        table.printstd();

    }
    print!("{}", Black.underline().paint("## bottom ##\n"));
}

/// pretty print the machine
pub fn print_machine(m: &Machine) {
    let mut cur_addrs : HashSet<Addr> = HashSet::new();
    for addr in m.stack.iter() {
        collect_addrs_from_heap_node(&m.heap, addr, &mut cur_addrs);

    }

    for s in m.dump.iter() {
        for addr in s.iter() {
            collect_addrs_from_heap_node(&m.heap, addr, &mut cur_addrs);
        }
    }

    println!("{}", Blue.paint(format!("Stack - {} items", m.stack.len())));
     print_stack(&m.heap, &m.stack);


    
    println!("{}", Blue.paint(format!("Heap - {} items", m.heap.len())));
    if cur_addrs.len() == 0 {
        println!("  Empty");
    }
    else {
        let mut table = Table::new();
        for addr in cur_addrs.iter() {

            let node = m.heap.get(addr);
            table.add_row(row![format_addr_string(addr),
            "->",
            format_heap_node(&m.heap, addr),
            format!("{:#?}", node)]);
        }
        table.set_format(*FORMAT_CLEAN);
        table.printstd();

    }

    println!("{}", Blue.paint("Dump"));
    if m.dump.len() == 0 {
        println!("  Empty");
    }
    else {
        for s in m.dump.iter() {
            print_stack(&m.heap, s);
            println!("{}", Black.paint("---"));
        }
    }

    println!("{}", Blue.paint(format!("Globals - {} items", m.globals.len())));
    let globals_in_use = {
        let mut globals = Vec::new();
        for (name, addr) in m.globals.iter() {
            if cur_addrs.contains(addr) {
                globals.push((name, addr));
            }
        }
        globals
    };
    if globals_in_use.len() == 0 {
        println!("None in Use");        
    }
    else {
        let mut table = Table::new();
        for &(name, addr) in globals_in_use.iter() {

            table.add_row(row![name,
                          "->",
                          format_addr_string(addr)]);
        }
        table.set_format(*FORMAT_CLEAN);
        table.printstd();

    }

    println!("{}", Red.bold().paint("===///==="));
}



