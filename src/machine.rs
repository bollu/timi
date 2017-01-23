extern crate ansi_term;
extern crate pad;

use std::fmt;
use std::collections::HashMap;
use std::collections::HashSet;

use std;
use std::fmt::Write;

use ir::*;
use frontend;


use self::ansi_term::Colour::{Blue, Red, Black};
use self::ansi_term::Style;
use self::pad::{PadStr, Alignment};


//primitive operations on the machine
#[derive(Clone, PartialEq, Eq)]
pub enum MachinePrimOp {
    Add,
    Sub,
    Mul,
    Div,
    Negate,
    G,
    GEQ,
    L,
    LEQ,
    EQ,
    NEQ,
    Construct {
        tag: DataTag,
        arity: u32
    },
    If
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
            &MachinePrimOp::Construct{ref tag, ref arity} => {
                write!(fmt, "Construct(tag:{:#?} | arity: {})", tag, arity)
            }
        }
    }
}


#[derive(Clone,PartialEq,Eq,Debug)]
pub enum DataTag {
    TagFalse = 0,
    TagTrue = 1,
    TagPair = 2,
}

fn raw_tag_to_data_tag (raw_tag: u32) -> Result<DataTag, MachineError> {
    match raw_tag {
        0 => Result::Ok(DataTag::TagFalse),
        1 => Result::Ok(DataTag::TagTrue),
        2 => Result::Ok(DataTag::TagPair),
        other @ _ => Result::Err(format!(
                                         "expected False(0), \
                                         True(1), or Pair(2). \
                                         found: {}",
                                         other))
    }
} 

//heap nodes
#[derive(Clone, PartialEq, Eq)]
pub enum HeapNode {
    Application {
        fn_addr: Addr,
        arg_addr: Addr
    },
    Supercombinator(SupercombDefn),
    Num(i32),
    Indirection(Addr),
    Primitive(MachinePrimOp),
    Data{tag: DataTag, component_addrs: Vec<Addr>}
}


impl fmt::Debug for HeapNode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {

        fn format_heap_tag(s: &str) -> String {
            format!("{}", Style::new().bold().paint(s))
        }

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

                try!(write!(fmt, "data: "));
                for addr in component_addrs.iter() {
                    try!(write!(fmt, "{} ", format_addr_string(addr)));
                }
                try!(write!(fmt, ")"));
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


fn format_heap_node(m: &Machine, env: &Bindings, node: &HeapNode) -> String {
    match node {
        &HeapNode::Indirection(addr) => format!("indirection({})", addr),
        &HeapNode::Num(num) => format!("{}", num),
        &HeapNode::Primitive(ref primop) => format!("{:#?}", primop),
        &HeapNode::Application{ref fn_addr, ref arg_addr} =>
        format!("({} $ {})",
                format_heap_node(m, env, &m.heap.get(fn_addr)),
                format_heap_node(m, env, &m.heap.get(arg_addr))),
        &HeapNode::Supercombinator(ref sc_defn) =>  {
            let mut sc_str = String::new();
            write!(&mut sc_str, "{}", sc_defn.name).unwrap();
            sc_str
        }
        &HeapNode::Data{ref tag, ref component_addrs} => {
            let mut data_str = String::new();
            data_str  += &format!("data-(tag:{:#?} components:{:#?})",
                                  tag,
                                  component_addrs);
            for c in component_addrs.iter() {
                data_str += 
                &format!("{}", format_heap_node(m, env, &m.heap.get(c)))
            }
            data_str
        }

    }
}

fn collect_addrs_from_heap_node(heap: &Heap, node: &HeapNode, collection: &mut HashSet<Addr>) {
    match node {
        &HeapNode::Indirection(ref addr) => {
            collection.insert(*addr);
            collect_addrs_from_heap_node(heap, &heap.get(addr), collection);
        }
        &HeapNode::Application{ref fn_addr, ref arg_addr} => {
            collection.insert(*fn_addr);
            collection.insert(*arg_addr);
            collect_addrs_from_heap_node(heap, &heap.get(fn_addr), collection);
            collect_addrs_from_heap_node(heap, &heap.get(arg_addr), collection);
        }
        &HeapNode::Data{ref component_addrs, ..} => {
            for addr in component_addrs.iter() {
                collection.insert(*addr);
                collect_addrs_from_heap_node(heap, &heap.get(addr), collection);
            }
        }
        &HeapNode::Supercombinator(_) | 
        &HeapNode::Primitive(..) |
        &HeapNode::Num(_) => {}
    };
}


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

//unsued for mark 1
// a dump is a vector of stacks
pub type Dump = Vec<Stack>;

//stack of addresses of nodes. "Spine"
#[derive(Clone,PartialEq,Eq,Debug)]
pub struct Stack {
    stack: Vec<Addr>
}

impl Stack {
    fn new() -> Stack {
        Stack {
            stack: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.stack.len()
    }

    fn push(&mut self, addr: Addr) {
        self.stack.push(addr)
    }

    fn pop(&mut self) -> Result<Addr, MachineError> {
        self.stack.pop().ok_or("top of stack is empty".to_string())
    }

    fn peek(&self) -> Result<Addr, MachineError> {
        self.stack.last().cloned().ok_or("top of stack is empty to peek".to_string())
    }

    fn iter(&self) -> std::iter::Rev<std::slice::Iter<Addr>> {
        self.stack.iter().rev()
    }

}

//maps names to addresses in the heap
pub type Bindings = HashMap<Name, Addr>;

//maps addresses to machine Nodes
#[derive(Clone)]
pub struct Heap {
    heap: HashMap<Addr, HeapNode>,
    next_addr: Addr
}

impl fmt::Debug for Heap {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut keyvals : Vec<(&Addr, &HeapNode)> = self.heap.iter().collect();
        keyvals.sort_by(|a, b| a.0.cmp(b.0));

        for &(key, val) in keyvals.iter().rev() {
            try!(write!(fmt, "\t{} => {:#?}\n", key, val));
        }

        return Result::Ok(())

    }
}

impl Heap {
    fn new()  -> Heap {
        Heap {
            heap: HashMap::new(),
            next_addr: 0
        }
    }

    //allocate the HeapNode on the heap
    fn alloc(&mut self, node: HeapNode) -> Addr {
        let addr = self.next_addr;
        self.next_addr += 1;

        self.heap.insert(addr, node);
        addr
    }

    fn get(&self, addr: &Addr) -> HeapNode {
        self.heap
        .get(&addr)
        .cloned()
        .expect(&format!("expected heap node at addess: {}", addr))
    }

    fn rewrite(&mut self, addr: &Addr, node: HeapNode) {
        assert!(self.heap.contains_key(addr),
                "asked to rewrite (address: {}) with \
                (node: {:#?}) which does not exist on heap",
                addr, node);
        self.heap.insert(*addr, node);
    }

}

//state of the machine
#[derive(Clone)]
pub struct MachineOptions {
    update_heap_on_sc_eval: bool,
}

#[derive(Clone)]
pub struct Machine {
    stack : Stack,
    heap : Heap,
    globals: Bindings,
    dump: Dump,
    options: MachineOptions,
}

type MachineError = String;



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


fn get_prelude() -> CoreProgram {
    frontend::string_to_program("I x = x;\
                                K x y = x;\
                                K1 x y = y;\
                                S f g x = f x (g x);\
                                compose f g x = f (g x);\
                                twice f = compose f f;\
                                False = Pack{0, 0};\
                                True = Pack{1, 0};\
                                Y f = f (Y f);\
                                facrec f n = if (n == 0) 1 (n * f (n - 1));\
                                fac n = (Y facrec) n\
                                ".to_string()).unwrap()
}

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
    ].iter().cloned().collect()
}


fn heap_build_initial(sc_defs: CoreProgram, prims: Vec<(Name, MachinePrimOp)>) 
-> (Heap, Bindings) {

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

// *** INTERPRETER ***
//interreter

impl Machine {
    pub fn new_minimal() -> Machine {
        let (initial_heap, globals) = heap_build_initial(get_prelude(),
                                                         get_primitives());

        Machine {
            dump: Vec::new(),
            //stack has addr main on top
            stack:  Stack::new(),
            globals: globals,
            heap: initial_heap,
            options: MachineOptions {
                update_heap_on_sc_eval: true
            }
        }

    }

    #[cfg(test)]
    pub fn new_with_main(program: CoreProgram) -> Machine {
        let mut m = Machine::new_minimal();
        for sc in program.into_iter() {
            m.add_supercombinator(sc);
        }

        //get main out of the heap
        let main_addr : Addr = match m.globals.get("main") {
            Some(main) => main,
            None => panic!("no main found")
        }.clone();

        m.stack.push(main_addr);
        m
    }

    pub fn add_supercombinator(&mut self, sc_defn: SupercombDefn) -> Addr{
        let name = sc_defn.name.clone();
        let node = HeapNode::Supercombinator(sc_defn);
        let addr = self.heap.alloc(node);

        //insert it into the globals, binding the name to the
        //heap address
        self.globals.insert(name, addr);
        addr
    }

    pub fn set_main_expr(&mut self, expr: &CoreExpr) {
        let main_defn = SupercombDefn {
            name: "main".to_string(),
            body: expr.clone(),
            args: Vec::new()
        };

        let main_addr = self.add_supercombinator(main_defn);
        self.stack = Stack { stack: vec![main_addr] };
    }

    //returns bindings of this run
    pub fn step(&mut self) -> Result<Bindings, MachineError>{
        //top of stack
        let tos_addr : Addr = try!(self.stack.peek());
        let heap_val = self.heap.get(&tos_addr);


        //there is something on the dump that wants to use this
        //data node, so pop it back.
        if heap_val.is_data_node() && self.dump.len() > 0 {
            self.stack = self.dump
            .pop()
            .expect("dump should have at least 1 element");
            Result::Ok(self.globals.clone())
        } else {
            self.heap_node_step(&heap_val)
        }
    }



    fn run_primitive_negate(&mut self) -> Result<(), MachineError> {
        //we need a copy of the stack to push into the dump
        let stack_copy = self.stack.clone();

        //pop the primitive off
        try!(self.stack.pop());

        //we rewrite this addres in case of
        //a raw number
        let neg_ap_addr = try!(self.stack.peek());

        //Apply <negprim> <argument>
        //look at what argument is and dispatch work
        let to_negate_val = 
        match try!(setup_heap_node_access(self,
                                          stack_copy, 
                                          neg_ap_addr,
                                          heap_try_num_access)) {
            HeapAccessValue::Found(val) => val,
            HeapAccessValue::SetupExecution => return Result::Ok(())
        };

        self.heap.rewrite(&neg_ap_addr, HeapNode::Num(-to_negate_val));
        Result::Ok(())
    }


    //extractor should return an error if a node cannot have data
    //extracted from. It should return None
    fn run_primitive_num_binop<F>(&mut self, handler: F) -> Result<(), MachineError> 
    where F: Fn(i32, i32) -> HeapNode {

        let stack_copy = self.stack.clone();

        //stack will be like

        //top--v
        //+
        //(+ a)
        //(+ a) b
        //bottom-^

        //fully eval a, b
        //then do stuff

        //pop off operator
        try!(self.stack.pop());


        let left_value = {
            //pop off left value
            let left_ap_addr = try!(self.stack.pop());
            match try!(setup_heap_node_access(self,
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
        let binop_ap_addr = try!(self.stack.peek());
        let right_value = 
        match try!(setup_heap_node_access(self, 
                                          stack_copy,
                                          binop_ap_addr,
                                          heap_try_num_access)) {
            HeapAccessValue::Found(val) => val,
            HeapAccessValue::SetupExecution => return Result::Ok(())
        };

        self.heap.rewrite(&binop_ap_addr, handler(left_value,
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
    fn run_constructor(&mut self,
                       tag: &DataTag,
                       arity: u32) -> Result<(), MachineError> {

        //pop out constructor
        let mut rewrite_addr = try!(self.stack.pop());

        if self.stack.len() < arity as usize {
            return Result::Err(format!("expected to have \
                                       {} arguments to {:#?} \
                                       constructor, found {}",
                                       arity, 
                                       tag,
                                       self.stack.len()));
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
            let arg_ap_addr = try!(self.stack.pop());
            rewrite_addr = arg_ap_addr;
            let (_, arg_addr) = try!(unwrap_heap_node_to_ap(self.heap.get(&arg_ap_addr)));
            arg_addrs.push(arg_addr);

        };

        self.heap.rewrite(&rewrite_addr, 
                          HeapNode::Data{
                              component_addrs: arg_addrs,
                              tag: tag.clone()
                          });
        //TODO: check if this is correct?
        self.stack.push(rewrite_addr);
        Result::Ok(())
    }
    fn dump_stack(&mut self, stack: Stack) {
        self.dump.push(stack);
        self.stack = Stack::new();
    }

    fn run_primitive_if(&mut self) -> Result<(), MachineError> {
        let stack_copy = self.stack.clone();

        //remove if condition
        try!(self.stack.pop());

        //## top of stack
        //if 
        //if $ <cond> <- if_ap_addr
        //if <cond> $ <then>
        //if <cond> <then> $ <else>
        //## bottom of stack
        let if_ap_addr = try!(self.stack.peek());
        println!("if_ap_addr: {}", if_ap_addr);

        let then_ap_addr = try!(self.stack.clone()
                                .iter()
                                .nth(1)
                                .ok_or("expected then application, was not found on stack".to_string())).clone();
        println!("then_ap_addr: {}", then_ap_addr);

        let else_ap_addr = try!(self.stack.clone()
                                .iter()
                                .nth(2)
                                .ok_or("expected else application, was not found on stack".to_string())).clone();
        println!("else_ap_addr: {}", else_ap_addr);

        let cond : bool = {
            println!("extracting cond addr...");
            match try!(setup_heap_node_access(self,
                                              stack_copy,
                                              if_ap_addr,
                                              heap_try_bool_access)) {
                HeapAccessValue::Found(b) => b,
                HeapAccessValue::SetupExecution => {
                    println!("setting up execution...");
                    return Result::Ok(())
                }
            }
        };

        println!("found cond: {}", cond);

        try!(self.stack.pop()); // if $ cond
        try!(self.stack.pop()); // if cond $ then

        if cond {
            let (_, then_addr) = try!(unwrap_heap_node_to_ap(self.heap.get(&then_ap_addr)));
            let then_node = self.heap.get(&then_addr);
            self.heap.rewrite(&else_ap_addr, then_node);
        }
        else {
            let (_, else_addr) = try!(unwrap_heap_node_to_ap(self.heap.get(&else_ap_addr)));
            let else_node = self.heap.get(&else_addr);
            self.heap.rewrite(&else_ap_addr, else_node);
        }
        Result::Ok(())
    }


    //actually step the computation
    fn heap_node_step(&mut self, heap_val: &HeapNode) -> Result<Bindings, MachineError> {
        match heap_val {
            &HeapNode::Num(n) =>
            return Result::Err(format!("number applied as a function: {}", n)),
            data @ &HeapNode::Data{..} => Result::Err(format!(
                                                              "data node applied as function: {:#?}", data)),
            &HeapNode::Application{fn_addr, ..} => {
                //push function address over the function
                self.stack.push(fn_addr);
                Result::Ok(self.globals.clone())
            }
            &HeapNode::Indirection(ref addr) => {
                //simply ignore an indirection during execution, and
                //push the indirected value on the stack
                try!(self.stack.pop());
                self.stack.push(*addr);
                Result::Ok(self.globals.clone())
            }
            &HeapNode::Primitive(MachinePrimOp::Negate) => {
                try!(self.run_primitive_negate());
                Result::Ok(self.globals.clone())
            }
            &HeapNode::Primitive(MachinePrimOp::Add) => {
                try!(self.run_primitive_num_binop(|x, y| HeapNode::Num(x + y)));
                Result::Ok(self.globals.clone())
            }
            &HeapNode::Primitive(MachinePrimOp::Sub) => {
                try!(self.run_primitive_num_binop(|x, y| HeapNode::Num(x - y)));
                Result::Ok(self.globals.clone())
            }
            &HeapNode::Primitive(MachinePrimOp::Mul) => {
                try!(self.run_primitive_num_binop(|x, y| HeapNode::Num(x * y)));
                Result::Ok(self.globals.clone())
            }
            &HeapNode::Primitive(MachinePrimOp::Div) => {
                try!(self.run_primitive_num_binop(|x, y| HeapNode::Num(x / y)));
                Result::Ok(self.globals.clone())
            }
            //construct a complex type
            &HeapNode::Primitive(MachinePrimOp::Construct {ref tag, arity}) => {
                try!(self.run_constructor(tag, arity));
                Result::Ok(self.globals.clone())
            }
            //boolean ops
            &HeapNode::Primitive(MachinePrimOp::G) => {
                try!(self.run_primitive_num_binop(
                                                  |x, y| bool_to_heap_node(x > y)));
                Result::Ok(self.globals.clone())
            }
            &HeapNode::Primitive(MachinePrimOp::GEQ) => {
                try!(self.run_primitive_num_binop(
                                                  |x, y| bool_to_heap_node(x >= y)));
                Result::Ok(self.globals.clone())
            }
            &HeapNode::Primitive(MachinePrimOp::L) => {
                try!(self.run_primitive_num_binop(
                                                  |x, y| bool_to_heap_node(x < y)));
                Result::Ok(self.globals.clone())
            }
            &HeapNode::Primitive(MachinePrimOp::LEQ) => {
                try!(self.run_primitive_num_binop(
                                                  |x, y| bool_to_heap_node(x <= y)));
                Result::Ok(self.globals.clone())
            }
            &HeapNode::Primitive(MachinePrimOp::EQ) => {
                try!(self.run_primitive_num_binop(
                                                  |x, y| bool_to_heap_node(x == y)));
                Result::Ok(self.globals.clone())
            }
            &HeapNode::Primitive(MachinePrimOp::NEQ) => {
                try!(self.run_primitive_num_binop(
                                                  |x, y| bool_to_heap_node(x != y)));
                Result::Ok(self.globals.clone())
            }
            //run if condition
            &HeapNode::Primitive(MachinePrimOp::If) => {
                try!(self.run_primitive_if());
                Result::Ok(self.globals.clone())
            }
            //expand supercombinator
            &HeapNode::Supercombinator(ref sc_defn) => {
                run_supercombinator(self, sc_defn)

            }
        }
    }


    fn instantiate(&mut self, expr: CoreExpr, env: &Bindings) -> Result<Addr, MachineError> {
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

fn instantiate_let_bindings(m: &mut Machine,
                            orig_env: &Bindings,
                            bindings: Vec<(Name, Box<CoreExpr>)>) 
-> Result<Bindings, MachineError> {

    let mut env : Bindings = orig_env.clone();

    for (&(ref name, _), addr) in bindings.iter().zip(1..(bindings.len()+1))  {
        env.insert(name.clone(), -(addr as i32));
    }

    let mut old_to_new_addr: HashMap<Addr, Addr> = HashMap::new();

    //instantiate RHS, while storing legit LHS addresses
    for (bind_name, bind_expr) in bindings.into_iter() {
        let new_addr = try!(m.instantiate(*bind_expr.clone(), &env));
        let old_addr = try!(env.get(&bind_name)
                            .ok_or(format!("unable to find |{}| in env", bind_name)))
        .clone();

        old_to_new_addr.insert(old_addr, new_addr);

        //insert the "correct" address into the let environment
        env.insert(bind_name.clone(), new_addr);
    }


    for (old, new) in old_to_new_addr.iter() {
        for to_edit in old_to_new_addr.values() {
            change_addr_in_heap_node(*old,
                                     *new,
                                     *to_edit,
                                     &mut m.heap);
        }

    }

    Result::Ok(env)

}


fn change_addr_in_heap_node(old_addr: Addr,
                            new_addr: Addr,
                            edit_addr: Addr,
                            mut heap: &mut Heap) {

    match heap.get(&edit_addr) {
        HeapNode::Data{..} => panic!("unimplemented rebinding of Data node"),
        HeapNode::Application{fn_addr, arg_addr} => {
            let new_fn_addr = if fn_addr == old_addr {
                new_addr
            } else {
                fn_addr
            };


            let new_arg_addr = if arg_addr == old_addr {
                new_addr
            } else {
                arg_addr
            };

            //if we have not replaced, then recurse
            //into the application calls
            if fn_addr != old_addr {
                change_addr_in_heap_node(old_addr,
                                         new_addr,
                                         fn_addr,
                                         &mut heap);

            };

            if arg_addr != old_addr {
                change_addr_in_heap_node(old_addr,
                                         new_addr,
                                         arg_addr,
                                         &mut heap);
            };

            heap.rewrite(&edit_addr,
                         HeapNode::Application{
                           fn_addr: new_fn_addr,
                           arg_addr: new_arg_addr
                       });

        },
        HeapNode::Indirection(ref addr) =>
        change_addr_in_heap_node(old_addr,
                                 new_addr,
                                 *addr,
                                 &mut heap),

        HeapNode::Primitive(_) => {}
        HeapNode::Supercombinator(_) => {}
        HeapNode::Num(_) => {},
    }
}


fn run_supercombinator(m: &mut Machine, sc_defn: &SupercombDefn) -> Result<Bindings, MachineError> {

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
                                "even though the supercombinator ",
                                "has >= 1 parameter"))
            }
        };
        m.heap.rewrite(&full_call_addr, HeapNode::Indirection(new_alloc_addr));
    }
    Result::Ok(env)
}



//make an environment for the execution of the supercombinator
fn make_supercombinator_env(sc_defn: &SupercombDefn,
                            heap: &Heap,
                            stack_args:&Vec<Addr>,
                            globals: &Bindings) -> Result<Bindings, MachineError> {

    assert!(stack_args.len() == sc_defn.args.len());

    let mut env = globals.clone();

    /*
     * let f a b c = <body>
     *
     * if a function call of the form f x y z was made,
     * the stack will look like
     * ---top---
     * f
     * f x
     * f x y
     * f x y z
     * --------
     *
     * the "f" will be popped beforehand (that is part of the contract
     * of calling make_supercombinator_env)
     *
     *
     * So, we go down the stack, removing function applications, and
     * binding the RHS to the function parameter names.
     *
     */
     for (arg_name, application_addr) in
     sc_defn.args.iter().zip(stack_args.iter()) {

        let application = heap.get(application_addr);
        let (_, param_addr) = try!(unwrap_heap_node_to_ap(application));
        env.insert(arg_name.clone(), param_addr);

    }
    Result::Ok(env)
}



//represents what happens when you try to access a heap node for a 
//primitive run. Either you found the required heap node,
//or you ask to setup execution since there is a frozen supercombinator
//node or something else that needs to be evaluated
enum HeapAccessValue<T> {
    Found(T),
    SetupExecution
}

type HeapAccessResult<T> = Result<HeapAccessValue<T>, MachineError>;

//get a heap node of the kind that handler wants to get,
//otherwise setup the heap so that unevaluated code
//is evaluated to get something of this type
//TODO: check if we can change semantics so it does not need to take the
//application node as the parameter that's a little awkward
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

fn heap_try_num_access(h: HeapNode) -> Result<i32, MachineError> {
    match h {
        HeapNode::Num(i) => Result::Ok(i),
        other @ _ => Result::Err(format!(
                                         "expected number, found: {:#?}", other))
    }
}


fn heap_try_bool_access(h: HeapNode) -> Result<bool, MachineError> {
    match h {
        //TODO: make a separate function that takes HeapNode::Data
        //and returns the correct rust boolean
        HeapNode::Data{tag: DataTag::TagFalse, ..} => Result::Ok(false),
        HeapNode::Data{tag: DataTag::TagTrue, ..} => Result::Ok(true),
        other @ _ => Result::Err(format!(
                                         "expected true / false, found: {:#?}", other))
    }
}


pub fn machine_is_final_state(m: &Machine) -> bool {
    assert!(m.stack.len() > 0, "expect stack to have at least 1 node");

    if m.stack.len() > 1 {
        false
    } else {
        let dump_empty = m.dump.len() == 0;
        m.heap.get(&m.stack.peek().unwrap()).is_data_node() &&
        dump_empty
    }
}

pub fn machine_get_final_val(m: &Machine) -> HeapNode {
    assert!(machine_is_final_state(m));
    m.heap.get(&m.stack.peek().unwrap())
}



fn print_stack(m: &Machine, env: &Bindings, s: &Stack) {
    println!( "{}", Blue.paint("Stack"));
    if s.len() == 0 {

    }
    else {
        println!("  {}", Black.underline().paint("## top ##"));

        for addr in s.iter() {
            println!("  {} -> {} | {:#?}",
                  format_addr_string(addr).pad_to_width_with_alignment(20, Alignment::Right),
                  format_heap_node(m, env, &m.heap.get(addr)).pad_to_width_with_alignment(30, Alignment::Left),
                  &m.heap.get(addr));
        }
    }
    print!("  {}", Black.underline().paint("## bottom ##\n"));
}

pub fn print_machine_stack(m: &Machine, env: &Bindings) {
    print_stack(m, env, &m.stack);
}


pub fn print_machine(m: &Machine, env: &Bindings) {

    let mut addr_collection = HashSet::new();
    for addr in m.stack.iter() {
        collect_addrs_from_heap_node(&m.heap, &m.heap.get(addr), &mut addr_collection);

    }

    for s in m.dump.iter() {
        for addr in s.iter() {
            collect_addrs_from_heap_node(&m.heap, &m.heap.get(addr), &mut addr_collection);
        }
    }

    print_stack(m, env, &m.stack);
    println!("{}", Blue.paint("Heap"));

    if addr_collection.len() == 0 {
        println!("  Empty");
    }
    else {
        for addr in addr_collection.iter() {
                println!("  {} -> {} {} {:#?}", 
                     format_addr_string(addr).pad_to_width_with_alignment(20, Alignment::Right),
                     format_heap_node(m, env, &m.heap.get(addr)).pad_to_width_with_alignment(30, Alignment::Left),
                     Style::new().bold().paint("|"),
                     m.heap.get(addr));


        }
    }
    println!("{}", Red.bold().paint("===///==="));
}


#[cfg(test)]
fn run_machine(program:  &str) -> Machine {
    use frontend::string_to_program;

    let main = string_to_program(program.to_string())
    .unwrap();
    let mut m = Machine::new_with_main(main);
    while !machine_is_final_state(&m) {
        let _ = m.step();
    }
    return m
}

#[test]
fn test_skk3() {
    let m = run_machine("main = S K K 3");
    assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(3));
}

#[test]
fn test_negate_simple() {
    let m = run_machine("main = negate 1");
    assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(-1));
}

#[test]
fn test_negate_inner_ap() {
    let m = run_machine("main = negate (negate 1)");
    assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(1));
}


#[test]
fn test_add_simple() {
    let m = run_machine("main = 1 + 1");
    assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(2));
}

#[test]
fn test_add_lhs_ap() {
    let m = run_machine("main = (negate 1) + 1");
    assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(0));
}


#[test]
fn test_add_rhs_ap() {
    let m = run_machine("main = 1 + (negate 3)");
    assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(-2));
}

#[test]
fn test_add_lhs_rhs_ap() {
    let m = run_machine("main = (negate 1) + (negate 3)");
    assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(-4));
}

#[test]
fn test_complex_arith() {
    let m = run_machine("main = 1 * 2 + 10 * 20 + 30 / 3");
    assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(212));
}

#[test]
fn test_if_true_branch() {
    let m = run_machine("main = if True 1 2");
    assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(1));
}


#[test]
fn test_if_false_branch() {
    let m = run_machine("main = if False 1 2");
    assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(2));
}

#[test]
fn test_if_cond_complex_branch() {
    let mut m = run_machine("main = if (1 < 2) 1 2");
    assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(1));

    m = run_machine("main = if (1 > 2) 1 2");
    assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(2));
}

#[test]
fn test_if_cond_complex_result() {
    let mut m = run_machine("main = if True (100 + 100) (100 - 100)");
    assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(200));

    m = run_machine("main = if False (100 + 100) (100 - 100)");
    assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(0));
}


