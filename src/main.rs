#[macro_use]
extern crate nom;

use std::collections::HashMap;
use std::fmt;
use std::fmt::{Write};

use std::cmp; //for max

type Addr = i32;
type Name = String;

type CoreVariable = Name;

#[derive(Clone, PartialEq, Eq, Debug)]
struct CoreLet {
    is_rec: bool,
    bindings: Vec<(Name, Box<CoreExpr>)>,
    expr: Box<CoreExpr>
}



#[derive(Clone, PartialEq, Eq)]
enum CoreExpr {
    //change this?
    Variable(Name),
    Num(i32),
    Application(Box<CoreExpr>, Box<CoreExpr>),
    Let(CoreLet)
}

impl fmt::Debug for CoreExpr {

    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &CoreExpr::Variable(ref name) => write!(fmt, "{}", name),
            &CoreExpr::Num(ref num) => write!(fmt, "n_{}", num),
            &CoreExpr::Application(ref e1, ref e2) =>
                write!(fmt, "({:#?} $ {:#?})", *e1, *e2),
            &CoreExpr::Let(CoreLet{ref is_rec, ref bindings, ref expr}) => {
                if *is_rec {
                    try!(write!(fmt, "letrec"));
                } else {
                    try!(write!(fmt, "let"));
                }
                try!(write!(fmt, " {{\n"));
                for &(ref name, ref expr) in bindings {
                    try!(write!(fmt, "{} = {:#?}\n", name, expr));
                }
                try!(write!(fmt, "in\n"));
                try!(write!(fmt, "{:#?}", expr));
                write!(fmt, "}}")
            }
        }
    }
}


#[derive(Clone, PartialEq, Eq)]
struct SupercombDefn {
    name: String,
    args: Vec<String>,
    body: CoreExpr
}


impl fmt::Debug for SupercombDefn {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(fmt, "{} ", &self.name));
        for arg in self.args.iter() {
            try!(write!(fmt, "{} ", &arg));
        }
        try!(write!(fmt, "{{ {:#?} }}", self.body));
        Result::Ok(())

    }

}


//a core program is a list of supercombinator
//definitions
type CoreProgram = Vec<SupercombDefn>;

//primitive operations on the machine
#[derive(Clone, PartialEq, Eq, Debug)]
enum MachinePrimOp {
    Add,
    Sub,
    Mul,
    Div,
    Negate,
}


//heap nodes
#[derive(Clone, PartialEq, Eq)]
enum HeapNode {
    Application {
        fn_addr: Addr,
        arg_addr: Addr
    },
    Supercombinator(SupercombDefn),
    Num(i32),
    Indirection(Addr),
    Primitive(Name, MachinePrimOp)
}

impl fmt::Debug for HeapNode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &HeapNode::Application{ref fn_addr, ref arg_addr} => {
                write!(fmt, "H-({} $ {})", fn_addr, arg_addr)
            }
            &HeapNode::Supercombinator(ref sc_defn) => {
                write!(fmt, "H-{:#?}", sc_defn)
            },
            &HeapNode::Num(ref num)  => {
                write!(fmt, "H-{}", num)
            }
            &HeapNode::Indirection(ref addr)  => {
                write!(fmt, "H-indirection-{}", addr)
            }
            &HeapNode::Primitive(ref name, ref primop)  => {
                write!(fmt, "H-prim-{} {:#?}", name, primop)
            }
        }
    }
}

impl HeapNode {
    fn is_data_node(&self) -> bool {
        match self {
            &HeapNode::Num(_) => true,
            _ => false
        }
    }
}


//unsued for mark 1
// a dump is a vector of stacks
type Dump = Vec<Stack>;

//stack of addresses of nodes. "Spine"
#[derive(Clone,PartialEq,Eq,Debug)]
struct Stack {
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

    fn pop(&mut self) -> Addr {
        self.stack.pop().expect("top of stack is empty")
    }

    fn peek(&self) -> Addr {
        self.stack.last().expect("top of stack is empty to peek").clone()
    }

    fn iter(&self) -> std::slice::Iter<Addr> {
        self.stack.iter()
    }

}

//maps names to addresses in the heap
type Bindings = HashMap<Name, Addr>;

//maps addresses to machine Nodes
#[derive(Clone)]
struct Heap {
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
            "asked to rewrite (address: {}) with (node: {:#?}) which does not exist on heap",
            addr, node);
        self.heap.insert(*addr, node);
    }

}

//state of the machine
#[derive(Clone)]
struct MachineOptions {
    update_heap_on_sc_eval: bool,
}

#[derive(Clone)]
struct Machine {
    stack : Stack,
    heap : Heap,
    globals: Bindings,
    dump: Dump,
    options: MachineOptions,
}



fn format_heap_node(m: &Machine, env: &Bindings, node: &HeapNode) -> String {
    match node {
        &HeapNode::Indirection(addr) => format!("indirection: {}", addr),
        &HeapNode::Num(num) => format!("{}", num),
        &HeapNode::Primitive(ref name,
                                     ref primop) => format!("prim-{}-{:#?}", name, primop),
        &HeapNode::Application{ref fn_addr, ref arg_addr} =>
            format!("({} $ {})",
                    format_heap_node(m, env, &m.heap.get(fn_addr)),
                    format_heap_node(m, env, &m.heap.get(arg_addr))),
        &HeapNode::Supercombinator(ref sc_defn) =>  {
            let mut sc_str = String::new();
            write!(&mut sc_str, "{}", sc_defn.name).unwrap();
            sc_str

        }
    }
}

fn print_machine(m: &Machine, env: &Bindings) {
    print!("\n\n===\n\n");

    print!( "*** stack: ***\n");
    print!( "## top ##\n");
    for addr in m.stack.iter().rev() {
        print!("heap[{}] :  {}\n",
               *addr,
               format_heap_node(m,
                                env,
                                &m.heap.get(addr)));
    }
    print!( "## bottom ##\n");


    print!("*** heap: ***\n");
    print!("{:#?}", m.heap);

    print!( "\n*** env: ***\n");
    let mut env_ordered : Vec<(&Name, &Addr)> = env.iter().collect();
    env_ordered.sort_by(|e1, e2| e1.0.cmp(e2.0));
    for &(name, addr) in env_ordered.iter() {
        print!("{} => {}\n", name, format_heap_node(m, env, &m.heap.get(addr)));
    }
}



fn get_prelude() -> CoreProgram {
    string_to_program("I x = x;\
                       K x y = x;\
                       K1 x y = y;\
                       S f g x = f x (g x);\
                       compose f g x = f (g x);\
                       twice f = compose f f\
                       ".to_string()).unwrap()
}

fn get_primitives() -> Vec<(Name, MachinePrimOp)> {
    [("+".to_string(), MachinePrimOp::Add),
     ("-".to_string(), MachinePrimOp::Sub),
     ("*".to_string(), MachinePrimOp::Mul),
     ("/".to_string(), MachinePrimOp::Div),
     ("negate".to_string(), MachinePrimOp::Negate),
    ].iter().cloned().collect()
}

fn heap_build_initial(sc_defs: CoreProgram, prims: Vec<(Name, MachinePrimOp)>) -> (Heap, Bindings) {
    let mut heap = Heap::new();
    let mut globals = HashMap::new();

    for sc_def in sc_defs.iter() {
        //create a heap node for the supercombinator definition
        //and insert it
        let node = HeapNode::Supercombinator(sc_def.clone());
        let addr = heap.alloc(node);

        //insert it into the globals, binding the name to the
        //heap address
        globals.insert(sc_def.name.clone(), addr);
    }

    for (name, prim_op) in prims.into_iter() {
        let addr = heap.alloc(HeapNode::Primitive(name.clone(),
                                                          prim_op));
        globals.insert(name, addr);
    }

    (heap, globals)
}


//interreter

impl Machine {
    fn new(program: CoreProgram) -> Machine {
        //all supercombinators = program + prelude
        let mut sc_defs = program.clone();
        sc_defs.extend(get_prelude().iter().cloned());

        let (initial_heap, globals) = heap_build_initial(sc_defs,
                                                         get_primitives());

        //get main out of the heap
        let main_addr : Addr = match globals.get("main") {
            Some(main) => main,
            None => panic!("no main found")
        }.clone();

        Machine {
            dump: Vec::new(),
            //stack has addr main on top
            stack:  {
                let mut s = Stack::new();
                s.push(main_addr);
                s
            },
            globals: globals,
            heap: initial_heap,
            options: MachineOptions {
                update_heap_on_sc_eval: true
            }
        }
    }

    //returns bindings of this run
    fn step(&mut self) -> Bindings {
        //top of stack
        let tos_addr : Addr = self.stack.peek();
        let heap_val = self.heap.get(&tos_addr);


        //there is something on the dump that wants to use this
        //data node, so pop it back.
        if heap_val.is_data_node() && self.dump.len() > 0 {
            self.stack = self.dump.pop().unwrap();
            self.globals.clone()
        } else {
            self.run_step(&heap_val)
        }
    }

    //make an environment for the execution of the supercombinator
    fn make_supercombinator_env(sc_defn: &SupercombDefn,
                        heap: &Heap,
                        stack_args:&Vec<Addr>,
                        globals: &Bindings) -> Bindings {

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
                let param_addr = match application {
                    HeapNode::Application{arg_addr, ..} => arg_addr,
                    _ => panic!(concat!("did not find application node when ",
                                        "unwinding stack for supercombinator"))
                };
            env.insert(arg_name.clone(), param_addr);

        }
        env
    }

    fn run_primitive_negate(&mut self) -> Bindings {
        //we need a copy of the stack to push into the dump
        let stack_copy = self.stack.clone();

        //pop the primitive off
        let negate_prim_addr = self.stack.pop();

        //we rewrite this addres in case of
        //a raw number
        let neg_ap_addr = self.stack.peek();

        //Apply <negprim> <argument>
        //look at what argument is and dispatch work
        if let HeapNode::Application{ref arg_addr, ref fn_addr} = self.heap.get(&neg_ap_addr) {
            let arg = self.heap.get(arg_addr);
            assert!(*fn_addr == negate_prim_addr, concat!("expected the function being called to be",
                                                         "the negate primitive"));

            match arg {
                HeapNode::Num(n) => {
                    //rewrite the function application (current thing)
                    self.heap.rewrite(&neg_ap_addr, HeapNode::Num(-n))
                }
                HeapNode::Indirection(ind_addr) => {
                    //pop off the application and then create a new
                    //application that does into the indirection address
                    self.heap.rewrite(&neg_ap_addr, HeapNode::Application{fn_addr: *fn_addr,
                                                                          arg_addr: ind_addr});

                }
                other @ _ => {
                    self.dump_stack(stack_copy);
                    self.stack.push(*arg_addr);
                }

            }

            self.globals.clone()

        }
        else {
            panic!("expected application node")
        }
    }



    fn run_primitive_arith_binop(&mut self, handler: fn (a: i32, b: i32) -> HeapNode) {
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
        self.stack.pop();


        //pop out (Ap (Prim +) a)
        let left_arg_addr = if let HeapNode::Application{arg_addr: left_arg_addr, ..} = self.heap.get(&self.stack.pop()) {
            left_arg_addr
        }
        else {
            panic!("expected function application of the form (+ a)");
        };

        match self.heap.get(&left_arg_addr) {
            HeapNode::Num(n_left) => {
                //do the same process for right argument
                //peek (+ a) b
                //we peek, since in the case where (+ a) b can be reduced,
                //we simply rewrite the node (+ a b) with the final value (instead of creating a fresh node)
                let binop_ap_addr = self.stack.peek();
                let binop_arg_addr = if let HeapNode::Application{arg_addr: right_arg_addr, ..} = self.heap.get(&binop_ap_addr) {
                    right_arg_addr
                }
                else {
                    panic!("expected function application of the form ((+ a) b)");
                };

                match self.heap.get(&binop_arg_addr) {
                    HeapNode::Num(n_right) => {
                        self.heap.rewrite(&binop_ap_addr, handler(n_left, n_right));
                    }
                    HeapNode::Indirection(ind_addr) => {
                        panic!("not handling indirection")
                    }
                    other @ _ => {
                        self.dump_stack(stack_copy);
                        self.stack.push(binop_arg_addr);
                    }

                }
            }
            HeapNode::Indirection(ind_addr) => {
                panic!("not handling indirection")
            }
            other @ _ => {
                self.dump_stack(stack_copy);
                self.stack.push(left_arg_addr);
            }

        } //close left_arg_addr pattern match

    } //close fn

    fn dump_stack(&mut self, stack: Stack) {
        self.dump.push(stack);
        self.stack = Stack::new();
    }

    //actually run_step the computation
    fn run_step(&mut self, heap_val: &HeapNode) -> Bindings {
        match heap_val {
            &HeapNode::Num(n) =>
                panic!("number applied as a function: {}", n),

            &HeapNode::Application{fn_addr, ..} => {
                //push function address over the function
                self.stack.push(fn_addr);
                self.globals.clone()
            }
            &HeapNode::Indirection(ref addr) => {
                //simply ignore an indirection during execution, and
                //push the indirected value on the stack
                self.stack.pop();
                self.stack.push(*addr);
                self.globals.clone()
            }
            &HeapNode::Primitive(ref name, ref prim) => {
                fn plus_handler(a: i32, b: i32) -> HeapNode {
                     HeapNode::Num(a + b)
                };

                match prim {
                    &MachinePrimOp::Negate => self.run_primitive_negate(),
                    &MachinePrimOp::Add => {
                        self.run_primitive_arith_binop(plus_handler);
                        self.globals.clone()
                    }
                    _ => panic!("unimplemented")
                }


            }
            &HeapNode::Supercombinator(ref sc_defn) => {

                //pop the supercombinator
                let sc_addr = self.stack.pop();

                //the arguments are the stack
                //values below the supercombinator. There
                //are (n = arity of supercombinator) arguments
                let arg_addrs = {
                    let mut addrs = Vec::new();
                    for _ in 0..sc_defn.args.len() {
                        addrs.push(self.stack.pop());
                    }
                    addrs
                };

                let env = Machine::make_supercombinator_env(&sc_defn,
                                                    &self.heap,
                                                    &arg_addrs,
                                                    &self.globals);

                let new_alloc_addr = self.instantiate(sc_defn.body.clone(), &env);

                self.stack.push(new_alloc_addr);

                if self.options.update_heap_on_sc_eval {
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
                    self.heap.rewrite(&full_call_addr, HeapNode::Indirection(new_alloc_addr));
                }
                env
            }
        }
    }

    fn rebind_vars_to_env(old_addr: Addr,
                          new_addr: Addr,
                          edit_addr: Addr,
                          mut heap: &mut Heap) {

        match heap.get(&edit_addr) {
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
                    Machine::rebind_vars_to_env(old_addr,
                                                new_addr,
                                                fn_addr,
                                                &mut heap);

                };

                if arg_addr != old_addr {
                    Machine::rebind_vars_to_env(old_addr,
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
                Machine::rebind_vars_to_env(old_addr,
                                   new_addr,
                                   *addr,
                                   &mut heap),

            HeapNode::Primitive(_, _) => {}
            HeapNode::Supercombinator(_) => {}
            HeapNode::Num(_) => {},
        }

    }

    fn instantiate(&mut self, expr: CoreExpr, env: &Bindings) -> Addr {
        match expr {
             CoreExpr::Let(CoreLet{expr: let_rhs, bindings, is_rec}) => {
                let mut let_env : Bindings = env.clone();

                if is_rec {
                    //TODO: change this to zip() with range

                    let mut addr = -1;
                    //first create dummy indeces for all LHS
                    for &(ref bind_name, _) in bindings.iter()  {
                        let_env.insert(bind_name.clone(), addr);
                        addr -= 1;
                    }

                    let mut old_to_new_addr: HashMap<Addr, Addr> = HashMap::new();

                    //instantiate RHS, while storing legit
                    //LHS addresses
                    //TODO: cleanup, check if into_iter is sufficient
                    for &(ref bind_name, ref bind_expr) in bindings.iter() {
                        let new_addr = self.instantiate(*bind_expr.clone(), &let_env);

                        let old_addr = let_env
                                        .get(bind_name)
                                        .expect(&format!("unable to find |{}| in env", bind_name))
                                        .clone();

                        old_to_new_addr.insert(old_addr, new_addr);

                        //insert the "correct" address into the
                        //let env
                        let_env.insert(bind_name.clone(), new_addr);

                    }

                    for (old, new) in old_to_new_addr.iter() {
                        for to_edit_addr in old_to_new_addr.values() {
                            Machine::rebind_vars_to_env(*old,
                                                        *new,
                                                        *to_edit_addr,
                                                        &mut self.heap);
                        }

                    }

                    print!("letrec env:\n {:#?}", let_env);
                    self.instantiate(*let_rhs, &let_env)

                }
                else {
                    for (bind_name, bind_expr) in bindings.into_iter() {
                        let addr = self.instantiate(*bind_expr, &let_env);
                        let_env.insert(bind_name.clone(), addr);
                    }
                    self.instantiate(*let_rhs, &let_env)

                }

            }
            CoreExpr::Num(x) => self.heap.alloc(HeapNode::Num(x)),
            CoreExpr::Application(fn_expr, arg_expr) => {
                let fn_addr = self.instantiate(*fn_expr, env);
                let arg_addr = self.instantiate(*arg_expr, env);

                self.heap.alloc(HeapNode::Application {
                                    fn_addr: fn_addr,
                                    arg_addr: arg_addr
                })

            },
            CoreExpr::Variable(vname) => {
                match env.get(&vname) {
                    Some(addr) => *addr,
                    None => panic!("unable to find variable in heap: {:?}", vname)
                }

            }
        }
    }

}


fn machine_is_final_state(m: &Machine) -> bool {
    assert!(m.stack.len() > 0, "expect stack to have at least 1 node");

    if m.stack.len() > 1 {
        false
    } else {
        let dump_empty = m.dump.len() == 0;
        m.heap.get(&m.stack.peek()).is_data_node() &&
            dump_empty
    }

}

//parsing ---

#[derive(Clone, Debug)]
enum ParseError {
    NoTokens,
    UnknownSymbol,
    UnexpectedToken {
        expected: Vec<CoreToken>,
        found: CoreToken
    },
    ParseErrorStr(String),

}

#[derive(Clone, PartialEq, Eq, Debug)]
enum CoreToken {
    Let,
    LetRec,
    In,
    Case,
    Ident(String),
    Equals,
    Semicolon,
    OpenRoundBracket,
    CloseRoundBracket,
    Integer(String),
    Lambda,
    Or,
    And,
    L,
    LEQ,
    G,
    GEQ,
    Plus,
    Minus,
    Mul,
    Div,
    //when you call peek(), it returns this token
    //if the token stream is empty.
    PeekNoToken
}

#[derive(Clone)]
struct ParserCursor {
    tokens: Vec<CoreToken>,
    pos: usize,
}

impl ParserCursor {
    fn new(tokens: Vec<CoreToken>) -> ParserCursor {
        ParserCursor {
            tokens: tokens,
            pos: 0
        }
    }

    fn peek(&self) -> CoreToken {
        match self.tokens.get(self.pos)
            .cloned() {
                Some(tok) => tok,
                None => CoreToken::PeekNoToken
            }

    }

    fn consume(&mut self) -> Result<CoreToken, ParseError> {
        match self.peek() {
            CoreToken::PeekNoToken => Result::Err(ParseError::NoTokens),
            other @ _ => {
                self.pos += 1;
                Result::Ok(other)
            }
        }

    }

    fn expect(&mut self, t: CoreToken) -> Result<(), ParseError> {
        let tok = self.peek();

        if tok == t {
            try!(self.consume());
            Result::Ok(())
        } else {
            Result::Err(ParseError::UnexpectedToken{
                expected: vec![t],
                found: tok
            })
        }
    }
}

fn identifier_str_to_token(token_str: &str) -> CoreToken {
    match token_str {
        "let" => CoreToken::Let,
        "letrec" => CoreToken::LetRec,
        "in" => CoreToken::In,
        "case" => CoreToken::Case,
        other @ _ => CoreToken::Ident(other.to_string())
    }

}


fn tokenize(program: String) -> Vec<CoreToken> {

    fn is_char_space(c: char) -> bool {
        c == ' ' || c == '\n' || c == '\t'
    }

    fn is_char_symbol(c: char) -> bool {
        !c.is_alphabetic() && !c.is_numeric()
    }
    //let char_arr : &[u8] = program.as_bytes();
    let char_arr : Vec<_> = program.clone().chars().collect();
    let mut i = 0;

    let mut tokens = Vec::new();

    loop {
        //break out if we have exhausted the loop
        if char_arr.get(i) == None {
            break;
        }

        //consume spaces
        while let Some(& c) = char_arr.get(i) {
            if !is_char_space(c) {
                break;
            }
            i += 1;
        }

        //we have a character
        if let Some(& c) = char_arr.get(i) {
            //alphabet: parse literal
            if c.is_alphabetic() {

                //get the identifier name
                let mut id_string = String::new();

                while let Some(&c) = char_arr.get(i) {
                    if c.is_alphanumeric() {
                        id_string.push(c);
                        i += 1;
                    } else {
                        break;
                    }
                }

                tokens.push(identifier_str_to_token(&id_string));
            }
            else if c.is_numeric() {
                //parse the number
                //TODO: take care of floats

                let mut num_string = String::new();

                while let Some(&c) = char_arr.get(i) {
                    if c.is_numeric() {
                        num_string.push(c);
                        i += 1;
                    } else {
                        break;
                    }
                }

                tokens.push(CoreToken::Integer(num_string));

            }
            else {
                assert!(is_char_symbol(c),
                        format!("{} is not charcter, digit or symbol", c));

                let symbol_token_map: HashMap<&str, CoreToken> =
                        [("=", CoreToken::Equals),
                         (";", CoreToken::Semicolon),
                         ("(", CoreToken::OpenRoundBracket),
                         (")", CoreToken::CloseRoundBracket),
                         ("(", CoreToken::OpenRoundBracket),
                         ("|", CoreToken::Or),
                         ("&", CoreToken::And),
                         ("<", CoreToken::L),
                         ("<=", CoreToken::LEQ),
                         (">", CoreToken::G),
                         (">=", CoreToken::GEQ),
                         ("+", CoreToken::Plus),
                         ("-", CoreToken::Minus),
                         ("*", CoreToken::Mul),
                         ("/", CoreToken::Div),
                         ("\\", CoreToken::Lambda)]
                         .iter().cloned().collect();


                let longest_op_len = symbol_token_map
                                        .keys()
                                        .map(|s| s.len())
                                        .fold(0, cmp::max);


                //take only enough to not cause an out of bounds error
                let length_to_take = cmp::min(longest_op_len,
                                              char_arr.len() - i);

                //take all lengths, starting from longest,
                //ending at shortest
                let mut longest_op_opt : Option<CoreToken> = None;
                let mut longest_taken_length = 0;

                for l in (1..length_to_take+1).rev() {
                    let op_str : String = char_arr[i..i + l]
                                            .iter()
                                            .cloned()
                                            .collect();

                    if let Some(tok) = symbol_token_map.get(&op_str.as_str()) {
                        //we found a token, break
                        longest_taken_length = l;
                        longest_op_opt = Some(tok.clone());
                        break;
                    }
                }

                //longest operator is tokenised
                //TODO: figure out why this fucks up
                let longest_op : CoreToken = (longest_op_opt
                                .ok_or(ParseError::UnknownSymbol)).unwrap();

                tokens.push(longest_op);
                i += longest_taken_length;




            }

        }

    }

    tokens

}


//TODO: write an infix parser for equals
//this is now totally possible since (=) is used to parse only
//<var> = <defn> for let expressions.


/*
fn parse_defn(mut c: &mut ParserCursor, p: &Parser) ->
    Result<(CoreVariable, Box<CoreExpr>), ParseError> {

    if let CoreToken::Ident(name) =  try!(c.consume()) {
        try!(c.expect(CoreToken::Equals));

        print!("found = ");
        if let CoreAST::Expr(rhs) = try!(p.parse(&mut c, 0)) {
            return Result::Ok((name, Box::new(rhs)))
        }
        else {
            panic!("expected expr at let binding rhs");
        }

    }
    else {
        panic!("variable name expected at defn");
    }
}
*/

//does this token allow us to start to parse an
//atomic expression?
fn is_token_atomic_expr_start(t: CoreToken) -> bool {
    match t {
        CoreToken::Integer(_) => true,
        CoreToken::Ident(_) => true,
        CoreToken::OpenRoundBracket => true,
        _ => false
    }

}


//atomic := <num> | <ident> | "(" <expr> ")"
fn parse_atomic_expr(mut c: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    match c.peek() {
        CoreToken::Integer(num_str) => {
            try!(c.consume());
            let num = try!(i32::from_str_radix(&num_str, 10)
                           .map_err(|_|
                                    ParseError::ParseErrorStr(
                                        format!("unable to parse {} as int",
                                                num_str).to_string())));
            Result::Ok(CoreExpr::Num(num))
        },
        CoreToken::Ident(ident) => {
            try!(c.consume());
            Result::Ok(CoreExpr::Variable(ident))
        },
        CoreToken::OpenRoundBracket => {
            try!(c.expect(CoreToken::OpenRoundBracket));
            let inner_expr = try!(parse_expr(&mut c));
            try!(c.expect(CoreToken::CloseRoundBracket));
            Result::Ok(inner_expr)
        },
        other @ _ =>
            panic!("expected integer, identifier or (<expr>), found {:#?}", other)
    }

}

//defn := <variable> "=" <expr>
fn parse_defn(mut c: &mut ParserCursor) ->
    Result<(CoreVariable, Box<CoreExpr>), ParseError> {

    if let CoreToken::Ident(name) = c.peek() {
        try!(c.consume());
        try!(c.expect(CoreToken::Equals));

        let rhs : CoreExpr = try!(parse_expr(&mut c));
        Result::Ok((name, Box::new(rhs)))

    }
    else {
        panic!("variable name expected at defn");
    }
}

//let := "let" <bindings> "in" <expr>
fn parse_let(mut c: &mut ParserCursor) -> Result<CoreLet, ParseError> {
    //<let>
    let let_token = match c.peek() {
        CoreToken::Let => try!(c.consume()),
        CoreToken::LetRec => try!(c.consume()),
        _ => panic!("expected let or letrec, found {:#?}", c.peek())
    };

    let mut bindings : Vec<(Name, Box<CoreExpr>)> = Vec::new();

    //<bindings>
    loop {
        let defn = try!(parse_defn(&mut c));
        bindings.push(defn);

        //check for ;
        //If htere is a ;, continue parsing
        if let CoreToken::Semicolon = c.peek() {
            try!(c.consume());
            continue;
        }
        else {
            break;
        }
    }
    //<in>
    try!(c.expect(CoreToken::In));

    //<expr>
    let rhs_expr = try!(parse_expr(c));

    let is_rec : bool = match let_token {
        CoreToken::Let => false,
        CoreToken::LetRec => true,
        other @ _ =>
            return Result::Err(ParseError::UnexpectedToken {
            expected: vec![CoreToken::Let, CoreToken::LetRec],
            found: other.clone()
        })
    };

    Result::Ok(CoreLet {
        is_rec: is_rec,
        bindings: bindings,
        expr: Box::new(rhs_expr)
    })
}

fn parse_application(mut cursor: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    let mut application_vec : Vec<CoreExpr> = Vec::new();
    loop {
        let c = cursor.peek();
        if is_token_atomic_expr_start(c) {
            let atomic_expr = try!(parse_atomic_expr(&mut cursor));
            application_vec.push(atomic_expr);
        } else {
            break;
        }
    }

    if application_vec.len() == 0 {
        Result::Err(
            ParseError::ParseErrorStr(
                concat!("wanted function application or atomic expr",
                        "found neither").to_string()))

    }
    else if application_vec.len() == 1 {
        //just an atomic expr
        Result::Ok(application_vec.remove(0))
    }
    else {

        //function application
        //convert f g x  y to
        //((f g) x) y
        let mut cur_ap_lhs = {
            let ap_lhs = application_vec.remove(0);
            let ap_rhs = application_vec.remove(0);
            CoreExpr::Application(Box::new(ap_lhs), Box::new(ap_rhs))
        };

        //drop the first two and start folding
        for ap_rhs in application_vec.into_iter() {
            cur_ap_lhs = CoreExpr::Application(Box::new(cur_ap_lhs), Box::new(ap_rhs));
        }

        Result::Ok(cur_ap_lhs)
    }
}

fn parse_mul_div(mut cursor: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    parse_application(&mut cursor)
}

//TODO: refactor this and add_relop to share code of branching and stuff
fn parse_add_sub(mut cursor: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    let mul_div_lhs : CoreExpr = try!(parse_mul_div(&mut cursor));

    let c : CoreToken = cursor.peek();

    let mul_div_rhs : CoreExpr = {
        if c == CoreToken::Plus {
            cursor.expect(CoreToken::Plus);
            try!(parse_mul_div(&mut cursor))
        }
        else {
            return Result::Ok(mul_div_lhs)
        }
    };

    let operator : CoreExpr = {
        if c == CoreToken::Plus {
            CoreExpr::Variable("+".to_string())
        }
        else {
            panic!("unknown token for add sub: {:#?}", c)
        }
    };

    let ap_inner =
        CoreExpr::Application(Box::new(operator), Box::new(mul_div_lhs));

    Result::Ok(CoreExpr::Application(Box::new(ap_inner),
                                     Box::new(mul_div_rhs)))


}

fn parse_relop(mut cursor: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    let add_sub_lhs : CoreExpr = try!(parse_add_sub(&mut cursor));

    let c : CoreToken = cursor.peek();

    let add_sub_rhs : CoreExpr = {
        if c == CoreToken::L {
            try!(parse_add_sub(&mut cursor))
        } else {
            return Result::Ok(add_sub_lhs);
        }
    };

    let operator : CoreExpr =
        if c == CoreToken::L {
            CoreExpr::Variable("<".to_string())
        } else {
            panic!("unknown token for relop: {:#?}", c)
        };


    let ap_inner =
        CoreExpr::Application(Box::new(operator), Box::new(add_sub_lhs));

    Result::Ok(CoreExpr::Application(Box::new(ap_inner),
                                     Box::new(add_sub_rhs)))

}

//TODO: make a higher order function to unify AND, OR, PLUS, etc. they're
//basically the same thing
//expr2 -> expr3 "&" expr2 | expr3
fn parse_and(mut cursor: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    (parse_relop(&mut cursor))
}

//expr1 -> expr2 "|" expr1 | expr1
fn parse_or(mut cursor: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    let and_lhs = try!(parse_and(&mut cursor));

    if let CoreToken::Or = cursor.peek() {
        try!(cursor.consume());
        let and_rhs = try!(parse_and(&mut cursor));

        let or_val = CoreExpr::Variable("|".to_string());
        let ap_inner =
            CoreExpr::Application(Box::new(or_val),
                                  Box::new(and_lhs));

        Result::Ok(CoreExpr::Application(Box::new(ap_inner),
                                         Box::new(and_rhs)))
    } else {
        Result::Ok(and_lhs)
    }
}



fn parse_expr(mut c: &mut ParserCursor) ->
    Result<CoreExpr, ParseError> {
    match c.peek() {
        CoreToken::Let => parse_let(&mut c).map(|l| CoreExpr::Let(l)),
        CoreToken::LetRec => parse_let(&mut c).map(|l| CoreExpr::Let(l)),
        CoreToken::Case => panic!("cannot handle case yet"),
        CoreToken::Lambda => panic!("cannot handle lambda yet"),
        _ => parse_or(&mut c)
    }
}




fn string_to_program(string: String) -> Result<CoreProgram, ParseError> {

    let tokens : Vec<CoreToken> = tokenize(string);
    let mut cursor: ParserCursor = ParserCursor::new(tokens);

    let mut program : CoreProgram = Vec::new();

    loop {
        if let CoreToken::Ident(sc_name) = cursor.peek() {
            try!(cursor.consume());

            let mut sc_args = Vec::new();
            //<args>* = <expr>
            while cursor.peek() != CoreToken::Equals &&
                  cursor.peek() != CoreToken::PeekNoToken {
                if let CoreToken::Ident(sc_arg) = cursor.peek() {
                    try!(cursor.consume());
                    sc_args.push(sc_arg);

                }
                else {
                    panic!("super combinator argument expected, {:#?} encountered",
                           cursor.consume());
                }
            }
            //take the equals
            try!(cursor.expect(CoreToken::Equals));
            let sc_body = try!(parse_expr(&mut cursor));

            program.push(SupercombDefn{
                name: sc_name,
                args: sc_args,
                body: sc_body
            });

            match cursor.peek() {
                //we ran out of tokens, this is the last SC
                //break
                CoreToken::PeekNoToken => break,
                //we got a ;, more SCs to come
                CoreToken::Semicolon => {
                    try!(cursor.expect(CoreToken::Semicolon));
                    continue
                },
                other @ _ => panic!("expected either ; or EOF, found {:#?}",
                                    other)
            }

        } else {
            panic!("super combinator name expected, {:#?} encountered",
                   cursor.consume());
        }
    }
    Result::Ok(program)
}


fn run_machine(program:  &str) -> Machine {
    let main = string_to_program(program.to_string())
               .unwrap()
               .remove(0);
    let mut m = Machine::new(vec![main]);
    while !machine_is_final_state(&m) {
        m.step();
    }
    return m
}

#[test]
fn test_skk3() {
    let m = run_machine("main = S K K 3");
    assert!(m.heap.get(&m.stack.peek()) == HeapNode::Num(3));
}

#[test]
fn test_negate_simple() {
    let m = run_machine("main = negate 1");
    assert!(m.heap.get(&m.stack.peek()) == HeapNode::Num(-1));
}

#[test]
fn test_negate_inner_ap() {
    let m = run_machine("main = negate (negate 1)");
    assert!(m.heap.get(&m.stack.peek()) == HeapNode::Num(1));
}


#[test]
fn test_add_simple() {
    let m = run_machine("main = 1 + 1");
    assert!(m.heap.get(&m.stack.peek()) == HeapNode::Num(2));
}

#[test]
fn test_add_lhs_ap() {
    let m = run_machine("main = (negate 1) + 1");
    assert!(m.heap.get(&m.stack.peek()) == HeapNode::Num(0));
}


#[test]
fn test_add_rhs_ap() {
    let m = run_machine("main = 1 + (negate 3)");
    assert!(m.heap.get(&m.stack.peek()) == HeapNode::Num(-2));
}

#[test]
fn test_add_lhs_rhs_ap() {
    let m = run_machine("main = (negate 1) + (negate 3)");
    assert!(m.heap.get(&m.stack.peek()) == HeapNode::Num(-4));
}

// main ---
fn main() {

    let main = string_to_program("main = letrec x = 3; y = x; z = y in (negate (negate (I x)))".to_string()).unwrap().remove(0);
    let mut m = Machine::new(vec![main]);

    let mut i = 1;
    while i <= 100 {
        let env = m.step();
        print!("\n\n>>> i: {} <<<\n", i);
        print_machine(&m, &env);
        i += 1;

        if machine_is_final_state(&m)  { break; }
    }
}
