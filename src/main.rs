#[macro_use]
extern crate nom;

use std::collections::HashMap;
use std::fmt;
use std::fmt::{Write};

type Addr = i32;
type Name = String;


#[derive(Clone, PartialEq, Eq)]
enum CoreExpr {
    Variable(Name),
    Num(i32),
    Application(Box<CoreExpr>, Box<CoreExpr>),
    Let {
        is_rec: bool,
        bindings: Vec<(Name, Box<CoreExpr>)>,
        expr: Box<CoreExpr>
    }
}

impl fmt::Debug for CoreExpr {

    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &CoreExpr::Variable(ref name) => write!(fmt, "{}", name),
            &CoreExpr::Num(ref num) => write!(fmt, "n_{}", num),
            &CoreExpr::Application(ref e1, ref e2) => 
                write!(fmt, "({:#?} {:#?})", *e1, *e2),
            &CoreExpr::Let{ref is_rec, ref bindings, ref expr} => {
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
        write!(fmt, "{} ", &self.name);
        for arg in self.args.iter() {
            write!(fmt, "{} ", &arg);
        }
        write!(fmt, "{{ {:#?} }}", self.body)

    }

}


//a core program is a list of supercombinator
//definitions
type CoreProgram = Vec<SupercombDefn>;

//heap nodes
#[derive(Clone, PartialEq, Eq)]
enum HeapNode {
    HeapNodeAp {
        fn_addr: Addr,
        arg_addr: Addr
    },
    HeapNodeSupercomb(SupercombDefn),
    HeapNodeNum(i32),
    HeapNodeIndirection(Addr),
}

impl fmt::Debug for HeapNode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &HeapNode::HeapNodeAp{ref fn_addr, ref arg_addr} => {
                write!(fmt, "H-({} $ {})", fn_addr, arg_addr)
            }
            &HeapNode::HeapNodeSupercomb(ref sc_defn) => {
                write!(fmt, "H-{:#?}", sc_defn)
            },
            &HeapNode::HeapNodeNum(ref num)  => {
                write!(fmt, "H-{}", num)
            }
            &HeapNode::HeapNodeIndirection(ref addr)  => {
                write!(fmt, "H-indirection-{}", addr)
            }
        }
    }
}

impl HeapNode {
    fn is_data_node(&self) -> bool {
        match self {
            &HeapNode::HeapNodeNum(_) => true,
            _ => false
        }
    }
}


//unsued for mark 1
#[derive(Clone, Debug)]
struct Dump {}

//stack of addresses of nodes. "Spine"
type Stack = Vec<Addr>;

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
            write!(fmt, "\t{} => {:#?}\n", key, val);
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

    fn get(&self, addr: &Addr) -> Option<HeapNode> {
        self.heap.get(&addr).cloned()
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
        &HeapNode::HeapNodeIndirection(addr) => format!("indirection: {}", addr),
        &HeapNode::HeapNodeNum(num) => format!("{}", num),
        &HeapNode::HeapNodeAp{ref fn_addr, ref arg_addr} => 
            format!("({} $ {})",
                    format_heap_node(m, env, &m.heap.get(fn_addr).unwrap()),
                    format_heap_node(m, env, &m.heap.get(arg_addr).unwrap())),
        &HeapNode::HeapNodeSupercomb(ref sc_defn) =>  {
            let mut sc_str = String::new();
            write!(&mut sc_str, "{}", sc_defn.name);
            sc_str
        
        }
    }
}

fn print_machine(m: &Machine, env: &Bindings) {
    print!("\n\n===\n\n");

    print!( "*** stack: ***\n");
    print!( "## top ##\n");
    for addr in m.stack.iter().rev() {
        print!("heap[{}] :  {}\n", *addr, format_heap_node(m, env, &m.heap.get(addr).unwrap()));
    }
    print!( "## bottom ##\n");


    print!("*** heap: ***\n");
    print!("{:#?}", m.heap);

    print!( "\n*** env: ***\n");
    let mut env_ordered : Vec<(&Name, &Addr)> = env.iter().collect();
    env_ordered.sort_by(|e1, e2| e1.0.cmp(e2.0));
    for &(name, addr) in env_ordered.iter() {
        print!("{} => {}\n", name, format_heap_node(m, env, &m.heap.get(addr).unwrap()));
    }
}



fn get_prelude() -> CoreProgram {
    let mut v = Vec::new();
    let I = SupercombDefn {
        name: "I".to_string(),
        args: vec!("x".to_string()),
        body: CoreExpr::Variable("x".to_string())
    };

    v.push(I);

    let K = SupercombDefn {
        name: "K".to_string(),
        args: vec!("x".to_string(), "y".to_string()),
        body: CoreExpr::Variable("x".to_string())
    };
    v.push(K);

    let K1 = SupercombDefn {
        name: "K1".to_string(),
        args: vec!("x".to_string(), "y".to_string()),
        body: CoreExpr::Variable("y".to_string())
    };
    v.push(K1);

    let S = SupercombDefn {
        name: "S".to_string(),
        args: vec!("f".to_string(),
                   "g".to_string(),
                   "x".to_string()),
        //f x (g x)
        body: CoreExpr::Application(
                // (f x)
                Box::new(CoreExpr::Application(
                            Box::new(CoreExpr::Variable("f".to_string())),
                            Box::new(CoreExpr::Variable("x".to_string()))
                        )
                ),
                // (g x)
                Box::new(CoreExpr::Application(
                            Box::new(CoreExpr::Variable("g".to_string())),
                            Box::new(CoreExpr::Variable("x".to_string()))
                        )
                )
        )
    };

    v.push(S);

    //compose f g x =  f (g x)
    let compose = SupercombDefn {
        name: "compose".to_string(),
        args: vec!("f".to_string(), "g".to_string(), "x".to_string()),
        body: {
                let gx = CoreExpr::Application(
                            Box::new(CoreExpr::Variable("g".to_string())),
                            Box::new(CoreExpr::Variable("x".to_string()))
                            );

                CoreExpr::Application(
                    Box::new(CoreExpr::Variable("f".to_string())),
                    Box::new(gx))
        }
    };

    v.push(compose);
    
    //twice f = (compose f) f
    let twice = SupercombDefn {
        name: "twice".to_string(),
        args: vec!("f".to_string()),
        body: {
                let compose_f = CoreExpr::Application(
                            Box::new(CoreExpr::Variable("compose".to_string())),
                            Box::new(CoreExpr::Variable("f".to_string()))
                            );

                CoreExpr::Application(
                    Box::new(compose_f),
                    Box::new(CoreExpr::Variable("f".to_string()))
                    )
        }
    };

    v.push(twice);

    v
}

fn heap_build_initial(sc_defs: CoreProgram) -> (Heap, Bindings) {
    let mut heap = Heap::new();
    let mut globals = HashMap::new();

    for sc_def in sc_defs.iter() {
        //create a heap node for the supercombinator definition
        //and insert it
        let node = HeapNode::HeapNodeSupercomb(sc_def.clone());
        let addr = heap.alloc(node);

        //insert it into the globals, binding the name to the
        //heap address
        globals.insert(sc_def.name.clone(), addr);
    }

    (heap, globals)
}

//interreter

impl Machine {
    fn new(program: CoreProgram) -> Machine {
        //all supercombinators = program + prelude
        let mut sc_defs = program.clone();
        sc_defs.extend(get_prelude().iter().cloned());

        let (initial_heap, globals) = heap_build_initial(sc_defs);
        
        //get main out of the heap
        let main_addr : Addr = match globals.get("main") {
            Some(main) => main,
            None => panic!("no main found")
        }.clone();

        Machine {
            dump: Dump{},
            //stack has addr main on top
            stack:  {
                let mut s = Vec::new();
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
        let tos_addr : Addr = match self.stack.last() {
            Some(addr) => *addr,
            None => panic!("unable to step: top of stack empty")
        };

        let heap_node = match self.heap.get(&tos_addr)  {
            Some(node) => node,
            None => panic!("heap access violation")
        };
        
        self.run_step(&heap_node)
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

                let application = heap.get(application_addr).unwrap();
                let param_addr = match application {
                    HeapNode::HeapNodeAp{arg_addr, ..} => arg_addr,
                    _ => panic!(concat!("did not find application node when ",
                                        "unwinding stack for supercombinator"))
                };
            env.insert(arg_name.clone(), param_addr);
    
        }
        env
    }

    //actually run_step the computation
    fn run_step(&mut self, heap_val: &HeapNode) -> Bindings {
        match heap_val {
            &HeapNode::HeapNodeNum(n) => 
                panic!("number applied as a function: {}", n),

            &HeapNode::HeapNodeAp{fn_addr, ..} => {
                //push function address over the function
                self.stack.push(fn_addr);
                self.globals.clone()
            }
            &HeapNode::HeapNodeIndirection(ref addr) => {
                //simply ignore an indirection during execution, and
                //push the indirected value on the stack
                self.stack.pop();
                self.stack.push(*addr);
                self.globals.clone()
            }
            &HeapNode::HeapNodeSupercomb(ref sc_defn) => {

                //pop the supercombinator
                let sc_addr = self.stack.pop().expect("stack must have value");

                //the arguments are the stack 
                //values below the supercombinator. There
                //are (n = arity of supercombinator) arguments
                let arg_addrs = {
                    let mut addrs = Vec::new();
                    for _ in 0..sc_defn.args.len() {
                        addrs.push(self.stack.pop().unwrap());
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
                    self.heap.rewrite(&full_call_addr, HeapNode::HeapNodeIndirection(new_alloc_addr));
                }
                env
            }
        }
    }

    
    fn instantiate(&mut self, expr: CoreExpr, env: &Bindings) -> Addr {
        match expr {
            
            CoreExpr::Let{expr, ..} => {
                panic!("need to setup environment for let");
                return self.instantiate(*expr, env);
            }
            CoreExpr::Num(x) => self.heap.alloc(HeapNode::HeapNodeNum(x)),
            CoreExpr::Application(fn_expr, arg_expr) => {
                let fn_addr = self.instantiate(*fn_expr, env);
                let arg_addr = self.instantiate(*arg_expr, env);

                self.heap.alloc(HeapNode::HeapNodeAp {
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
    match m.stack.last() {
        Some(addr) => {
           //machine has more than 1 address, just return false
           if m.stack.len() > 1 {
                false
           } else {
                match  m.heap.get(addr) {
                    Some(node) => node.is_data_node(),
                    None => panic!("top of stack points to invalid address")
                }
           }
        }
        None => panic!("stack empty")
    }

}

//parsing ---

#[derive(Clone)]
enum ParseError {
    NoTokens,
    NoPrefixParserFound(CoreToken),
}

#[derive(Clone, PartialEq, Eq)]
enum CoreToken {
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

    fn peek(&self) -> Result<CoreToken, ParseError> {
        self.tokens.get(self.pos)
            .cloned()
            .ok_or(ParseError::NoTokens)
            
    }

    fn consume(&mut self) -> Result<CoreToken, ParseError> {
        let tok = try!(self.peek());
        self.pos += 1;
        Result::Ok(tok)

    }
}

struct Parser{
    prefix_parselets: Vec<PrefixParselet>,
    infix_parselets: Vec<InfixParselet>,
}


impl Parser {
    fn new(prefix_parselets: Vec<PrefixParselet>,
            infix_parselets: Vec<InfixParselet>) -> Parser {
        Parser {
            prefix_parselets: prefix_parselets,
            infix_parselets: infix_parselets,
        }
    }

    //TODO: I am cloning the whole vector, this is retarded. Must find
    //better way to escape borrow checker
    fn parse(&self, mut cursor: &mut ParserCursor, precedence: i32) -> Result<CoreExpr, ParseError> {

        //if we have no more tokens, return.
        //othewise, continue parsing
        let tok_prefix = try!(cursor.peek());
           
        let mut expr_left = try!({
            for prefix in self.prefix_parselets.clone() {
                if (prefix.will_parse)(&tok_prefix) {
                    cursor.consume();
                    return (prefix.parse)(&mut cursor, &self, &tok_prefix);
                }
            }

            return Result::Err(ParseError::NoPrefixParserFound(tok_prefix));
        });

        loop {
            let mut parser_found = false;

            //try to look for left parser
            let tok_infix_peek = try!(cursor.peek());

            expr_left = try!({
                for infix in self.infix_parselets.clone() {
                    if (infix.will_parse)(&tok_infix_peek) &&
                        infix.precedence < precedence {
                        
                        cursor.consume();
                        return (infix.parse)(&mut cursor,
                                             &self,
                                                  &expr_left,
                                                  &tok_infix_peek);
                    }
                };

                //no suitable infix parser found
                //so quit
                return Result::Ok(expr_left);
            });
        }
    }

}

struct PrefixParselet {
    will_parse: fn (t: &CoreToken) -> bool,
    parse: fn (c: &mut ParserCursor, p: &Parser, t: &CoreToken) -> Result<CoreExpr, ParseError>

}

impl Clone for PrefixParselet {
    fn clone(&self) -> Self {
        PrefixParselet {
            parse: self.parse,
            will_parse: self.will_parse
        }
    }

}

struct InfixParselet {
    will_parse: fn (t: &CoreToken) -> bool,
    parse: fn(c: &mut ParserCursor, p: &Parser, left: &CoreExpr, t: &CoreToken) -> 
            Result<CoreExpr, ParseError>,
    precedence: i32
}

impl Clone for InfixParselet {
    fn clone(&self) -> Self {
        InfixParselet {
            parse: self.parse,
            will_parse: self.will_parse,
            precedence: self.precedence
        }
    }
}

// main ---
fn main() {
    //I 3
    let program_expr = CoreExpr::Application(
        Box::new(CoreExpr::Variable("I".to_string())),
        Box::new(CoreExpr::Num(10))
    );

    //S K I 3
    let ski3 = CoreExpr::Application(
        Box::new(
            //(SK) I
            CoreExpr::Application(
                //SK
                Box::new(
                    CoreExpr::Application(
                        Box::new(CoreExpr::Variable("S".to_string())),
                        Box::new(CoreExpr::Variable("K".to_string()))
                    )
                )
            ,
            Box::new(CoreExpr::Variable("I".to_string()))
        )),
        Box::new(CoreExpr::Num(3)));

    //(((twice twice) Id)  3)
    let twice_twice_id_3 = {
        
        let twice_twice = CoreExpr::Application(
                            Box::new(CoreExpr::Variable("twice".to_string())),
                            Box::new(CoreExpr::Variable("twice".to_string())));
        let twice_twice_id = CoreExpr::Application(
                                Box::new(twice_twice), 
                                Box::new(CoreExpr::Variable("I".to_string())));
        let twice_twice_id_3 = CoreExpr::Application(
                            Box::new(twice_twice_id),
                            Box::new(CoreExpr::Num(3)));
        twice_twice_id_3
    };

    let main = SupercombDefn {
        name: "main".to_string(),
        args: Vec::new(),
        body: twice_twice_id_3
    };

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
