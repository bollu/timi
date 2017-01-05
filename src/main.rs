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
enum CoreAST {
    Expr(CoreExpr),
    Unecessary
}

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
    Let(CoreLet),
    /*
    Or(Box<CoreExpr>, Box<CoreExpr>),
    And(Box<CoreExpr>, Box<CoreRxpr>),
    //relational expressions
    L(Box<CoreExpr>, Box<CoreExpr>),
    //Add/Sub expr
    Add(Box<CoreExpr>, Box<CoreExpr>),
    //mul/div
    MulExpr(Box<CoreExpr>, Box<CoreExpr>),
    */
}

impl fmt::Debug for CoreExpr {

    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &CoreExpr::Variable(ref name) => write!(fmt, "{}", name),
            &CoreExpr::Num(ref num) => write!(fmt, "n_{}", num),
            &CoreExpr::Application(ref e1, ref e2) => 
                write!(fmt, "({:#?} {:#?})", *e1, *e2),
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
            
             CoreExpr::Let(CoreLet{expr, ..}) => {
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

#[derive(Clone, Debug)]
enum ParseError {
    NoTokens,
    NoPrefixParserFound(CoreToken),
    UnknownSymbol,
    UnexpectedToken { 
        expected: Vec<CoreToken>, 
        found: CoreToken
    },
    UnexpectedAST {
        found: CoreAST
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
            self.consume();
            Result::Ok(())
        } else {
            Result::Err(ParseError::UnexpectedToken{
                expected: vec![t],
                found: tok
            })
        }
    }
}




/*

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
    fn parse(&self, mut cursor: &mut ParserCursor, precedence: i32) -> Result<CoreAST, ParseError> {

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
    parse: fn (c: &mut ParserCursor, p: &Parser, t: &CoreToken) -> Result<CoreAST, ParseError>

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
    parse: fn(c: &mut ParserCursor, p: &Parser, left: &CoreAST, t: &CoreToken) -> 
            Result<CoreAST, ParseError>,
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
*/


fn identifier_str_to_token(token_str: &str) -> CoreToken {
    match token_str {
        "let" => CoreToken::Let,
        "letrec" => CoreToken::LetRec,
        "in" => CoreToken::In,
        "case" => CoreToken::Case,
        other @ _ => CoreToken::Ident(other.to_string())
    }

}

fn is_char_space(c: char) -> bool {
    c == ' ' || c == '\n' || c == '\t'
}

fn is_char_symbol(c: char) -> bool {
    !c.is_alphabetic() && !c.is_numeric()
}

fn tokenize(program: String) -> Vec<CoreToken> {
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
            c.consume();
            let num = try!(i32::from_str_radix(&num_str, 10)
                           .map_err(|_| 
                                    ParseError::ParseErrorStr(
                                        format!("unable to parse {} as int",
                                                num_str).to_string())));
            Result::Ok(CoreExpr::Num(num))
        },
        CoreToken::Ident(ident) => {
            c.consume();
            Result::Ok(CoreExpr::Variable(ident))
        },
        CoreToken::OpenRoundBracket => {
            try!(c.expect(CoreToken::OpenRoundBracket));
            let inner_expr = try!(parse_expr(&mut c));
            c.expect(CoreToken::CloseRoundBracket);
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
        c.consume();
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
        other @ _ => panic!("expected let or letrec, found {:#?}", c.peek())
    };

    let mut bindings : Vec<(Name, Box<CoreExpr>)> = Vec::new();

    //<bindings>
    loop {
        let defn = try!(parse_defn(&mut c));
        bindings.push(defn);
            
        //check for ;
        //If htere is a ;, continue parsing
        if let CoreToken::Semicolon = c.peek() {
            c.consume();
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
        if (is_token_atomic_expr_start(c)) {
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
        for ap_rhs in application_vec.into_iter().skip(2) {
            cur_ap_lhs = CoreExpr::Application(Box::new(cur_ap_lhs), Box::new(ap_rhs));
        }

        Result::Ok(cur_ap_lhs)
    }
}

fn parse_mul_div(mut cursor: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    parse_application(&mut cursor)
}


fn parse_add_sub(mut cursor: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    parse_mul_div(&mut cursor)
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
        cursor.consume();
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
        other @ _ => parse_or(&mut c)
    }
}


fn string_to_program(string: String) -> Result<CoreProgram, ParseError> {

    let tokens : Vec<CoreToken> = tokenize(string);
    let mut cursor: ParserCursor = ParserCursor::new(tokens);

    let mut program : CoreProgram = Vec::new();

    loop {
        if let CoreToken::Ident(sc_name) = cursor.peek() {
            cursor.consume();

            let mut sc_args = Vec::new();
            //<args>* = <expr>
            while cursor.peek() != CoreToken::Equals && 
                  cursor.peek() != CoreToken::PeekNoToken {
                if let CoreToken::Ident(sc_arg) = cursor.peek() {
                    print!("found arg: {}\n", sc_arg);
                    cursor.consume();
                    sc_args.push(sc_arg);
                
                }
                else {
                    panic!("super combinator argument expected, {:#?} encountered",
                           cursor.consume());
                }
            }
            //take the equals
            cursor.expect(CoreToken::Equals);
            print!("trying to parse expr...\n");
            let sc_body = try!(parse_expr(&mut cursor));

            program.push(SupercombDefn{
                name: sc_name,
                args: sc_args,
                body: sc_body
            });


            print!("supercombs: \n{:#?}\n", program);


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

// main ---
fn main() {
    print!("tokens: {:#?\n}",
           tokenize("1 2 3;" .to_string()));


    print!("parse: {:#?}",
           string_to_program("I x = 12".to_string()));
    return;
    
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
