//use std::collections::vec_deque::VecDeque;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;
//use std::collections::Vec;

type Addr = i32;
type Name = String;


#[derive(Clone)]
enum CoreExpr {
    Variable(Name),
    Num(i32),
    Application(Box<CoreExpr>, Box<CoreExpr>),
}

impl fmt::Debug for CoreExpr {

    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &CoreExpr::Variable(ref name) => write!(fmt, "{}", name),
            &CoreExpr::Num(ref num) => write!(fmt, "n_{}", num),
            &CoreExpr::Application(ref e1, ref e2) => 
                write!(fmt, "({:#?} {:#?})", *e1, *e2)
        }
    }
}


#[derive(Clone)]
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
        write!(fmt, "=  {:#?}", self.body)

    }

}


//a core program is a list of supercombinator
//definitions
type CoreProgram = Vec<SupercombDefn>;



//heap nodes
#[derive(Clone)]
enum HeapNode {
    HeapNodeAp {
        fn_addr: Addr,
        arg_addr: Addr
    },
    HeapNodeSupercomb(SupercombDefn),
    HeapNodeNum(i32),
}

impl fmt::Debug for HeapNode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &HeapNode::HeapNodeAp{ref fn_addr, ref arg_addr} => {
                write!(fmt, "({} $ {})", fn_addr, arg_addr)
            }
            &HeapNode::HeapNodeSupercomb(ref sc_defn) => {
                write!(fmt, "{:#?}", sc_defn)
            },
            &HeapNode::HeapNodeNum(ref num)  => {
                write!(fmt, "{}", num)
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
        write!(fmt, "{:#?}", self.heap)
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

}

//state of the machine
#[derive(Clone)]
struct Machine {
    stack : Stack,
    heap : Heap,
    globals: Bindings,
    dump: Dump
}

fn print_machine(m: &Machine, env: &Bindings) {
    print!("\n\n\n");
    print!( "\n*** env: ***\n");
    for (name, addr) in env.iter() {
        print!("{} => {:#?}\n", name, m.heap.get(addr).unwrap());
    }

    print!( "*** stack: ***\n");
    for addr in m.stack.iter() {
        print!("{} => {:#?}\n", *addr, m.heap.get(addr).unwrap());
    }

    print!("*** heap: ***\n");
    print!("{:#?}", m.heap);
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
            heap: initial_heap
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
        
        self.run_step(heap_node)
    }

    //make an environment for the execution of the supercombinator
    fn make_environment(sc_defn: &SupercombDefn,
                        heap: &Heap,
                        stack_args:&Vec<Addr>,
                        globals: &Bindings) -> Bindings {

        assert!(stack_args.len() == sc_defn.args.len());

        let mut env = globals.clone();

        for (arg_name, arg_addr) in 
                sc_defn.args.iter().zip(stack_args.iter()) {
                let heap_lookup = heap.get(arg_addr).unwrap();
                let param_addr = match heap_lookup {
                    HeapNode::HeapNodeAp{arg_addr, ..} => arg_addr,
                    _ => panic!("did not find application node")
                };
            env.insert(arg_name.clone(), param_addr);
    
        }
        env
    }

    //actually run_step the computation
    fn run_step(&mut self, heap_val: HeapNode) -> Bindings {
        match heap_val {
            HeapNode::HeapNodeNum(n) => 
                panic!("number applied as a function: {}", n),

            HeapNode::HeapNodeAp{fn_addr, ..} => {
                //push function address over the function
                self.stack.push(fn_addr);
                self.globals.clone()
            }
            HeapNode::HeapNodeSupercomb(sc_defn) => {
                //pop the supercombinator
                let _ = self.stack.pop();

                let arg_addrs = {
                    let mut addrs = Vec::new();
                    for _ in (0..sc_defn.args.len()) {
                        addrs.push(self.stack.pop().unwrap());
                    }
                    addrs
                };

                let env = Machine::make_environment(&sc_defn,
                                                    &self.heap,
                                                    &arg_addrs,
                                                    &self.globals);

                let new_alloc_addr = self.instantiate(sc_defn.body, &env);

                self.stack.push(new_alloc_addr);
                env
            }
        }
    }

    
    fn instantiate(&mut self, expr: CoreExpr, env: &Bindings) -> Addr {
        match expr {
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



fn main() {
    //I 3
    let program_expr = CoreExpr::Application(
        Box::new(CoreExpr::Variable("I".to_string())),
        Box::new(CoreExpr::Num(10))
    );

    //let program_expr = (CoreExpr::Num(10));

    let main = SupercombDefn {
        name: "main".to_string(),
        args: Vec::new(),
        body: program_expr
    };

    let mut m = Machine::new(vec![main]);
    
    let mut i = 1;
    while i <= 4 {
        let env = m.step();
        print_machine(&m, &env);
        i += 1;

        if machine_is_final_state(&m)  { break; }
    }
}
