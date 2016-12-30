use std::collections::vec_deque::VecDeque;
use std::collections::HashMap;


type Addr = i32;
type Name = String;


#[derive(Clone, Debug)]
enum CoreExpr {
    Variable(Name),
    Num(i32),
    Application(Box<CoreExpr>, Box<CoreExpr>),
}


#[derive(Clone, Debug)]
struct SupercombDefn {
    name: String,
    args: Vec<String>,
    body: CoreExpr
}


//a core program is a list of supercombinator
//definitions
type CoreProgram = Vec<SupercombDefn>;



//heap nodes
#[derive(Clone, Debug)]
enum HeapNode {
    HeapNodeAp {
        from_addr: Addr,
        to_addr: Addr
    },
    HeapNodeSupercomb(SupercombDefn),
    HeapNodeNum(f32),
}

//unsued for mark 1
#[derive(Clone, Debug)]
struct Dump {}

//stack of addresses of nodes. "Spine"
type Stack = VecDeque<Addr>;

//maps names to addresses in the heap
type Globals = HashMap<Name, Addr>;

//maps addresses to machine Nodes
#[derive(Clone, Debug)]
struct Heap {
    heap: HashMap<Addr, HeapNode>,
    next_addr: Addr
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
#[derive(Clone, Debug)]
struct Machine {
    stack : Stack,
    heap : Heap,
    globals: Globals,
    dump: Dump
}

fn get_prelude() -> CoreProgram {
    Vec::new()
}

fn heap_build_initial(sc_defs: CoreProgram) -> (Heap, Globals) {
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
                let mut s = VecDeque::new();
                s.push_back(main_addr);
                s
            },
            globals: globals,
            heap: initial_heap
        }

    }
}




fn main() {
    println!("Hello, world!");
}
