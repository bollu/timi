//! Pretty printing mainly focuses on generating clean and readable output
//! for errors, machine state, etc.
use std::collections::HashSet;
use std::fmt::Write;

use ir::*;
use machine::{HeapNode, Stack, Heap, Machine, DataTag, is_addr_phantom, MachineError};

extern crate ansi_term;
use self::ansi_term::Colour::{Blue, Red, Black, Green};

use prettytable::Table;
use prettytable::format::consts::FORMAT_CLEAN;




/// format a heap node, by pretty printing the node.
/// If the node contains recursive structure, this will handle it and
/// print `<<recursive_defn>>`
pub fn format_heap_node(heap: &Heap, addr: &Addr) -> String {
    
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



/// Format an address to be of the form `0xaddress_in_hex`. Used for
/// pretty printing addresses
pub fn format_addr_string(addr: &Addr)  -> String {
    format!("{}{}", Green.paint("0x"), Green.underline().paint(format!("{:X}", addr)))
}

/// returns the pretty-printed version of the final heap node on top
/// of the stack.
///
/// ### Errors
/// Returns an error if the machine is not in the final state
pub fn machine_get_final_val_string(m: &Machine) -> Result<String, MachineError> {
    try!(m.is_final_state());
    Result::Ok(format_heap_node(&m.heap, &m.stack.peek().unwrap()))
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


