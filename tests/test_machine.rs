extern crate timi;

#[cfg(test)]
mod test {

    use timi::machine::*;
    use timi::frontend::*;



    pub fn run_machine(program:  &str) -> Machine {

        let main = match string_to_program(&program.to_string()) {
            Result::Ok(main) => main,
            Result::Err(e) => panic!("parse error:\n{}",  e.pretty_print(program))
        };
        let mut m = Machine::new_with_main(main).unwrap();
        while !m.is_final_state() {
            let _ = m.step().unwrap();
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
    fn test_let_simple() {
        let m = run_machine("main = let y = 10 in y + y");
        assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(20));
    }

    #[test]
    fn test_let_forward_dependency() {
        let m = run_machine("main = let y = x; x = 10 in y + y");
        assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(20));
    }

    #[test]
    fn test_let_back_dependency() {
        let m = run_machine("main = let x = 10; y = x in y + y");
        assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(20));
    }

    #[test]
    fn test_let_mutual_uninstantiatable() {
        let m = run_machine("main = let y = K x 20; x = K1 y 10 in y + y");
        assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(20));
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

    #[test]
    fn test_case_pair_simple_left_access() {
        let m = run_machine("main = casePair (MkPair 1 2) K");
        assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(1));
    }

    #[test]
    fn test_case_pair_simple_right_access() {
        let m = run_machine("main = casePair (MkPair 1 2) K1");
        assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(2));
    }


    #[test]
    fn test_case_pair_complex_access_function() {
        let m = run_machine("main = casePair (MkPair 7 4) (compose K fac)");
        assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(5040));
    }

    #[test]
    fn test_list_cons_simple() {
        let m = run_machine("main = caseList (Cons 1 Nil) undef K");
        assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(1));
    }

    #[test]
    fn test_list_cons_complex() {
        //TODO: improve this test by encoding a fold
        let m = run_machine("main = caseList (Cons 1 (Cons 2 Nil)) undef K");
        assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(1));
    }

    #[test]
    fn test_list_nil_simple() {
        let m = run_machine("main = caseList Nil (10) undef");
        assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(10));
    }

    #[test]
    fn test_nil_complex() {
        let m = run_machine("main = caseList Nil (10 * 20) undef");
        assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(200));
    }

    #[test]
    fn test_comments_simple() {
        let m = run_machine("# this is a comment\n\
                            main = XX; # this is also a comment\n\
                            XX = 3");
        assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(3));
    }

    #[test]
    fn test_comments_folow_whitespace() {

        let m = run_machine("# this is a comment\n\
                                   # this is the next comment that follows\n\
                                   # with whitespace\n\
                            main = XX; # this is also a comment\n\
                            XX = 3");
        assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(3));
    }


    #[test]
    fn test_foldr() {
        let m = run_machine("main = foldr plus 0 list;\n\
                             list = Cons 1 (Cons 2 (Cons 3 Nil));\n\
                             plus a b = a + b;\n\
                             foldr_go f seed x xs = (foldr f (f seed x) xs);\n\
                             foldr f seed list = caseList list seed (foldr_go f seed)");
        assert!(m.heap.get(&m.stack.peek().unwrap()) == HeapNode::Num(6));

    }

}
