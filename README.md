[![Build Status](https://travis-ci.org/bollu/timi.svg?branch=master)](https://travis-ci.org/bollu/timi)
[![Coverage Status](https://coveralls.io/repos/github/bollu/TIM-template-instantiation/badge.svg?branch=master)](https://coveralls.io/github/bollu/TIM-template-instantiation?branch=master)
[![Crates.io](https://img.shields.io/crates/v/timi.svg)](https://crates.io/crates/timi)


TIMi - Template Instantiation Machine Interpreter 
=================================================

A visual, user-friendly implementation of a template instantiation machine. Built to understand how 
lazily evaluate programming language evaluates.

[![asciicast](https://asciinema.org/a/33a5xa2rcglfw1ff6hv6yqu84.png)](https://asciinema.org/a/33a5xa2rcglfw1ff6hv6yqu84)


# Table of Contents
- [Quickstart](#quickstart)
- [Interpreter Options & Usage](#interpreter-options)
- [Language Introduction](#language-introduction)
    - [Top level (Supercombinators)](#top-level)
    - [The `main` value](#main-value)
    - [Expressions](#expressions)
    - [Lack of Lambda & Case](#lack-of-lambda-and-case)
- [Runtime](#runtime)
    - [Components of the machine](#components-of-machine)
    - [Evaluation](#evaluation)
    - [Instantiation](#instantiation)
    - [Primitives](#primitives)
    - [How does evaluation provide laziness?](#how-does-eval-provide-laziness)
    - [The Dump](#the-dump)
- [Roadmap](#roadmap)
- [Design Decisions](#design-decisions)
- [Things Learnt](#things-learnt)
- [References](#references)

## Quickstart

#### Binary from  `cargo`
To get the interpreter `timi` with `cargo` (Rust's package manager), run
```bash
$ cargo install timi && timi
```

#### Build from source
Run
```bash
$ git clone https://github.com/bollu/timi.git && cd timi && cargo run
```
to download and build from source.


#### Using the interpreter

##### Expressions
Type out expressions to evaluate. For example:
```
> 1 + 1
```
will cause `1 + 1` to be evaluated


##### Definitions
Use `define <name> [<param>]* = <expr>` to create new supercombinators.
```
> define plus x y = x + y
```
Will create a function called `plus` that takes two parameters `x` and `y`. To run
this function, call
```
> plus 1 1
```

## Interpreter Options & Usage


#### `>:step`
To go through the execution step-by-step, use
```
>:step

On entering an expression in this mode, the prompt will change to >>
```

- Use `>>s` (for `step`) or `>>n` (for `next`) to go to the next step


#### `>:nostep`
to enable continuous execution of the entire program, use
```
>:nostep
```

## Language Introduction

The language is a small, lazily evaluated language. Lazy evaluation means that
evaluation is delayed till a value is needed.


#### Top level (Supercombinators)

Top level declarations (which are also called *supercombinators*) are of
the form:
```
<name> [args]* = <core expr>
```

Example:
```
K x y = x
```

Multiple top-level declarations are separated by use of `;`

```
I x = x;
K x y = x;
K1 x y = y
```

notice that __the last expression does not have a `;`__

#### The `main` value

when writing a program (not an expression that is run in the interpreter),
the execution starts with a top level function (supercombinator) called as 
`main`.


#### Expressions

Expressions can be one of the given alternatives. Note that `lambda` and
`case` are missing, since they are difficult to implement in this style of
machine. More is talked about this in the section [Lack of Lambda and Case](#lack-of-lambda-and-case).

    - **Let**

        ```haskell
        let <...bindings...> in <expr>
        ```

        Let bindings can be recursive and can refer to each other

    - **Function application**

        ```haskell
        function <args>
        ```

        Like Haskell's function application. The `<args>` are primitive values or
        variables.

        All n-ary application are represented by nested 1-ary applications.
        Functions are curried by default.

        ```haskell
        f x y z == (((f x) y) z)
        ```


    - **Data Declaration**

        ```haskell
        Pack{<tag>, <arity>}
        ```

        The `Pack` primitive operation takes a tag and an arity. When used, it packages
        up an `arity` number of expressions into a single object and tags it with `tag`.

        Example:
        ```haskell
        False = Pack{0, 0}
        True = Pack{1, 0}
        ```

        `True` and `False` are represented as `1` tagged and `0` tagged objects that
        have arity `0`.

        ```haskell
        MkPair = Pack{2, 2}
        my_tuple = MkPair 42 -42
        ```

        `MkPair`, a function used to create tuples uses a tag of `2` and requires two
        arguments. `my_tuple` is now a data node that holds the values `42` and `-42`.

        **NOTE:** using custom tags will not be very beneficial since the language does
        not have `case` expressions. Rather, `List` and `Tuple` are created as language
        inbuilts with custom de-structuring functions called `caseList` and `casePair`
        respectively.


    - **Primitive application**

        ```haskell
        <arg1> primop <arg2>
        ```

        Primitive operation on integers.

        The following operations are supported:

          - Arithmetic
            - `+`: addition
            - `-`: subtraction
            - `*`: multiplication
            - `/`: integer division

          - Boolean, returning `True` (`Pack{1, 0}`) for truth and `False` (`Pack{0, 0}`)
            for falsehood:
            `<`, `<=`, `==`, `/=`, `>=`, `>`

    - **Primitive literal**
        An integer declaration.

    - **Booleans**
        ```
        True = Pack{1, 0}
        False = Pack{0, 0}
        ```

        `True` and `False` are represented by `1` tagged and `0` tagged data
        types. 

    - **Tuples**
        Tuples are a language inbuilt and are constructed by using `MkPair`.

        ```haskell
        MkPair <left> <right>
        ```

        Tuples are pattern matched on by using `casePair`
        ```haskell
        casePair (MkPair a b) f = f a b
        ```

        Note that the default `fst` and `snd` are defined as follows
        ```haskell
        K x y = x;
        K1 x y = y;
        fst t = casePair t K;
        snd t = casePair t K1;
        ```

    - ** Lists
        Lists are language inbuilts and have two constructors: `Nil` and `Cons`        

        ```
        Nil
        Cons <value> <list>
        ```

        Lists are pattern matched by using

        ```haskell
        caseList <nil-handler> <cons-handler>
        ```
        `nil-handler` is a value
        `cons-handler` is a function that takes 2 parameters, the value in the
        `Cons` cell and the rest of the list.

#### Lack of Lambda and Case

##### Case
`Case` requires us to have some notion of
pattern matching / destructuring which is not present in this machine.


##### Lambda
`Lambda` needs a way to control scope. However, there is no real "scope" in
this machine, since the only time variables are bound is:

* during template instantiation
* let bindings

I think it is possible to implement lambdas, but I'm not super sure. If someone knows,
please drop me an e-mail or a pull request!




## Runtime

functions are curried by default. Thus `(f x y z)` is actually `(((f x) y) z)`

#### Components of the machine
The runtime has 4 components:
- **Heap**: a map from addresses to Heap Nodes
- **Stack**: a stack of Heap Addresses
- **Dump**: a stack of stacks to hold intermediate evaluations
- **Globals**: a map from names to addresses

Everything the machine uses during runtime must be allocated on the heap
before the machine starts executing. So, we need a way to convert a `CoreExpr`
into a `Heap`.

##### Example: Sample program `1 + 1`
Consider the program `1 + 1`. The initial state of the machine is
```
>1 + 1
*** ITERATION: 1
Stack - 1 items
## top ##
 0x21  ->  ((+ 1) 1)  H-Ap(0x1F $ 0x20)
## bottom ##
Heap - 34 items
 0x21  ->  ((+ 1) 1)  H-Ap(0x1F $ 0x20)
 0x1F  ->  (+ 1)      H-Ap(0xE $ 0x1E)
 0x1E  ->  1          H-Num(1)
 0xE   ->  +          H-Primitive(+)
 0x20  ->  1          H-Num(1)
Dump
  Empty
Globals - 30 items
 +  ->  0xE
```

Every single part of the expression `1 + 1` is on the heap, and the
symbol `+` is mapped to its address `0xE` in the `Globals` section. The
whole expression sits on top of the stack, waiting to be evaluated.


##### Example: `(((S K) K) 3)`

Consider the definitions:

```
S f g x = f x (g x)
K x y = x
```
(these are the `S` and `K` combinators from lambda calculus)

Now, let us understand how the program `S K K 3` evaluates.

```
*** ITERATION: 1
Stack - 1 items
## top ##
 0x21  ->  (((S K) K) 3)  H-Ap(0x1F $ 0x20)
## bottom ##
...
===///===
```
Initially, the code that we want to run `(((S K) K) 3)` is on the top of the stack.

```haskell
*** ITERATION: 2
Stack - 2 items
## top ##
 0x1F  ->  ((S K) K)      H-Ap(0x1E $ 0x1)
 0x21  ->  (((S K) K) 3)  H-Ap(0x1F $ 0x20)
## bottom ##
...
===///===
```
Remember that all application is always curried. That is `(((S K) K) 3)` is thought of
as `(((S K) K)` applied on `3`.


The LHS of the function application `((S K) K)`
is pushed on top of the current stack. This process continues till there
is a supercombinator on the top of the stack.


```haskell
*** ITERATION: 3
Stack - 3 items
## top ##
 0x1E  ->  (S K)          H-Ap(0x3 $ 0x1)
 0x1F  ->  ((S K) K)      H-Ap(0x1E $ 0x1)
 0x21  ->  (((S K) K) 3)  H-Ap(0x1F $ 0x20)
## bottom ##
...
===///===
*** ITERATION: 4
Stack - 4 items
## top ##
 0x3   ->  S              H-Supercombinator(S f g x = { ((f $ x) $ (g $ x)) })
 0x1E  ->  (S K)          H-Ap(0x3 $ 0x1)
 0x1F  ->  ((S K) K)      H-Ap(0x1E $ 0x1)
 0x21  ->  (((S K) K) 3)  H-Ap(0x1F $ 0x20)
## bottom ##
Heap - 34 items
 0x1F  ->  ((S K) K)      H-Ap(0x1E $ 0x1)
 0x1E  ->  (S K)          H-Ap(0x3 $ 0x1)
 0x1   ->  K              H-Supercombinator(K x y = { x })
 0x3   ->  S              H-Supercombinator(S f g x = { ((f $ x) $ (g $ x)) })
 0x20  ->  3              H-Num(3)
 0x21  ->  (((S K) K) 3)  H-Ap(0x1F $ 0x20)
===///===
```

Look at the current state of the stack.
T he left argument of the application keeps getting pushed onto the stack. 
This continues till there is a supercombinator on the top of the stack.

This process is called as *unwinding the spine* of the function call.

#### Instantiation
Now that a supercombinator (`S`) is on the top of the stack, we need to actually
apply it by passing the arguments. At this stage, the "spine is unwound".

- The top entry of the stack (`S`) is popped off to be evaluated.
- Since `S` takes 3 parameters (`f`, `g`, and `x`), 3 more entries are popped off
- The arguments to `S` are taken from the popped off elements.
    - The argument of the 1st application(`(S K)` at `0x1E`) becomes `f`
    - The argument of the 2nd application (`(S K) K` at `0x1F`) becomes `g`
    - The argument of the 3rd application (`((S K) K ) 3)` at `0x21`) becomes `3`

So, summarizing the current stage:
- `S`: supercombinator to unwind
- `K`: first parameter, `f`
- `K`: second parameter, `g`
- `3`: third parameter, `x`


Next, in **iteration 5** we push onto an empty stack the
body of the supercombinator, with variables replaced.

```haskell
*** ITERATION: 5
Stack - 1 items
## top ##
 0x24  ->  ((K 3) (K 3))  H-Ap(0x22 $ 0x23)
## bottom ##
Heap - 37 items
 0x24  ->  ((K 3) (K 3))  H-Ap(0x22 $ 0x23)
 0x1   ->  K              H-Supercombinator(K x y = { x })
 0x20  ->  3              H-Num(3)
 0x23  ->  (K 3)          H-Ap(0x1 $ 0x20)
 0x22  ->  (K 3)          H-Ap(0x1 $ 0x20)
 ...
 ===///===
```

Notice that the parameters for `S` have now been **instantiated** on the heap.
This is why it is called as an "instantiation machine" - it expands supercombinators
by **instantiating** parameters on the heap.


- `f` (`K`) is at `0x22`
- `g` (`K`) is at `0x23`
- `x` (`3`) is at `0x20`


#### How does evaluation provide laziness?

First, we shall make some observations about the evaluation process:

- During supercombinator expansion, only variables that are used are instantiated
- parameters are not evaluated, only replaced in function bodies.

Hence, we can state that:
- Evaluation occurs from the **outside in**

this is true because of the way in which application is unwound:

```haskell
*** ITERATION: 7
Stack - 3 items
## top ##
 0x1   ->  K              H-Supercombinator(K x y = { x })
 0x22  ->  (K 3)          H-Ap(0x1 $ 0x20)
 0x24  ->  ((K 3) (K 3))  H-Ap(0x22 $ 0x23)
## bottom ##
```

Notice that the `K` which is the most "outside" part of the expression `((K 3) (K 3))`
gets evaluated first.

When `K` is expanded, it expands like so:

```
*** ITERATION: 8
Stack - 1 items
## top ##
 0x20  ->  3  H-Num(3)
## bottom ##
```

The second parameter to `((K 3 (K 3))`, the `(K 3)` is never even evaluated! the `3`
is replaced as `x` in the body of `K x y = x`.

Thus, laziness is achieved by evaluating from the outside-in, and only replacing
function bodies without evaluating arguments. 

#### Primitives

`+`, `-`, etc. are similar in some ways - they also follow the
same model of unwinding the spine of the execution.

```
>1 + 1
*** ITERATION: 1
Stack - 1 items
## top ##
 0x31  ->  ((+ 1) 1)  H-Ap(0x2F $ 0x30)
## bottom ##
===///===
*** ITERATION: 2
Stack - 2 items
## top ##
 0x2F  ->  (+ 1)      H-Ap(0xE $ 0x2E)
 0x31  ->  ((+ 1) 1)  H-Ap(0x2F $ 0x30)
## bottom ##
===///===
*** ITERATION: 3
Stack - 3 items
## top ##
 0xE   ->  +          H-Primitive(+)
 0x2F  ->  (+ 1)      H-Ap(0xE $ 0x2E)
 0x31  ->  ((+ 1) 1)  H-Ap(0x2F $ 0x30)
## bottom ##
===///===
*** ITERATION: 4
Stack - 1 items
## top ##
 0x31  ->  2  H-Num(2)
## bottom ##
===///===
=== FINAL: "2" ===
```


Computing something like `(I 3) + 1` is not as straightforward, since
`I 3` now needs to be evaluated before the `+` can be evaluated. the section [The Dump](#the-dump)
explains this process.

#### Indirection

When we instantiate a supercombinator, we do not cache the results of an application.
Function application is optimized by rewriting the value of the
application node with the result obtained. This caches the computation.
This is what `Indirection` nodes do - they redirect a heap address to
another address.

We will consider the example where we define `x = I 3` where `I x = x`.

```
>define x = I 3
>x
*** ITERATION: 1
Stack - 1 items
## top ##
 0x22  ->  x  H-Supercombinator(x = { (I $ n_3) })
## bottom ##
...
===///===
*** ITERATION: 2
Stack - 1 items
## top ##
 0x25  ->  (I 3)  H-Ap(0x0 $ 0x24)
## bottom ##
...
===///===
*** ITERATION: 3
Stack - 2 items
## top ##
 0x0   ->  I      H-Supercombinator(I x = { x })
 0x25  ->  (I 3)  H-Ap(0x0 $ 0x24)
## bottom ##
...
===///===
*** ITERATION: 4
Stack - 1 items
## top ##
 0x24  ->  3  H-Num(3)
## bottom ##
...
===///===
=== FINAL: "3" ===
```

Now that we have run `x` once, let us re-run it and see what the value is

```
>x
*** ITERATION: 1
Stack - 1 items
## top ##
 0x22  ->  indirection(indirection(3))  H-Indirection(0x25)
## bottom ##
Heap - 39 items
 0x22  ->  indirection(indirection(3))  H-Indirection(0x25)
 0x25  ->  indirection(3)               H-Indirection(0x24)
 0x24  ->  3                            H-Num(3)
...
===///===
*** ITERATION: 2
Stack - 1 items
## top ##
 0x25  ->  indirection(3)  H-Indirection(0x24)
## bottom ##
...
===///===
*** ITERATION: 3
Stack - 1 items
## top ##
 0x24  ->  3  H-Num(3)
## bottom ##
Heap - 39 items
 0x24  ->  3  H-Num(3)
...
===///===
=== FINAL: "3" ===
>
```

Notice that the value of `x` has now become an indirection to `0x25` that used
to hold (`I 3`).

`0x25` is an indirection the value of `I 3`, which is `3` (at `0x24`).

This way, the value of `I 3` is not evaluated. It re-routes to `3`.

#### The Dump

Now that we've seen how function application works, we would like to understand
how primitives such as `+`, `-`, etc. work.

Let us consider the sample code `(I 1) + 3` where `I x = x` (Identity).

```
>I 1 + 3
*** ITERATION: 1
Stack - 1 items
## top ##
 0x2C  ->  ((+ (I 1)) 3)  H-Ap(0x2A $ 0x2B)
...
===///===
*** ITERATION: 2
Stack - 2 items
## top ##
 0x2A  ->  (+ (I 1))      H-Ap(0xE $ 0x29)
 0x2C  ->  ((+ (I 1)) 3)  H-Ap(0x2A $ 0x2B)
## bottom ##
...
===///===
*** ITERATION: 3
Stack - 3 items
## top ##
 0xE   ->  +              H-Primitive(+)
 0x2A  ->  (+ (I 1))      H-Ap(0xE $ 0x29)
 0x2C  ->  ((+ (I 1)) 3)  H-Ap(0x2A $ 0x2B)
## bottom ##
...
Dump
  Empty
===///===
```

we now have `+` on the top of the stack, but the LHS is a computation that needs
to be performed. Thus, we need to have some way of performing the computation.


The solution is to migrate the current stack into the Dump, and push `I 1` on top
of the stack and have it evaluate.

```
*** ITERATION: 4
Stack - 1 items
## top ##
 0x29  ->  (I 1)  H-Ap(0x0 $ 0x28)
## bottom ##
Heap - 45 items
 0x29  ->  (I 1)      H-Ap(0x0 $ 0x28)
 0x2A  ->  (+ (I 1))  H-Ap(0xE $ 0x29)
 0x0   ->  I          H-Supercombinator(I x = { x })
 0xE   ->  +          H-Primitive(+)
 0x28  ->  1          H-Num(1)
 0x2B  ->  3          H-Num(3)
Dump
## top ##
 0xE   ->  +              H-Primitive(+)
 0x2A  ->  (+ (I 1))      H-Ap(0xE $ 0x29)
 0x2C  ->  ((+ (I 1)) 3)  H-Ap(0x2A $ 0x2B)
## bottom ##
---
===///===
```

Notice how `I 1` is now on top of the stack and the Dump contains the previous
stack contents.

We proceed to see I 1 get evaluated.

```
*** ITERATION: 5
Stack - 2 items
## top ##
 0x0   ->  I      H-Supercombinator(I x = { x })
 0x29  ->  (I 1)  H-Ap(0x0 $ 0x28)
## bottom ##
Dump
## top ##
 0xE   ->  +              H-Primitive(+)
 0x2A  ->  (+ (I 1))      H-Ap(0xE $ 0x29)
 0x2C  ->  ((+ (I 1)) 3)  H-Ap(0x2A $ 0x2B)
## bottom ##
===///===
*** ITERATION: 6
Stack - 1 items
## top ##
 0x28  ->  1  H-Num(1)
## bottom ##
Dump
## top ##
 0xE   ->  +                       H-Primitive(+)
 0x2A  ->  (+ indirection(1))      H-Ap(0xE $ 0x29)
 0x2C  ->  ((+ indirection(1)) 3)  H-Ap(0x2A $ 0x2B)
## bottom ##
===///===
```

Notice how in _Iteration 6_, the rewrite of the `I 1` at `0x2A` also causes
the stack to change. The stack now has

```
0x2A  ->  (+ indirection(1))      H-Ap(0xE $ 0x29)
```

while at _Iteration 5_ had

```
0x2A  ->  (+ (I 1))      H-Ap(0xE $ 0x29)
```

This allows the `+` execution to "pick up" the value of 1 later. The 
rewriting is _essential_ to this evaluation. It allows the dumped stack
to get the output of the execution of `I 3`.

The stack now has one element `1`. Nothing is left to be evaluated.
So, we know that the value of `(I 1)` is `1`.

We have the rest of the computation in the Dump which we bring back.

```
*** ITERATION: 7
Stack - 3 items
## top ##
 0xE   ->  +                       H-Primitive(+)
 0x2A  ->  (+ indirection(1))      H-Ap(0xE $ 0x29)
 0x2C  ->  ((+ indirection(1)) 3)  H-Ap(0x2A $ 0x2B)
## bottom ##
Dump
===///===
*** ITERATION: 8
Stack - 1 items
## top ##
 0x2C  ->  ((+ 1) 3)  H-Ap(0x2A $ 0x2B)
## bottom ##
Dump
  Empty
===///===
```

At _Iteration 7_, the stack has
```
 0x2A  ->  (+ indirection(1))      H-Ap(0xE $ 0x29)
```
We remove the indirection by "short circuiting" the indirection and replacing
it with the value we want.


```
*** ITERATION: 9
Stack - 2 items
## top ##
 0x2A  ->  (+ 1)      H-Ap(0xE $ 0x28)
 0x2C  ->  ((+ 1) 3)  H-Ap(0x2A $ 0x2B)
## bottom ##
===///===
*** ITERATION: 10
Stack - 3 items
## top ##
 0xE   ->  +          H-Primitive(+)
 0x2A  ->  (+ 1)      H-Ap(0xE $ 0x28)
 0x2C  ->  ((+ 1) 3)  H-Ap(0x2A $ 0x2B)
## bottom ##
===///===
*** ITERATION: 11
Stack - 1 items
## top ##
 0x2C  ->  4  H-Num(4)
## bottom ##
Dump
  Empty
===///===
=== FINAL: "4" ===
```

Now that we have a simple expression, evaluation proceeds as usual, ending
with the machine evaluating `1 + 3` on seeing `+` at the top of the stack.


## Roadmap
- [x] Mark 1 (template instantiation)
- [x] let, letrec
- [x] template updates (do not naively instantiate each time)
- [x] numeric functions
- [x] Booleans
- [x] Tuples
- [x] Lists
- [x] nicer interface for stepping through execution


### Design Decisions


`TIMi` is written in Rust because:

- Rust is a systems language, so it allows for more control over memory, references, etc.
  which I enjoy.
- Rust has nice libraries for `readline`, table printing, and a slick `stdlib` for
  pretty code


### Things Learnt

##### Difference between `[..]` and `&[..]`

[Slice without ref](https://github.com/bollu/TIM-template-instantiation/blob/master/src/main.rs#L1124)
versus
[Slice with ref](https://github.com/bollu/TIM-template-instantiation/blob/d8515212f899ad185bec4bd1812bd493322b8d5d/src/main.rs#L1163)

the difference is that the second slice `[..]` maintains length information which it needs
at compile-time.

### References
- [Implementing Functional languages, a tutorial](http://research.microsoft.com/en-us/um/people/simonpj/Papers/pj-lester-book/)
- A huge thanks to [quchen's `STGi` implementation](https://github.com/quchen/stgi)
  whose style of documentation I copied for this machine.