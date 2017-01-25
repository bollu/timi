[![Build Status](https://travis-ci.org/bollu/TIM-template-instantiation.svg?branch=master)](https://travis-ci.org/bollu/TIM-template-instantiation)
[![Coverage Status](https://coveralls.io/repos/github/bollu/TIM-template-instantiation/badge.svg?branch=master)](https://coveralls.io/github/bollu/TIM-template-instantiation?branch=master)
[![Crates.io](https://img.shields.io/crates/v/timi.svg)](https://crates.io/crates/timi)


TIMi - Template Instantiation Machine Interpreter 
=================================================

A visual, user-friendly implementation of a template instantiation machine to understand how a 
lazily evaluated functional programming language is evaluated.

[![asciicast](https://asciinema.org/a/33a5xa2rcglfw1ff6hv6yqu84.png)](https://asciinema.org/a/33a5xa2rcglfw1ff6hv6yqu84)


# Table of Contents
- [Quickstart](#quickstart)
- [Interpreter Options & Usage](#interpreter-options)
- [Language Introduction](#language-introduction)
    - [Top level](#top-level)
    - [Expressions](#expressions)
    - [Lack of Lambda & Case](#lack-of-lambda-and-case)
    - [Code Examples](#code-examples)
- [Runtime](#runtime)
    - [Instantiation](#instantiation)
    - [Evaluation](#evaluation)
    - [Supercombinators](#supercombinators)
    - [Primitive](#supercombinators)
    - [Data Constructors](#supercombinators)
- [Roadmap](#roadmap)
- [Design Decisions](#design-decisions)
- [Stuff I Learnt](#stuff-I-learnt)
- [References](#references)

## Quickstart

#### Binary from  `cargo`
To quickly get the interpreter `timi` if you have `cargo` (Rust's package manager), run
```bash
$ cargo install timi && timi
```

#### Build from source
run
```bash
$ git clone https://github.com/bollu/timi.git && cd timi && cargo run
```


#### Using the interpreter

Type out expressions if you want them to be evaluated. For example:
```
> 1 + 1
```
will cause `1 + 1` to be evaluated


Use `let <id> [<param>]* = <expr>` to create function bindings
```
> let plus x y = x + y
> plus 1 1
```
will create a function called `plus` that takes two parameters `x` and `y`


## Interpreter Options & Usage


#### `>:step`
To go through the execution step-by-step, use
```
>:step

On entering an expression in this mode, the prompt will change to
```
{step of execution}>>
```

- Use `>>s` (for `step`) or `>>n` (for `next`) to go to the next step


#### `>:nostep`
to enable continuous execution of the entire program, use
```
>:nostep
```

## Language Introduction

The language can be seen as a reduced Haskell-like language. It is called `Core`,
both in `TIMi` and in `GHC (Glasgow haskell compiler)`, as it forms the minimal
subset for interpretation.

#### Top Level
#### Expressions
#### Lack of Lambda & Case
#### Code Examples

## Runtime

## Roadmap
- [x] Mark 1 (template instantiation)
- [x] let, letrec
- [x] template updates (do not stupidly instatiate each time)
- [x] numeric functions
- [x] Booleans
- [x] Tuples
- [x] Lists
- [x] nicer interface for stepping through execution


### Design Decisions

`TIM` is written in Rust because:
- Rust is a systems language, so it'll hopefully be faster than an implementation in Haskell
- Rust is strict, which means that implementing certain things like letrec needs some more elbow grease
- Rust has nice libraries (and a slick `stdlib`) that let you write safe and pretty code

##### Why does `peek()` return `PeekNoToken` instead of error?
in a lot of the parsing, we usually check if the next token is something.
if it isn't, we just return immediately.

Semantically, it makes sense, since you're just "peeking" at the next token,
so we can signal that there is no token by returning a sentinel token.

`try!(cursor.peek())` causes us to lose control flow and propogate
an __error__ if we peek at something that doesn't exist, which is the wrong
semantics. We want the user to be able to peek and make decisions based on
whether something is present ahead or not. `consume()` and `expect()` should
return errors since you're asking the cursor to go one token ahead.

### Stuff I learnt

##### Difference between `[..]` and `&[..]`

[Slice without ref](https://github.com/bollu/TIM-template-instantiation/blob/master/src/main.rs#L1124)
versus
[Slice with ref](https://github.com/bollu/TIM-template-instantiation/blob/d8515212f899ad185bec4bd1812bd493322b8d5d/src/main.rs#L1163)

the difference is that the second slice is being taken inside `tokenize`, which somehow maintains length info
(which is needed for `[..]` (since `[..]` means that you know the number of elements in it).

So, when it is outside, you need to take a slice `&[..]`, so that `Sized` information is not needed

### References
- [Implementing Functional languages, a tutorial](http://research.microsoft.com/en-us/um/people/simonpj/Papers/pj-lester-book/)
- A huge thanks to [quchen's `STGi` implementation](https://github.com/quchen/stgi) whose style I straight up copied for this machine.
