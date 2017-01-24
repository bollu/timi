TIMmy (Template Instantiation Machine)
===============

A visual implementation of a template instantiation machine, to understand how a 
lazily evaluated functional programming language works

![Example run of template instantiation machine Run](http://i.imgur.com/PKNywHC.png)

# Table of Contents
- [Quickstart](#quickstart)
- [Language Introduction](#language-introduction)
- [Runtime](#runtime)
- [Roadmap](#roadmap)
- [Design Decisions](#design-decisions)
- [Stuff I Learnt](#stuff-I-learnt)
- [References](#references)

## Quickstart

## Language Introduction


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
- [ ] write higher order functions in Rust to help with implementing structured data
- [ ] Rust docs for `Supercombinator`
- [ ] Rust docs for `Update`
- [ ] Rust docs for `Primitive`



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
