TIMmy (Template Instantiation Machine)
===============

An implementation of a template instantiation machine  from the excellent book
[Implementing Functional languages, a tutorial](http://research.microsoft.com/en-us/um/people/simonpj/Papers/pj-lester-book/)

It is written in Rust because I like the fact that
- Rust is a systems language, so it'll hopefully be faster than an implementation in Haskell
- Rust is strict, which means that implementing certain things like letrec needs some more elbow grease
- Rust is well thought out as a language

## Roadmap
- [x] Mark 1 (template instantiation)
- [x] let, letrec
- [x] template updates (do not stupidly instatiate each time)
- [x] numeric functions
- [ ] Booleans (WIP)
- [ ] nicer interface for stepping through execution


### Design Decisions

#### Why does `peek()` return `PeekNoToken` instead of error?
in a lot of the parsing, we usually check if the next token is something.
if it isn't, we just return immediately.

Semantically, it makes sense, since you're just "peeking" at the next token,
so we can signal that there is no token by returning a sentinel token.


`try!(cursor.peek())` causes us to lose control flow and propogate
an __error__ if we peek at something that doesn't exist, which is the wrong
semantics. We want the user to be able to peek and make decisions based on
whether something is present ahead or not. `consume()` and `expect()` should
return errors since you're asking the cursor to go one token ahead.
