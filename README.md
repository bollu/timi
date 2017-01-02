Graph Reduction
===============

An implementation of a graph reducer from the excellent book
[Implementing Functional languages, a tutorial](http://research.microsoft.com/en-us/um/people/simonpj/Papers/pj-lester-book/)

It is written in Rust because I like the fact that
- Rust is a systems language, so it'll hopefully be faster than an implementation in Haskell
- Rust is strict, which means that implementing certain things like letrec needs some more elbow grease
- Rust is well thought out as a language

## Roadmap
- [x] Mark 1 (template instantiation)
- [ ] let, letrec
- [ ] template updates (do not stupidly instatiate each time)
- [ ] numeric functions
- [ ] Pratt Parser for core
- [ ] nicer interface for stepping through execution
