extern crate ansi_term;
extern crate rand;

use std::fmt;
use std::collections::HashMap;
use std::cmp; //for max
use ir::*;



#[derive(Clone, Debug)]
pub enum ParseError {
    NoTokens,
        UnexpectedToken {
            expected: Vec<CoreToken>,
            found: CoreToken
        },
        ErrorStr(String),

}

impl fmt::Display for ParseError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &ParseError::NoTokens => {
                write!(fmt, "no more tokens found")
            }
            &ParseError::UnexpectedToken{ref expected, ref found} => {
                write!(fmt, "expected one of {:#?}, \
                       found: |{:#?}|", expected, found)
            }
            &ParseError::ErrorStr(ref s) => write!(fmt, "{}",&s)
        }
    }
}

impl<T> Into<Result<T, ParseError>> for ParseError {
    fn into(self) -> Result<T, ParseError> {
        Result::Err(self)
    }

}


/*
struct DebugInfo<'a, T> {
    col: usize,
    line: usize,
    value: T,
    raw_program: &'a Vec<char>
}*/


#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum CoreToken {
    Let,
    In,
    Ident(String),
    Assignment,
    Semicolon,
    OpenRoundBracket,
    CloseRoundBracket,
    OpenCurlyBracket,
    CloseCurlyBracket,
    Comma,
    Integer(String),
    Or,
    And,
    L,
    LEQ,
    G,
    GEQ,
    EQ,
    NEQ,
    Plus,
    Minus,
    Mul,
    Div,
    Pack,
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
            CoreToken::PeekNoToken => ParseError::NoTokens.into(),
            other @ _ => {
                self.pos += 1;
                Result::Ok(other)
            }
        }

    }

    fn expect(&mut self, t: CoreToken) -> Result<(), ParseError> {
        let tok = self.peek();

        if tok == t {
            try!(self.consume());
            Result::Ok(())
        } else {
            Result::Err(ParseError::UnexpectedToken{
                expected: vec![t],
                found: tok
            })
        }
    }
}



struct TokenizerCursor {
    program: Vec<char>,
    line: usize,
    col: usize,
    i: usize
}

impl TokenizerCursor {
    fn new(program: String) -> TokenizerCursor {
        TokenizerCursor {
            program: program.chars().collect(),
            line: 0,
            col: 0,
            i: 0
        }
    }

    fn peek(&self) -> Option<&char> {
        self.program.get(self.i)
    }


    fn consume_longest_match(&mut self, matches: Vec<&str>) -> Result<String, ParseError> {

        let longest_len = matches.iter()
        .map(|s| s.len())
        .fold(0, cmp::max);


        //take only enough to not cause an out of bounds error
        let length_to_take = cmp::min(longest_len,
                                      self.program.len() - self.i);

        //take all lengths, starting from longest,
        //ending at shortest
        let mut longest_match : Option<String> = None;

        for l in (1..length_to_take+1).rev() {
            let possible_match : String = self.program[self.i..self.i + l].iter().cloned().collect();
            if matches.contains(&possible_match.as_str()) {
                longest_match = Some(possible_match);
                self.i += l;
                break;
            }
        }

        //longest operator is tokenized
        match longest_match {
            Some(m) => Result::Ok(m),
            None => {
                ParseError::ErrorStr(format!(
                 "expected one of {:#?}, found none to match", matches)).into()
            }
        }

    }

    /// Consumes from the character stream as long as `pred` returns true.
    ///
    /// `pred` is given the string consumed so far and the current character. It
    /// is expected to return whether the current character should be consumed
    /// or not. 
    /// NOTE: on finding `EOF`, this returns whatever has been consumed so far
    fn consume_while<F>(&mut self, pred: F) -> String
    where F: Fn(&String, &char) -> bool {
        let mut accum = String::new();
        loop {
            let c = match self.peek() {
                Some(c) => c.clone(),
                None => return accum
            };
            
            if pred(&accum, &c) {
                let _ = self.consume();
                accum.push(c.clone());
            }
            else {
                break;
            }
        }
        accum
    }


    fn consume(&mut self) -> Result<(), ParseError> {
        let c = match self.peek() {
            Some(c) => c.clone(),
            None => return Result::Err(ParseError::ErrorStr("unexpected EOF when tokenising".to_string()))
        };
        self.i += 1;
        self.col += 1;

        if c == '\n' {
            self.col = 0;
            self.line += 1;
        };

        Result::Ok(())
    }


}

fn identifier_str_to_token(token_str: &str) -> CoreToken {
    match token_str {
        "let" => CoreToken::Let,
        "in" => CoreToken::In,
        "Pack" => CoreToken::Pack,
        other @ _ => CoreToken::Ident(other.to_string())
    }
}


fn is_char_space(c: &char) -> bool {
    *c == ' ' || *c == '\n' || *c == '\t'
}


fn tokenize_symbol(cursor: &mut TokenizerCursor) -> Result<CoreToken, ParseError> {


    let symbol_token_map: HashMap<&str, CoreToken> =
        [("=", CoreToken::Assignment),
    (";", CoreToken::Semicolon),

    ("(", CoreToken::OpenRoundBracket),
    (")", CoreToken::CloseRoundBracket),

    ("{", CoreToken::OpenCurlyBracket),
    ("}", CoreToken::CloseCurlyBracket),

    (",", CoreToken::Comma),
    ("|", CoreToken::Or),
    ("&", CoreToken::And),
    ("<", CoreToken::L),
    ("<=", CoreToken::LEQ),
    (">", CoreToken::G),
    (">=", CoreToken::GEQ),

    ("!=", CoreToken::NEQ),
    ("==", CoreToken::EQ),
    //arithmetic
    ("+", CoreToken::Plus),
    ("-", CoreToken::Minus),
    ("*", CoreToken::Mul),
    ("/", CoreToken::Div)]
        .iter().cloned().collect();


    let keys : Vec<&str> = symbol_token_map.clone()
                                .keys()
                                .map(|&s| s.clone())
                                .collect();
    let op_str = try!(cursor.consume_longest_match(keys));

    let tok = symbol_token_map
                    .get(&op_str.as_str())
                    .expect(&format!("expected symbol for string: {}", op_str));
    Result::Ok(tok.clone())
}

fn eat_whitespace(cursor: &mut TokenizerCursor) {
    cursor.consume_while(|_, c| is_char_space(c));
}


fn is_ident_char(c: &char) -> bool {
    c.is_alphanumeric() || *c == '?' || *c == '-' || *c == '_'
}
 


fn tokenize(program: String) -> Result<Vec<CoreToken>, ParseError> {

    //let char_arr : &[u8] = program.as_bytes();
    //let char_arr : Vec<char> = program.clone().chars().collect();
    let mut cursor = TokenizerCursor::new(program);

    let mut tokens = Vec::new();

    loop {
        //consume spaces
        eat_whitespace(&mut cursor);

        //break out if we have exhausted the loop
        let peek = match cursor.peek() {
            None => break,
            Some(&c) => c
        };


        //alphabet: parse literal
        if peek.is_alphabetic() {

            //get the identifier name
            let mut id_string = String::new();
            id_string.push(peek);

            try!(cursor.consume());
            id_string += &cursor.consume_while(|_, c| is_ident_char(c));

            tokens.push(identifier_str_to_token(&id_string));
        }
        else if peek.is_numeric() {
            //parse the number
            //TODO: take care of floats
            let num_string = cursor.consume_while(|_, c| c.is_numeric());
            tokens.push(CoreToken::Integer(num_string));

        }
        else {
            let symbol= try!(tokenize_symbol(&mut cursor));
            tokens.push(symbol);
        }
    }

    Result::Ok(tokens)

}

fn parse_string_as_int(num_str: String) -> Result<i32, ParseError> {
    i32::from_str_radix(&num_str, 10)
        .map_err(|_| ParseError::ErrorStr(format!(
                    "unable to parse {} as int", num_str)))
}


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
fn parse_atomic_expr(mut c: &mut ParserCursor) ->
Result<CoreExpr, ParseError> {
    match c.peek() {
        CoreToken::Integer(num_str) => {
            try!(c.consume());
            let num = try!(parse_string_as_int(num_str));

            Result::Ok(CoreExpr::Num(num))
        },
        CoreToken::Ident(ident) => {
            try!(c.consume());
            Result::Ok(CoreExpr::Variable(ident))
        },
        CoreToken::OpenRoundBracket => {
            try!(c.expect(CoreToken::OpenRoundBracket));
            let inner_expr = try!(parse_expr(&mut c));
            try!(c.expect(CoreToken::CloseRoundBracket));
            Result::Ok(inner_expr)
        },
        other @ _ =>
            return Result::Err(ParseError::ErrorStr(format!(
                        "expected integer, identifier or (<expr>), found {:#?}",
                        other)))
    }

}

//defn := <ident> "=" <expr>
fn parse_defn(mut c: &mut ParserCursor) ->
Result<(CoreVariable, Box<CoreExpr>), ParseError> {

    if let CoreToken::Ident(name) = c.peek() {
        try!(c.consume());
        try!(c.expect(CoreToken::Assignment));

        let rhs : CoreExpr = try!(parse_expr(&mut c));
        Result::Ok((name, Box::new(rhs)))

    }
    else {
        return Result::Err(ParseError::ErrorStr(format!(
                    "variable name expected at defn, found {:#?}", c.peek())));
    }
}

//<let> := "let" <bindings> "in" <expr>
//<bindings> := <defn> | <defn> ";" <bindings>
fn parse_let(mut c: &mut ParserCursor) -> Result<CoreLet, ParseError> {
    //<let>
    match c.peek() {
        CoreToken::Let => try!(c.consume()),
        _ => return Result::Err(ParseError::ErrorStr(format!(
                    "expected let or letrec, found {:#?}", c.peek())))
    };

    let mut bindings : Vec<(Name, Box<CoreExpr>)> = Vec::new();

    //<bindings>
    loop {
        let defn = try!(parse_defn(&mut c));
        bindings.push(defn);

        //check for ;
        //If there is a ;, continue parsing
        if let CoreToken::Semicolon = c.peek() {
            try!(c.consume());
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

    Result::Ok(CoreLet {
        bindings: bindings,
        expr: Box::new(rhs_expr)
    })
}

//pack := Pack "{" tag "," arity "}"
fn parse_pack(mut c: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    try!(c.expect(CoreToken::Pack));
    try!(c.expect(CoreToken::OpenCurlyBracket));

    let tag : u32 = match c.peek() {
        CoreToken::Integer(s) => {
            try!(c.consume());
            try!(parse_string_as_int(s)) as u32
        }
        other @ _ => 
            return Result::Err(ParseError::ErrorStr(format!(
                        "expected integer tag, found {:#?}", other)))
    };

    try!(c.expect(CoreToken::Comma));

    let arity : u32 = match c.peek() {
        CoreToken::Integer(s) => {
            try!(c.consume());
            try!(parse_string_as_int(s)) as u32
        }
        other @ _ => 
            return Result::Err(ParseError::ErrorStr(format!(
                        "expected integer arity, found {:#?}", other)))
    };

    try!(c.expect(CoreToken::CloseCurlyBracket));
    Result::Ok(CoreExpr::Pack{tag: tag, arity: arity })


}


//aexpr := variable | number | Pack "{" num "," num "}" | "(" expr ")" 
fn parse_application(mut cursor: &mut ParserCursor) -> 
Result<CoreExpr, ParseError> {
    let mut application_vec : Vec<CoreExpr> = Vec::new();
    loop {
        let c = cursor.peek();
        //we have a "pack" expression
        if let CoreToken::Pack = c {
            let pack_expr = try!(parse_pack(&mut cursor));
            application_vec.push(pack_expr);
        } else if is_token_atomic_expr_start(c) {
            let atomic_expr = try!(parse_atomic_expr(&mut cursor));
            application_vec.push(atomic_expr);
        } else {
            break;
        }
    }

    if application_vec.len() == 0 {
        Result::Err(ParseError::ErrorStr(
                concat!("wanted function application or atomic expr ",
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
        for ap_rhs in application_vec.into_iter() {
            cur_ap_lhs = CoreExpr::Application(Box::new(cur_ap_lhs), Box::new(ap_rhs));
        }

        Result::Ok(cur_ap_lhs)
    }
}

fn parse_binop_at_precedence(mut cursor: &mut ParserCursor,
                             lhs_parse_fn: fn(&mut ParserCursor) -> Result<CoreExpr, ParseError>,
                             rhs_parse_fn: fn(&mut ParserCursor) -> Result<CoreExpr, ParseError>,
                             variable_bindings: HashMap<CoreToken, CoreExpr>) -> Result<CoreExpr, ParseError> {

    let lhs_expr : CoreExpr = try!(lhs_parse_fn(&mut cursor));

    let c : CoreToken = cursor.peek();

    let (rhs_expr, operator_variable) = {
        if let Some(&CoreExpr::Variable(ref op_str)) = variable_bindings.get(&c) {
            let op_var = CoreExpr::Variable(op_str.clone());
            try!(cursor.expect(c));
            let rhs = try!(rhs_parse_fn(&mut cursor));

            (rhs, op_var)
        }
        else {
            return Result::Ok(lhs_expr);
        }


    };

    let ap_inner =
        CoreExpr::Application(Box::new(operator_variable), Box::new(lhs_expr));

    Result::Ok(CoreExpr::Application(Box::new(ap_inner),
                                     Box::new(rhs_expr)))


}

fn parse_mul_div(mut cursor: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    parse_binop_at_precedence(cursor,
                              parse_application,
                              parse_mul_div,
                              [(CoreToken::Mul, CoreExpr::Variable("*".to_string())),
                              (CoreToken::Div, CoreExpr::Variable("/".to_string()))
                              ].iter().cloned().collect())
        //parse_application(&mut cursor)
}


fn parse_add_sub(mut cursor: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    parse_binop_at_precedence(cursor,
                              parse_mul_div,
                              parse_add_sub,
                              [(CoreToken::Plus, CoreExpr::Variable("+".to_string())),
                              (CoreToken::Minus, CoreExpr::Variable("-".to_string()))
                              ].iter().cloned().collect())
}

fn parse_relop(mut cursor: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    parse_binop_at_precedence(cursor,
                              parse_add_sub,
                              parse_relop,
                              [(CoreToken::L, CoreExpr::Variable("<".to_string())),
                              (CoreToken::LEQ, CoreExpr::Variable("<=".to_string())),
                              (CoreToken::G, CoreExpr::Variable(">".to_string())),
                              (CoreToken::GEQ, CoreExpr::Variable(">=".to_string())),
                              (CoreToken::EQ, CoreExpr::Variable("==".to_string())),
                              (CoreToken::NEQ, CoreExpr::Variable("!=".to_string()))
                              ].iter().cloned().collect())


}


//expr2 -> expr3 "&" expr2 | expr3
fn parse_and(mut cursor: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    parse_binop_at_precedence(cursor,
                              parse_relop,
                              parse_and,
                              [(CoreToken::And, CoreExpr::Variable("&".to_string()))
                              ].iter().cloned().collect())

}

//expr1 -> expr2 "|" expr1 | expr1
fn parse_or(mut cursor: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    parse_binop_at_precedence(cursor,
                              parse_and,
                              parse_or,
                              [(CoreToken::And, CoreExpr::Variable("|".to_string()))
                              ].iter().cloned().collect())
}




fn parse_expr(mut c: &mut ParserCursor) ->
Result<CoreExpr, ParseError> {
    match c.peek() {
        CoreToken::Let => parse_let(&mut c).map(|l| CoreExpr::Let(l)),
        _ => parse_or(&mut c)
    }
}


fn parse_supercombinator(mut c: &mut ParserCursor) ->
Result<SupercombDefn, ParseError> {

    let sc_name = match try!(c.consume()) {
        CoreToken::Ident(name) => name,
        other @ _ => return Result::Err(ParseError::ErrorStr(format!(
                        "super combinator name expected, {:#?} encountered",
                        other)))
    };

    let mut sc_args = Vec::new();

    //<args>* = <expr>
    while c.peek() != CoreToken::Assignment &&
        c.peek() != CoreToken::PeekNoToken {

        if let CoreToken::Ident(sc_arg) = c.peek() {
            try!(c.consume());
            sc_args.push(sc_arg);

        }
        else {
            return Result::Err(ParseError::ErrorStr(format!(
                                "super combinator argument expected, \
                                {:#?} encountered",
                                c.consume())));
            }
    }

    //take the equals
    try!(c.expect(CoreToken::Assignment));
    let sc_body = try!(parse_expr(&mut c));

    Result::Ok(SupercombDefn{
        name: sc_name,
        args: sc_args,
        body: sc_body
    })

}

pub fn string_to_expr(string : String) -> Result<CoreExpr, ParseError> {
    let tokens : Vec<CoreToken> = try!(tokenize(string));
    let mut cursor: ParserCursor = ParserCursor::new(tokens);

    parse_expr(&mut cursor)
}

pub fn string_to_sc_defn(string: String) -> Result<SupercombDefn, ParseError> {
    let tokens : Vec<CoreToken> = try!(tokenize(string));
    let mut cursor: ParserCursor = ParserCursor::new(tokens);
    parse_supercombinator(&mut cursor)    
}



pub fn string_to_program(string: String) -> Result<CoreProgram, ParseError> {

    let tokens : Vec<CoreToken> = try!(tokenize(string));
    let mut cursor: ParserCursor = ParserCursor::new(tokens);

    let mut program : CoreProgram = Vec::new();

    loop {
        //we need an identifier that is the supercombinator name
        match cursor.peek() {
            CoreToken::Ident(_)  => {}
            _ => return Result::Err(ParseError::ErrorStr(format!(
                        "super combinator name expected, {:#?} encountered",
                        cursor.consume())))   
        };

        let sc = try!(parse_supercombinator(&mut cursor));
        program.push(sc);

        match cursor.peek() {
            //we ran out of tokens, this is the last SC
            //break
            CoreToken::PeekNoToken => break,
            //we got a ;, more SCs to come
            CoreToken::Semicolon => {
                try!(cursor.expect(CoreToken::Semicolon));
                continue
            },
            other @ _ => {
                return Result::Err(ParseError::ErrorStr(format!(
                            "expected either ; or EOF, found {:#?}",
                            other)));
            }
        }

    }
    Result::Ok(program)
}