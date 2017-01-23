
use std::fmt;
use std::collections::HashMap;
use std::cmp; //for max

use ir::*;


//*** parsing & tokensnisation ***

#[derive(Clone)]
pub enum ParseError {
    NoTokens,
        UnexpectedToken {
            expected: Vec<CoreToken>,
            found: CoreToken
        },
        ErrorStr(String),

}

impl fmt::Debug for ParseError {
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


//*** tokenisation ***

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum CoreToken {
    Let,
        LetRec,
        In,
        Case,
        Ident(String),
        Assignment,
        Semicolon,
        OpenRoundBracket,
        CloseRoundBracket,
        OpenCurlyBracket,
        CloseCurlyBracket,
        Comma,
        Integer(String),
        Lambda,
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
            CoreToken::PeekNoToken => Result::Err(ParseError::NoTokens),
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

fn identifier_str_to_token(token_str: &str) -> CoreToken {
    match token_str {
        "let" => CoreToken::Let,
        "letrec" => CoreToken::LetRec,
        "in" => CoreToken::In,
        "case" => CoreToken::Case,
        "Pack" => CoreToken::Pack,
        other @ _ => CoreToken::Ident(other.to_string())
    }
}


fn is_char_space(c: char) -> bool {
    c == ' ' || c == '\n' || c == '\t'
}

fn is_char_symbol(c: char) -> bool {
    !c.is_alphabetic() && !c.is_numeric()
}

fn tokenize_symbol(char_arr: Vec<char>, i: usize) -> 
Result<(CoreToken, usize), ParseError> {


    let c = match char_arr.get(i) {
        Some(c) => c.clone(),
        None => return 
            Result::Err(ParseError::ErrorStr(format!(
                        "unable to get value out of: {} from: {:?}", i, char_arr)))
    };
    assert!(is_char_symbol(c),
            format!("{} is not charcter, digit or symbol", c));

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
    ("/", CoreToken::Div),
    ("\\", CoreToken::Lambda)]
        .iter().cloned().collect();


    let longest_op_len = symbol_token_map
        .keys()
        .map(|s| s.len())
        .fold(0, cmp::max);


    //take only enough to not cause an out of bounds error
    let length_to_take = cmp::min(longest_op_len,
                                  char_arr.len() - i);

    //take all lengths, starting from longest,
    //ending at shortest
    let mut longest_op_opt : Option<CoreToken> = None;
    let mut longest_taken_length = 0;

    for l in (1..length_to_take+1).rev() {
        let op_str : &String = &char_arr[i..i + l]
            .iter()
            .cloned()
            .collect();

        if let Some(tok) = symbol_token_map.get(&op_str.as_str()) {
            //we found a token, break
            longest_taken_length = l;
            longest_op_opt = Some(tok.clone());
            break;
        }
    }

    //longest operator is tokenised
    let longest_op : CoreToken = match longest_op_opt {
        Some(op) => op,
        None => {
            let symbol = &char_arr[i..i + length_to_take];
            return Result::Err(ParseError::ErrorStr(format!(
                        "unknown symbol {:?}", symbol)))
        }
    };

    Result::Ok((longest_op, longest_taken_length))
        //tokens.push(longest_op);
        //i += longest_taken_length;
}



fn tokenize(program: String) -> Result<Vec<CoreToken>, ParseError> {

    //let char_arr : &[u8] = program.as_bytes();
    let char_arr : Vec<char> = program.clone().chars().collect();
    let mut i = 0;

    let mut tokens = Vec::new();

    loop {
        //break out if we have exhausted the loop
        if char_arr.get(i) == None {
            break;
        }

        //consume spaces
        while let Some(&c) = char_arr.get(i) {
            if !is_char_space(c) {
                break;
            }
            i += 1;
        }

        //we have a character
        if let Some(& c) = char_arr.get(i) {
            //alphabet: parse literal
            if c.is_alphabetic() {

                //get the identifier name
                let mut id_string = String::new();

                while let Some(&c) = char_arr.get(i) {
                    if c.is_alphanumeric() {
                        id_string.push(c);
                        i += 1;
                    } else {
                        break;
                    }
                }

                tokens.push(identifier_str_to_token(&id_string));
            }
            else if c.is_numeric() {
                //parse the number
                //TODO: take care of floats

                let mut num_string = String::new();

                while let Some(&c) = char_arr.get(i) {
                    if c.is_numeric() {
                        num_string.push(c);
                        i += 1;
                    } else {
                        break;
                    }
                }

                tokens.push(CoreToken::Integer(num_string));

            }
            else {
                let (symbol, stride) = try!(tokenize_symbol(char_arr.clone(), i));
                i += stride;
                tokens.push(symbol);
            }
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

//defn := <variable> "=" <expr>
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

//let := "let" <bindings> "in" <expr>
fn parse_let(mut c: &mut ParserCursor) -> Result<CoreLet, ParseError> {
    //<let>
    let let_token = match c.peek() {
        CoreToken::Let => try!(c.consume()),
        CoreToken::LetRec => try!(c.consume()),
        _ => return Result::Err(ParseError::ErrorStr(format!(
                    "expected let or letrec, found {:#?}", c.peek())))
    };

    let mut bindings : Vec<(Name, Box<CoreExpr>)> = Vec::new();

    //<bindings>
    loop {
        let defn = try!(parse_defn(&mut c));
        bindings.push(defn);

        //check for ;
        //If htere is a ;, continue parsing
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

    let is_rec : bool = match let_token {
        CoreToken::Let => false,
        CoreToken::LetRec => true,
        other @ _ =>
            return Result::Err(ParseError::UnexpectedToken {
                expected: vec![CoreToken::Let, CoreToken::LetRec],
                found: other.clone()
            })
    };

    Result::Ok(CoreLet {
        is_rec: is_rec,
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
        CoreToken::LetRec => parse_let(&mut c).map(|l| CoreExpr::Let(l)),
        CoreToken::Case => panic!("cannot handle case yet"),
        CoreToken::Lambda => panic!("cannot handle lambda yet"),
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