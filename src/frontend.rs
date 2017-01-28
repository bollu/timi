extern crate ansi_term;
extern crate rand;

use std::fmt;
use std::collections::HashMap;
use std::cmp; //for max
use ir::*;
use std::str;

/// represents a Point in a file
#[derive(Debug, Clone, Copy)]
struct Point {
    /// index is the raw seek index starting from 0
    index: usize,
    /// the line number of the location
    line: usize,
    /// the column number
    col: usize
}

impl Point {
    fn new(index: usize, line: usize, col: usize) -> Point {
        Point {
            index: index,
            line: line,
            col: col
        }
    }
    fn as_range(&self) -> Range {
        Range {
            start: *self,
            end: *self
        }
    }
}

/// represents a value of T associated with a start and an end Point
#[derive(Debug, Clone, Copy)]
struct Range {
    start: Point,
    end: Point,
}


pub enum ParseErrorKind {
    EOF(String),
    UnexpectedToken{ expected: Vec<CoreToken>, found: CoreToken, error: String },
    Generic(String)
}

impl fmt::Debug for ParseErrorKind {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &ParseErrorKind::EOF(ref s) => {
                write!(fmt, "EOF reached: {}", s)
            }
            &ParseErrorKind::UnexpectedToken{ref expected, ref found, ref error} => {
                write!(fmt, "expected one of {:#?}\n\
                       found: {:#?}\n\
                       error: {}",
                       expected, found, error)
            }
            &ParseErrorKind::Generic(ref err) => write!(fmt, "{}", err)
        }
    }
}

#[derive(Debug)]
pub struct ParseError(Range, ParseErrorKind);

impl ParseError {
    fn generic<T>(range: Range, s: String) -> Result<T, ParseError> {
        ParseError(range, ParseErrorKind::Generic(s)).into()
    }

    pub fn pretty_print(&self, program: &str) -> String {

        let &ParseError(range, ref error_kind) = self;
        let program_lines  : Vec<Vec<char>> = program
                                              .split('\n')
                                              .map(|s| s.chars().collect())
                                              .collect();

        let source_lines = {
            if range.start.line == range.end.line {
                ParseError::pretty_print_single_line(range.start.line,
                                                     range.start.col,
                                                     range.end.col, &program_lines[range.start.line])

            } else {
                ParseError::pretty_print_multiple_lines(range, &program_lines[range.start.line..range.end.line])
     
            }
        };

        format!("{}\n{:#?}", source_lines, error_kind)
    }

    fn get_line_number_pretty(line: usize) -> String {
        format!("{}| ", line + 1)
    }

    fn pretty_print_single_line(line: usize, col_begin: usize, col_end: usize, line_str: &Vec<char>) -> String{
        use std::iter;

        use self::ansi_term::Colour::{Blue, Red};

        let line_number_pretty = ParseError::get_line_number_pretty(line);

        let error_pointer_line = iter::repeat(" ").take(line_number_pretty.chars().count() +  col_begin - 1).collect::<String>() +
                                 "^" + 
                                &iter::repeat("-").take(col_end - (col_begin )).collect::<String>();



        let source_line = Blue.paint(line_number_pretty).to_string() +
                          &line_str.iter().cloned().collect::<String>();


        format!("{}\n{}", source_line, Red.paint(error_pointer_line))
    }


    fn pretty_print_multiple_lines(range: Range, lines_str: &[Vec<char>]) -> String{
        let mut out = String::new();
        for (i, line)  in lines_str.iter().enumerate() {
            out += &ParseError::get_line_number_pretty(range.start.line + i);
            out += &line.iter().cloned().collect::<String>();
            out += "\n";

        }
        out

    }
}

impl<T> Into<Result<T, ParseError>> for ParseError {
    fn into(self) -> Result<T, ParseError> {
        Result::Err(self)
    }


}



#[derive(Clone, PartialEq, Eq, Hash)]
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
}

impl fmt::Debug for CoreToken {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &CoreToken::Let => write!(fmt, "let"),
            &CoreToken::In => write!(fmt, "in"),
            &CoreToken::Ident(ref s) => write!(fmt, "{}", s),
            &CoreToken::Assignment => write!(fmt, "="),
            &CoreToken::Semicolon => write!(fmt, ";"),
            &CoreToken::OpenRoundBracket => write!(fmt, "("),
            &CoreToken::CloseRoundBracket => write!(fmt, ")"),
            &CoreToken::OpenCurlyBracket => write!(fmt, "{{"),
            &CoreToken::CloseCurlyBracket => write!(fmt, "}}"),
            &CoreToken::Comma => write!(fmt, ","),
            &CoreToken::Integer(ref s) => write!(fmt, "{}", s),
            &CoreToken::Or => write!(fmt, "||"),
            &CoreToken::And => write!(fmt, "&&"),
            &CoreToken::L => write!(fmt, "<"),
            &CoreToken::LEQ => write!(fmt, "<="),
            &CoreToken::G => write!(fmt, ">"),
            &CoreToken::GEQ => write!(fmt, ">="),
            &CoreToken::EQ => write!(fmt, "=="),
            &CoreToken::NEQ => write!(fmt, "!="),
            &CoreToken::Plus => write!(fmt, "+"),
            &CoreToken::Minus => write!(fmt, "-"),
            &CoreToken::Mul => write!(fmt, "*"),
            &CoreToken::Div => write!(fmt, "/"),
            &CoreToken::Pack => write!(fmt, "Pack"),
     }
 }
}


#[derive(Clone)]
struct ParserCursor {
    tokens: Vec<(Range, CoreToken)>,
    pos: usize,
    cur_range: Range,
}

impl ParserCursor {
    fn new(tokens: Vec<(Range, CoreToken)>) -> ParserCursor {
        let cur_range = match tokens.get(0) {
            Some(&(r, _)) => r,
            None => Point::new(0, 0, 0).as_range()
        };

        ParserCursor {
            tokens: tokens,
            pos: 0,
            cur_range:  cur_range
        }
    }

    fn peek(&self) -> (Range, Option<CoreToken>) {
        match self.tokens.get(self.pos) {
            Some(&(r, ref t)) => (r, Some(t.clone())),
            None => (self.cur_range, None)
        }

    }

    fn consume(&mut self, error: &str) -> Result<(Range, CoreToken), ParseError> {
        match self.peek() {
            (range, None) => ParseError(range, ParseErrorKind::EOF(error.to_string())).into(),
            (range, Some(token)) => {
                self.cur_range = range;
                self.pos += 1;
                Result::Ok((range, token))
            }
        }

    }

    fn expect(&mut self, t: CoreToken, error: &str) -> Result<(), ParseError> {
        match self.peek() {
            (range, None) => return ParseError(range, ParseErrorKind::EOF(format!(
                            "was expecting {:#?}, found EOF\n{}", t, error))).into(),
            (range, Some(tok)) => {
                if tok == t {
                    self.cur_range = range;
                    self.pos += 1; 
                    Result::Ok(())
                }
                else {
                    return ParseError(range, ParseErrorKind::UnexpectedToken{
                        expected: vec![t],
                        found: tok,
                        error: error.to_string()
                    }).into()
                }
            }
        }
    }
}



struct TokenizerCursor {
    program: Vec<char>,
    line: usize,
    col: usize,
    i: usize,
}

impl TokenizerCursor {
    fn new(program: &str) -> TokenizerCursor {
        TokenizerCursor {
            program: program.chars().collect(),
            line: 0,
            col: 0,
            i: 0
        }
    }

    fn peek(&self) -> Option<(Range, &char)> {
        self.program.get(self.i).map(|c| (self.point().as_range(), c))
    }


    /// returns the Point corresponding to the current cursor location
    fn point(&self) -> Point {
        Point {
            index: self.i,
            line: self.line,
            col: self.col
        }
    }


    fn range_till_cur(&self, start_point: Point) -> Range {
        Range {
            start: start_point,
            end: self.point(),
        }

    }

    fn consume_longest_match(&mut self, matches: Vec<&str>) -> Result<(Range, String), ParseError> {
        let start = self.point();

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
                longest_match = Some(possible_match.clone());

                for _ in 0..l {
                    try!(self.consume(&format!("consuming matching token: {}", possible_match)));
                }
                break;
            }
        }
        //longest operator is tokenized
        match longest_match {
            Some(m) => Result::Ok((self.range_till_cur(start), m)),
            None => {
                ParseError::generic(self.range_till_cur(start),
                                    format!("expected one of {:#?}, found none to match", matches))
            }
        }

    }

    /// Consumes from the character stream as long as `pred` returns true.
    ///
    /// `pred` is given the string consumed so far and the current character. It
    /// is expected to return whether the current character should be consumed
    /// or not. 
    /// NOTE: on finding `EOF`, this returns whatever has been consumed so far
    fn consume_while<F>(&mut self, pred: F, error: &str) -> (Range, String)
    where F: Fn(&String, &char) -> bool {
        let start = self.point();
        let mut accum = String::new();

        loop {
            let c = match self.peek() {
                Some((_, c)) => c.clone(),
                None => return (self.range_till_cur(start), accum)
            };
            
            if pred(&accum, &c) {
                let _ = self.consume(error);
                accum.push(c.clone());
            }
            else {
                break;
            }
        }
        (self.range_till_cur(start), accum)
    }


    fn consume(&mut self, error: &str) -> Result<(), ParseError> {
        let c = match self.peek() {
            Some((_, c)) => c.clone(),
            None => return ParseError::generic(self.point().as_range(), error.to_string())
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


fn tokenize_symbol(cursor: &mut TokenizerCursor) -> Result<(Range, CoreToken), ParseError> {


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
    let (range, op_str) = try!(cursor.consume_longest_match(keys));

    let tok = symbol_token_map
                    .get(&op_str.as_str())
                    .expect(&format!("expected symbol for string: {}", op_str));
    Result::Ok((range, tok.clone()))
}

fn eat_whitespace(cursor: &mut TokenizerCursor) {
    cursor.consume_while(|_, c| is_char_space(c), "eating whitespace");
}


fn is_ident_char(c: &char) -> bool {
    c.is_alphanumeric() || *c == '?' || *c == '-' || *c == '_'
}
 

/// eats comments till the end of the line
/// returns whether a comment was consumed or not
/// this is used to restart tokenization for the next line
/// if a comment was found
fn eat_comment(cursor: &mut TokenizerCursor) -> bool {
    match cursor.peek() {
        Some((_, &'#')) => { 
            cursor.consume_while(|_, c| *c != '\n',
                                "eating comment line");
            return true
        }
        Some(_) | None => return false
    };
}

fn tokenize(program: &str) -> Result<Vec<(Range, CoreToken)>, ParseError> {

    let mut cursor = TokenizerCursor::new(program);

    let mut tokens : Vec<(Range, CoreToken)> = Vec::new();

    loop {
        //consume spaces
        eat_whitespace(&mut cursor);

        //if a comment was eaten, restart tokenization for the next line
        if eat_comment(&mut cursor) {
            continue;
        }

        //break out if we have exhausted the loop
        let peek = match cursor.peek() {
            None => break,
            Some((_, &c)) => c
        };


        //alphabet: parse literal
        if peek.is_alphabetic() {

            //get the identifier name
            let mut id_string = String::new();
            let start = cursor.point();

            id_string.push(peek);

            try!(cursor.consume("consuming alphabet token"));
            let &(mut range, ref consumed_str) = &cursor.consume_while(|_, c| is_ident_char(c), "consuming identifier string");
            range.start = start;

            id_string += &consumed_str;

            tokens.push((range, identifier_str_to_token(&id_string)));
        }
        else if peek.is_numeric() {
            //parse the number
            //TODO: take care of floats
            let (range, num_string) = cursor.consume_while(|_, c| c.is_numeric(),
                                                           "trying to tokenize number");
            tokens.push((range, CoreToken::Integer(num_string)));

        }
        else {
            let symbol= try!(tokenize_symbol(&mut cursor));
            tokens.push(symbol);
        }
    }

    Result::Ok(tokens)

}

fn parse_string_as_int(range: Range, num_str: String) -> Result<i32, ParseError> {
    //FIXME: rewrite using map_err
    match i32::from_str_radix(&num_str, 10) {
        Result::Ok(num) => Result::Ok(num),
        Result::Err(_) => ParseError::generic(range,
                                        format!("unable to parse {} as int", num_str))
    }
}


//does this token allow us to start to parse an
//atomic expression?
fn is_token_atomic_expr_start(t: &CoreToken) -> bool {
    match t {
        &CoreToken::Integer(_) => true,
        &CoreToken::Ident(_) => true,
        &CoreToken::OpenRoundBracket => true,
        _ => false
    }

}

//atomic := <num> | <ident> | "(" <expr> ")"
fn parse_atomic_expr(mut c: &mut ParserCursor) ->
Result<CoreExpr, ParseError> {
    match try!(c.consume("parsing atomic expression")) {
        (range, CoreToken::Integer(num_str)) => {
            let num = try!(parse_string_as_int(range, num_str));
            Result::Ok(CoreExpr::Num(num))
        },
        (_, CoreToken::Ident(ident)) => {
            Result::Ok(CoreExpr::Variable(ident))
        },
        (_, CoreToken::OpenRoundBracket) => {
            let inner_expr = try!(parse_expr(&mut c));
            //TODO: create c.match() that prints errors about matching parens
            try!(c.expect(CoreToken::CloseRoundBracket, "looking for ) for matching ("));
            Result::Ok(inner_expr)
        },
        (range, other) =>
            return ParseError(range, ParseErrorKind::Generic(format!(
                        "expected integer, identifier or (<expr>), found {:#?}",
                        other))).into()
    }

}

//defn := <ident> "=" <expr>
fn parse_defn(mut c: &mut ParserCursor) ->
Result<(CoreVariable, Box<CoreExpr>), ParseError> {

    match try!(c.consume("looking for identifier of let binding LHS")) {
        (_, CoreToken::Ident(name)) => {
            try!(c.expect(CoreToken::Assignment, "expected = after name in let bindings"));
            let rhs : CoreExpr = try!(parse_expr(&mut c));
            Result::Ok((name, Box::new(rhs)))
        }
        (range, other) =>  {
            return  ParseError::generic(range, format!("expected LHS of let binding, found {:#?}", other))
        }
    }
}

//<let> := "let" <bindings> "in" <expr>
//<bindings> := <defn> | <defn> ";" <bindings>
fn parse_let(mut c: &mut ParserCursor) -> Result<CoreLet, ParseError> {
    //<let>
    try!(c.expect(CoreToken::Let, "expected let"));

    let mut bindings : Vec<(Name, Box<CoreExpr>)> = Vec::new();

    //<bindings>
    loop {
        println!("trying to parse defn...");
        let defn = try!(parse_defn(&mut c));
        bindings.push(defn);

        //check for ;
        //If there is a ;, continue parsing
        if let (_, Some(CoreToken::Semicolon)) = c.peek() {
            try!(c.expect(CoreToken::Semicolon, "expected ; since peek() returned ;"));
            continue;
        }
        else {
            break;
        }
    }
    //<in>
    try!(c.expect(CoreToken::In, "expected <in> after let bindings"));

    //<expr>
    let rhs_expr = try!(parse_expr(c));

    Result::Ok(CoreLet {
        bindings: bindings,
        expr: Box::new(rhs_expr)
    })
}

//pack := Pack "{" tag "," arity "}"
fn parse_pack(mut c: &mut ParserCursor) -> Result<CoreExpr, ParseError> {
    let pack_grammar = "pack := Pack \"{\" tag \",\" arity \"}\"";

    try!(c.expect(CoreToken::Pack, pack_grammar));
    try!(c.expect(CoreToken::OpenCurlyBracket, pack_grammar));

    let tag : u32 = match try!(c.consume(pack_grammar)) {
        (range, CoreToken::Integer(s)) => {
            try!(parse_string_as_int(range, s)) as u32
        }
        (range, other) => 
            return ParseError(range, ParseErrorKind::Generic(format!(
                        "expected integer tag, found {:#?}", other))).into()
    };

    try!(c.expect(CoreToken::Comma, "expected , after <tag> in Pack"));

    let arity : u32 = match try!(c.consume("expected arity after , in Pack")) {
        (range, CoreToken::Integer(s)) => {
            try!(parse_string_as_int(range, s)) as u32
        }
        (range, other) => 
            return ParseError::generic(range, format!(
                        "expected integer arity, found {:#?}", other))
    };

    //TODO: create cursor.match(..)
    try!(c.expect(CoreToken::CloseCurlyBracket, "expecting closing curly bracket for Pack"));
    Result::Ok(CoreExpr::Pack{tag: tag, arity: arity })


}


//aexpr := variable | number | Pack "{" num "," num "}" | "(" expr ")" 
fn parse_application(mut cursor: &mut ParserCursor) -> 
Result<CoreExpr, ParseError> {

    let mut application_vec : Vec<CoreExpr> = Vec::new();
    loop {

        //we have a "pack" expression
        match cursor.peek() {
            (_, Some(CoreToken::Pack)) => {
                let pack_expr = try!(parse_pack(&mut cursor));
                application_vec.push(pack_expr);

            }
            (_, Some(other)) => {
                if is_token_atomic_expr_start(&other) {
                    let atomic_expr = try!(parse_atomic_expr(&mut cursor));
                    application_vec.push(atomic_expr);
                }
                else {
                    break;
                }
            }
            _ => break
        };
    }

    if application_vec.len() == 0 {
        return ParseError::generic(cursor.cur_range,
                concat!("wanted function application or expression").to_string())

    }
    else if application_vec.len() == 1 {
        //just an atomic expr
        return Result::Ok(application_vec.remove(0))
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

    let c = match cursor.peek() {
        (_, None) => return Result::Ok(lhs_expr),
        (_, Some(c)) => c
    };

    let (rhs_expr, operator_variable) = {
        if let Some(&CoreExpr::Variable(ref op_str)) = variable_bindings.get(&c) {
            let op_var = CoreExpr::Variable(op_str.clone());
            try!(cursor.expect(c, &format!("parsing binary operator {:#?}", op_var)));
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
        (_, Some(CoreToken::Let)) => parse_let(&mut c).map(|l| CoreExpr::Let(l)),
        _ => parse_or(&mut c)
    }
}


fn parse_supercombinator(mut c: &mut ParserCursor) ->
Result<SupercombDefn, ParseError> {

    let sc_name = match try!(c.consume("parsing supercombinator, looking for name")) {
        (_, CoreToken::Ident(name)) => name,
        (range, other)=> return ParseError::generic(range, format!(
                        "super combinator name expected, {:#?} encountered",
                        other))
    };

    let mut sc_args = Vec::new();

    //<args>* = <expr>
    loop {
        match try!(c.consume("looking for <args>* =")) {
            (_, CoreToken::Ident(sc_arg)) => {
                sc_args.push(sc_arg);
            }
            (_, CoreToken::Assignment) => { break; }
            (range, other) => {
                return ParseError::generic(range, format!(
                                    "identifier that is supercombinator argument or \"=\" expected, \
                                    {:#?} encountered",
                                    other)).into();
            }
        }
    }

    let sc_body = try!(parse_expr(&mut c));

    Result::Ok(SupercombDefn{
        name: sc_name,
        args: sc_args,
        body: sc_body
    })

}

pub fn string_to_expr(string: &str) -> Result<CoreExpr, ParseError> {
    let tokens : Vec<(Range, CoreToken)> = try!(tokenize(string));
    let mut cursor: ParserCursor = ParserCursor::new(tokens);

    parse_expr(&mut cursor)
}

pub fn string_to_sc_defn(string: &str) -> Result<SupercombDefn, ParseError> {
    let tokens : Vec<(Range, CoreToken)> = try!(tokenize(string));
    let mut cursor: ParserCursor = ParserCursor::new(tokens);
    parse_supercombinator(&mut cursor)    
}



pub fn string_to_program(string: &str) -> Result<CoreProgram, ParseError> {

    let tokens : Vec<(Range, CoreToken)> = try!(tokenize(string));
    let mut cursor: ParserCursor = ParserCursor::new(tokens);

    let mut program : CoreProgram = Vec::new();

    loop {
        //we need an identifier that is the supercombinator name
        match cursor.peek() {
            (_, Some(CoreToken::Ident(_)))  => {}
            (range, Some(other)) => return ParseError(range, ParseErrorKind::Generic(format!(
                        "super combinator name expected, {:#?} encountered", other))).into(),
            (range, None) => return ParseError(range, ParseErrorKind::EOF("supercombinator name expected".to_string())).into()
        };

        let sc = try!(parse_supercombinator(&mut cursor));
        program.push(sc);

        match cursor.peek() {
            //we ran out of tokens, this is the last SC
            //break
            (_, None) => break,
            //we got a ;, more SCs to come
            (_, Some(CoreToken::Semicolon)) => {
                try!(cursor.expect(CoreToken::Semicolon, "parsing supercombinator"));
                continue
            },
            (range, Some(other)) => {
                return ParseError::generic(range, format!(
                            "expected either ; or EOF, found {:#?}",
                            other));
            }
        }

    }
    Result::Ok(program)
}