from lark import Lark
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.exceptions import UnexpectedCharacters, UnexpectedToken
from lark.lexer import PatternStr
from sentencepiece import SentencePieceProcessor
from typing import Literal, TypeAlias
from torch import tensor, Tensor
from regex import Pattern, compile as re_compile

FIXED_TOKEN_TYPES: tuple[str] = (
    "LBRACE", # {
    "RBRACE", # }
    "LSQB", # [
    "RSQB", # ]
    "LPAREN", # (
    "RPAREN", # )
    "COLON", # :
    "COMMA", # ,
    "TRUE", # true
    "FALSE", # false
    "NULL", # null
    "STR", # any string within double quotes
    "INT", # any signed/unsigned integer
    "FLOAT", # any signed/unsigned float
)
TokenType: TypeAlias = Literal[FIXED_TOKEN_TYPES]|str # type: ignore # Union with str to allow uppercase token types created by lark for any literals


class GrammarConstrainedParser:
    """A parser that constrains token acceptance based on a Lark grammar,
    with special handling for LLM tokenization mismatches."""
    
    def __init__(self, grammar: str, llm_tokenizer: SentencePieceProcessor):
        self.parser: Lark = Lark(grammar, parser="lalr", lexer="basic")
        self.interactive: InteractiveParser = self.parser.parse_interactive("")
        self.llm_tokenizer: SentencePieceProcessor = llm_tokenizer
        self.llm_tokenizer_vocab_size: int = llm_tokenizer.vocab_size()
        self.is_complete: bool = "$END" in self.interactive.accepts()
        self.parsed_str: str = ""
        
        # Track partial tokens being built
        self.current_token_types: set[str] = set() # non-empty only for STR, INT, FLOAT, or uppercase token types created by lark for any literals
        self.current_token_value: str = ""

    def _match_regex_prefix(self, prefix_string: str, regex_pattern: str) -> bool:
        """"Return True if `prefix_string` can be a prefix of any string that matches `regex_pattern`."""
        if regex_pattern[0] != '^':
            regex_pattern = "^" + regex_pattern
        pattern: Pattern = re_compile(regex_pattern)
        return bool(pattern.fullmatch(prefix_string, partial=True))
    
    
    def get_acceptable_llm_tokens(self) -> Tensor:
        """Check if the current parser state can accept this token."""
        
        acceptable_llm_tokens: list[int] = []

        if not self.current_token_types: # not currently accepting any partial tokens
            for id in range(self.llm_tokenizer_vocab_size): # for each llm token
                llm_token: str = self.llm_tokenizer.Decode(id) # decode the llm token
                added: bool = False
                # add any llm tokens that the grammar can accept as 0 or more (maximal) complete grammar tokens + prefix of any next grammar token
                for i in range(len(llm_token)+1):
                    interactive_copy: InteractiveParser = self.interactive.copy()
                    try:
                        # try to lex the partial llm token as 0 or more complete grammar tokens
                        for token in self.parser.lex(llm_token[:len(llm_token)-i]): # raises error if lexer fails
                            interactive_copy.feed_token(token) # raises error if parser fails

                        # try to lex the rest of the llm token as a prefix of any next grammar token
                        acceptable_token_types: list[TokenType] = interactive_copy.accepts()
                        for acceptable_token_type in acceptable_token_types:
                            last_token_regex_pattern: PatternStr = self.parser._terminals_dict[acceptable_token_type].pattern
                            if self._match_regex_prefix(llm_token[len(llm_token)-i:], last_token_regex_pattern.to_regexp()): # try to match the rest of the llm token as a prefix of any string that matches the regex pattern
                                acceptable_llm_tokens.append(id) # add llm token if everything passes
                                added = True
                                break
                    except UnexpectedCharacters:
                        pass
                    except UnexpectedToken:
                        pass
                    except Exception as e:
                        print(e)
                        pass
                    if added:
                        break
                            
        else: # self.current_token_type is empty means we're building a partial token
            for id in range(self.llm_tokenizer_vocab_size): # for each llm token 
                llm_token: str = self.llm_tokenizer.Decode(id) # decode the llm token
                added: bool = False
                concatenated: str = self.current_token_value + llm_token # concatenate the current partial token with the llm token
                # add any llm tokens that when concatenated to self.current_token_value, the grammar can accept as self.current_token_type + 0 or more (maximal) complete grammar tokens + prefix of any next grammar token or partially accept as any self.current_token_types
                for i in range(len(concatenated)+1):
                    interactive_copy: InteractiveParser = self.interactive.copy()
                    try:
                        # try to lex the partial concatenated string as self.current_token_type + 0 or more (maximal) complete grammar tokens
                        for j, token in enumerate(self.parser.lex(concatenated[:len(concatenated)-i])): # raises error if lexer fails
                            if j == 0:
                                assert token.type in self.current_token_types # assert that the first token was one of the current token types being built
                            interactive_copy.feed_token(token) # raises error is parser fails

                        # try to lex the rest of the concatenated string as a prefix of any next grammar token
                        # if nothing was lexed in the above for loop, try to partially accept entire concatenated string as prefix of self.current_token_type
                        acceptable_token_types: list[TokenType] = interactive_copy.accepts()
                        for acceptable_token_type in acceptable_token_types:
                            last_token_regex_pattern: PatternStr = self.parser._terminals_dict[acceptable_token_type].pattern
                            if self._match_regex_prefix(concatenated[len(concatenated)-i:], last_token_regex_pattern.to_regexp()): # try to match the rest of the llm token as a prefix of any string that matches the regex pattern
                                acceptable_llm_tokens.append(id) # add llm token if everything passes
                                added = True
                                break
                    except UnexpectedCharacters:
                        pass
                    except UnexpectedToken:
                        pass
                    except Exception as e:
                        print(e)
                        pass
                    if added:
                        break

        return tensor(acceptable_llm_tokens, dtype=int).view(-1)
    
    def accept_token(self, token_id: int) -> None:
        """Accept a token provided its id."""
        llm_token: str = self.llm_tokenizer.Decode(token_id) # decode the llm token
        accepted: bool = False
        if not self.current_token_types: # not currently accepting any partial tokens
            for i in range(len(llm_token)+1):
                interactive_copy: InteractiveParser = self.interactive.copy()
                try:
                    # try to lex the partial llm token as 0 or more complete grammar tokens
                    for token in self.parser.lex(llm_token[:len(llm_token)-i]): # raises error if lexer fails
                        interactive_copy.feed_token(token) # raises error if parser fails

                    # try to lex the rest of the llm token as a prefix of any next grammar token
                    acceptable_token_types: list[TokenType] = interactive_copy.accepts()
                    for acceptable_token_type in acceptable_token_types:
                        last_token_regex_pattern: PatternStr = self.parser._terminals_dict[acceptable_token_type].pattern
                        if self._match_regex_prefix(llm_token[len(llm_token)-i:], last_token_regex_pattern.to_regexp()): # try to match the rest of the llm token as a prefix of any string that matches the regex pattern
                            if not accepted:
                                self.interactive = interactive_copy
                                self.parsed_str += llm_token
                                accepted = True
                            if i == 0:
                                # reset current token types and value as full llm token was accepted
                                self.current_token_types = set()
                                self.current_token_value = ""
                                self.is_complete: bool = "$END" in self.interactive.accepts()
                                return
                            # add acceptable_token_type to current token types and update current token value
                            self.current_token_types.add(acceptable_token_type)
                            self.current_token_value = llm_token[len(llm_token)-i:]

                    if accepted:
                        self.is_complete: bool = "$END" in self.interactive.accepts()
                        return
                except UnexpectedCharacters:
                    pass
                except UnexpectedToken:
                    pass
                except Exception as e:
                    print(e)
                    pass

        else: # currently accepting a partial token
            concatenated: str = self.current_token_value + llm_token
            for i in range(len(concatenated)+1):
                interactive_copy: InteractiveParser = self.interactive.copy()
                try:
                    # try to lex the partial llm token as 0 or more complete grammar tokens
                    for j, token in enumerate(self.parser.lex(concatenated[:len(concatenated)-i])): # raises error if lexer fails
                        if j == 0:
                            assert token.type in self.current_token_types # assert that the first token was one of the current token types being built
                        interactive_copy.feed_token(token) # raises error if parser fails

                    # try to lex the rest of the llm token as a prefix of any next grammar token
                    acceptable_token_types: list[TokenType] = interactive_copy.accepts()
                    for acceptable_token_type in acceptable_token_types:
                        last_token_regex_pattern: PatternStr = self.parser._terminals_dict[acceptable_token_type].pattern
                        if self._match_regex_prefix(concatenated[len(concatenated)-i:], last_token_regex_pattern.to_regexp()): # try to match the rest of the llm token as a prefix of any string that matches the regex pattern
                            if not accepted:
                                self.interactive = interactive_copy
                                self.current_token_types = set() # reset current token types after first accept
                                self.parsed_str += llm_token
                                accepted = True
                            if i == 0:
                                # reset current token types and value as full llm token was accepted
                                self.current_token_value = ""
                                return
                            # add acceptable_token_type to current token types and update current token value
                            self.current_token_types.add(acceptable_token_type)
                            self.current_token_value = concatenated[len(concatenated)-i:]

                    if accepted:
                        return
                except UnexpectedCharacters:
                    pass
                except UnexpectedToken:
                    pass
                except Exception as e:
                    print(e)
                    pass
