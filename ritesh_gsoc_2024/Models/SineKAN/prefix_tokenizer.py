from collections import Counter, OrderedDict
import re
from tqdm import tqdm
import pandas as pd
from torchtext.vocab import vocab


class PrefixTokenizer:
    def __init__(self, df, special_symbols, UNK_IDX):
        # Initializing amplitude and squared amplitude sequences
        self.amps = df.amp.tolist()
        self.sqamps = df.sqamp.tolist()


        # Regex patterns
        self.pattern_momentum = re.compile(r'\b[ijkl]_\d{1,}\b')
        self.pattern_num_123 = re.compile(r'\b(?![ps]_)\w+_\d{1,}\b')
        self.pattern_special = re.compile(r'\b\w+_+\w+\b\\')
        self.pattern_prop = re.compile(r'Prop')
        self.pattern_int = re.compile(r'int\{')
        self.pattern_plus = re.compile(r'\+')
        self.pattern_minus = re.compile(r'-')
        self.pattern_times = re.compile(r'\*')
        self.pattern_div = re.compile(r'/')
        self.pattern_comma = re.compile(r',')
        self.pattern_exponent = re.compile(r'\^')
        self.pattern_percent = re.compile(r'%')
        self.pattern_right_bracket = re.compile(r'\}')
        self.pattern_mass = re.compile(r'\b\w+_\w\b')
        self.pattern_s = re.compile(r'\b\w+_\d{2,}\b')
        self.pattern_reg_prop = re.compile(r'\b\w+_\d{1}\b')
        self.pattern_left_parentheses = re.compile(r'\(')
        self.pattern_right_parentheses = re.compile(r'\)')
        self.pattern_antipart = re.compile(r'(\w)_\w+_\d+\(X\)\^\(\*\)')
        self.pattern_part = re.compile(r'(\w)_\w+_\d+\(X\)')
        self.pattern_index = re.compile(r'\b\w+_\w+_\d{2,}\b')

        self.special_symbols = special_symbols
        self.UNK_IDX = UNK_IDX

    @staticmethod
    def remove_whitespace(expression):
        """Remove all forms of whitespace from the expression."""
        return re.sub(r'\s+', '', expression)

    @staticmethod
    def split_expression(expression):
        """Split the expression by space delimiter."""
        return re.split(r' ', expression)

    @staticmethod
    def replace_elements(lst, old_element, new_element):
        """Replace elements in a list."""
        return [new_element if x == old_element else x for x in lst]

    def build_tgt_vocab(self):
        """Build vocabulary for target sequences."""
        counter = Counter()
        for eqn in tqdm(self.sqamps, desc='Processing tgt vocab'):
            tokens = self.tgt_tokenize(eqn)
            counter += Counter(tokens)
        ordered_dict = OrderedDict(counter)
        voc = vocab(
            ordered_dict, specials=self.special_symbols[:], special_first=True)
        voc.set_default_index(self.UNK_IDX)
        return voc

    def build_src_vocab(self,_):
        """Build vocabulary for source sequences."""
        counter = Counter()
        for diag in tqdm(self.amps, desc='Processing src vocab'):
            tokens = self.src_tokenize(diag,None)
            counter += Counter(tokens)
        ordered_dict = OrderedDict(counter)
        voc = vocab(
            ordered_dict, specials=self.special_symbols[:], special_first=True)
        voc.set_default_index(self.UNK_IDX)
        return voc

    def pre_src_tokenize(self, ampl):
        """Pre-tokenize source expressions."""
        temp_ampl = ampl
        if temp_ampl[0] == "-":
            temp_ampl = "INT_NEG " + temp_ampl[1:]
        temp_ampl = temp_ampl.replace("+-", "+INT_NEG ")
        temp_ampl = temp_ampl.replace("(-", "(INT_NEG ")

        temp_ampl = temp_ampl.replace("\\\\", "\\")
        temp_ampl = self.pattern_special.sub(lambda match: ' ' + match.group(0) + ' ', temp_ampl)
        temp_ampl = self.pattern_prop.sub(' Prop( ', temp_ampl)
        temp_ampl = self.pattern_int.sub(' int{ ', temp_ampl)
        temp_ampl = re.sub(r"\^\(\*\)", " CONJ ", temp_ampl)
        temp_ampl = temp_ampl.replace("_CONJ", " CONJ ")

        # Replacing operators with space-wrapped versions
        temp_ampl = self.pattern_plus.sub(' + ', temp_ampl)
        temp_ampl = self.pattern_minus.sub(' - ', temp_ampl)
        temp_ampl = self.pattern_times.sub(' * ', temp_ampl)
        temp_ampl = self.pattern_div.sub(' / ', temp_ampl)
        temp_ampl = self.pattern_comma.sub(' , ', temp_ampl)
        temp_ampl = self.pattern_percent.sub('', temp_ampl)
        temp_ampl = self.pattern_exponent.sub(' ^ ', temp_ampl)
        temp_ampl = self.pattern_right_bracket.sub(' } ', temp_ampl)
        temp_ampl = self.pattern_right_parentheses.sub(' ) ', temp_ampl)
        temp_ampl = self.pattern_left_parentheses.sub(' ( ', temp_ampl)
        temp_ampl = temp_ampl.replace("s_"," s_ ")
        temp_ampl = temp_ampl.replace("INDEX"," INDEX")
        temp_ampl = temp_ampl.replace("MOMENTUM"," MOMENTUM")
        temp_ampl = re.sub(r'   ', ' ', temp_ampl)
        temp_ampl = re.sub(r'  ', ' ', temp_ampl)
        temp_ampl = re.sub(r'_\\', '', temp_ampl)
        temp_ampl = re.sub(r'\\', '', temp_ampl)
        temp_ampl = re.sub(r"_\{", "_{ ", temp_ampl)

        # Replace signed tokens
        temp_ampl = re.sub(r"-\s*INDEX", "INT_NEG INDEX", temp_ampl)
        temp_ampl = re.sub(r"\+\s*INDEX", "INT_POS INDEX", temp_ampl)
        
        temp_ampl = re.sub(r"-\s*MOMENTUM", "INT_NEG MOMENTUM", temp_ampl)
        temp_ampl = re.sub(r"\+\s*MOMENTUM", "INT_POS MOMENTUM", temp_ampl)
        
        ampl_tokens = self.split_expression(temp_ampl)
        ampl_tokens = [item for item in ampl_tokens if item != '']

        return ampl_tokens

    def pre_tgt_tokenize(self, sqampl):
        """Pre-tokenize target expressions."""
        sqampl = self.remove_whitespace(sqampl)
        sqampl = sqampl.replace("+-", "+INT_NEG ")
        sqampl = sqampl.replace("(-", "(INT_NEG ")
        temp_sqampl = sqampl
        temp_sqampl = self.pattern_plus.sub(' + ', temp_sqampl)
        temp_sqampl = self.pattern_minus.sub(' - ', temp_sqampl)
        temp_sqampl = self.pattern_times.sub(' * ', temp_sqampl)
        temp_sqampl = self.pattern_div.sub(' / ', temp_sqampl)
        temp_sqampl = self.pattern_exponent.sub(' ^ ', temp_sqampl)
        temp_sqampl = self.pattern_left_parentheses.sub(' ( ', temp_sqampl)
        temp_sqampl = self.pattern_right_parentheses.sub(' ) ', temp_sqampl)
        temp_sqampl = self.pattern_reg_prop.sub(lambda match: ' ' + match.group(0) + ' ', temp_sqampl)
        temp_sqampl = self.pattern_mass.sub(lambda match: ' ' + match.group(0) + ' ', temp_sqampl)
        temp_sqampl = self.pattern_s.sub(lambda match: ' ' + match.group(0) + ' ', temp_sqampl)
        temp_sqampl = temp_sqampl.replace("s_"," s_ ")
        temp_sqampl = temp_sqampl.replace("reg_prop"," reg_prop ")
        temp_sqampl = re.sub(r'   ', ' ', temp_sqampl)
        temp_sqampl = re.sub(r'  ', ' ', temp_sqampl)
        
        sqampl_tokens = self.split_expression(temp_sqampl)
        sqampl_tokens = [item for item in sqampl_tokens if item != '']

        return sqampl_tokens

    def src_tokenize(self, expr, _):
        """Convert source expressions to prefix notation."""
        expr = self.pre_src_tokenize(expr)
        precedence = {
            '^': 4,  # Exponentiation
            '*': 3,  # Multiplication
            '/': 3,  # Division
            '+': 2,  # Addition
            '-': 2,  # Subtraction
            '(': 1,  # Left Parenthesis
            ')': 1,  # Right Parenthesis
        }
        
        stack = []
        prefix = []
        expr = expr[::-1]

        for e in expr:
            if e in precedence.keys():
                if e == ')':
                    stack.append(e)
                elif e == '(':
                    while stack and stack[-1] != ')':
                        prefix.append(stack.pop())
                    stack.pop()  # Remove the ')' from the stack
                else:
                    while stack and precedence[stack[-1]] > precedence[e]:
                        prefix.append(stack.pop())
                    stack.append(e)
            else:
                prefix.append(e)

        while stack:
            prefix.append(stack.pop())

        prefix.reverse()
        prefix = self.replace_elements(prefix, '-', 'sub')
        prefix = self.replace_elements(prefix, '+', 'add')
        prefix = self.replace_elements(prefix, '^', 'pow')
        prefix = self.replace_elements(prefix, '*', 'mul')
        prefix = self.replace_elements(prefix, '/', 'div')
        
        return prefix
    
    def tgt_tokenize(self, expr):
        expr = self.pre_tgt_tokenize(expr)
        precedence = {
            '^': 4,  # Exponentiation
            '*': 3,  # Multiplication
            '/': 3,  # Division
            '+': 2,  # Addition
            '-': 2,  # Subtraction
            '(': 1,  # Left Parenthesis
            ')': 1   # Right Parenthesis
        }

        stack = []
        prefix = []
        expr = expr[::-1]

        for e in expr:
            if e in precedence.keys():
                if e == ')':
                    stack.append(e)
                elif e == '(':
                    while stack and stack[-1] != ')':
                        prefix.append(stack.pop())
                    stack.pop()  # Remove the ')' from the stack
                else:
                    while stack and precedence[stack[-1]] > precedence[e]:
                        prefix.append(stack.pop())
                    stack.append(e)
            else:
                prefix.append(e)

        while stack:
            prefix.append(stack.pop())

        prefix.reverse()
        prefix = self.replace_elements(prefix, '-', 'sub')
        prefix = self.replace_elements(prefix, '+', 'add')
        prefix = self.replace_elements(prefix, '^', 'pow')
        prefix = self.replace_elements(prefix, '*', 'mul')
        prefix = self.replace_elements(prefix, '/', 'div') 

        return prefix