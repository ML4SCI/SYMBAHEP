from collections import Counter, OrderedDict
from itertools import cycle
import re
import random
from torchtext.vocab import vocab
from tqdm import tqdm
import warnings

class Tokenizer:
    """
    Tokenizer for processing symbolic mathematical expressions.
    """
    def __init__(self, df, index_token_pool_size, momentum_token_pool_size, special_symbols, UNK_IDX, to_replace):
        self.amps = df.amp.tolist()
        self.sqamps = df.sqamp.tolist()

        # Issue warnings if token pool sizes are too small
        if index_token_pool_size < 100:
            warnings.warn(f"Index token pool size ({index_token_pool_size}) is small. Consider increasing it.", UserWarning)
        if momentum_token_pool_size < 100:
            warnings.warn(f"Momentum token pool size ({momentum_token_pool_size}) is small. Consider increasing it.", UserWarning)
        
        # Generate token pools
        self.tokens_pool = [f"INDEX_{i}" for i in range(index_token_pool_size)]
        self.momentum_pool = [f"MOMENTUM_{i}" for i in range(momentum_token_pool_size)]
        
        # Regular expression patterns for token replacement
        self.pattern_momentum = re.compile(r'\b[ijkl]_\d{1,}\b')
        self.pattern_num_123 = re.compile(r'\b(?![ps]_)\w+_\d{1,}\b')
        self.pattern_special = re.compile(r'\b\w+_+\w+\b\\')
        self.pattern_prop = re.compile(r'Prop')
        self.pattern_int = re.compile(r'int\{')
        self.pattern_operators = {
            '+': re.compile(r'\+'), '-': re.compile(r'-'), '*': re.compile(r'\*'),
            ',': re.compile(r','), '^': re.compile(r'\^'), '%': re.compile(r'%'),
            '}': re.compile(r'\}'), '(': re.compile(r'\('), ')': re.compile(r'\)')
        }
        self.pattern_mass = re.compile(r'\b\w+_\w\b')
        self.pattern_s = re.compile(r'\b\w+_\d{2,}\b')
        self.pattern_reg_prop = re.compile(r'\b\w+_\d{1}\b')
        self.pattern_antipart = re.compile(r'(\w)_\w+_\d+\(X\)\^\(\*\)')
        self.pattern_part = re.compile(r'(\w)_\w+_\d+\(X\)')
        self.pattern_index = re.compile(r'\b\w+_\w+_\d{2,}\b')
        
        self.special_symbols = special_symbols
        self.UNK_IDX = UNK_IDX
        self.to_replace = to_replace

    @staticmethod
    def remove_whitespace(expression):
        """Remove all forms of whitespace from the expression."""
        return re.sub(r'\s+', '', expression)

    @staticmethod
    def split_expression(expression):
        """Split the expression by space delimiter."""
        return re.split(r' ', expression)

    def build_tgt_vocab(self):
        """Build vocabulary for target sequences."""
        counter = Counter()
        for eqn in tqdm(self.sqamps, desc='Processing target vocab'):
            counter.update(self.tgt_tokenize(eqn))
        voc = vocab(OrderedDict(counter), specials=self.special_symbols[:], special_first=True)
        voc.set_default_index(self.UNK_IDX)
        return voc

    def build_src_vocab(self, seed):
        """Build vocabulary for source sequences."""
        counter = Counter()
        for diag in tqdm(self.amps, desc='Processing source vocab'):
            counter.update(self.src_tokenize(diag, seed))
        voc = vocab(OrderedDict(counter), specials=self.special_symbols[:], special_first=True)
        voc.set_default_index(self.UNK_IDX)
        return voc
    
    def src_replace(self, ampl, seed):
        """Replace indexed and momentum variables with tokenized equivalents."""
        ampl = self.remove_whitespace(ampl)
        
        random.seed(seed)
        token_cycle = cycle(random.sample(self.tokens_pool, len(self.tokens_pool)))
        momentum_cycle = cycle(random.sample(self.momentum_pool, len(self.momentum_pool)))
        
        # Replace momentum tokens
        temp_ampl = ampl
        momentum_mapping = {match: next(momentum_cycle) for match in set(self.pattern_momentum.findall(ampl))}
        for key, value in momentum_mapping.items():
            temp_ampl = temp_ampl.replace(key, value)
        
        # Replace index tokens
        num_123_mapping = {match: next(token_cycle) for match in set(self.pattern_num_123.findall(ampl))}
        for key, value in num_123_mapping.items():
            temp_ampl = temp_ampl.replace(key, value)
        
        return temp_ampl
    
    def src_tokenize(self, ampl, seed):
        """Tokenize source expression, optionally applying replacements."""
        temp_ampl = self.src_replace(ampl, seed) if self.to_replace else ampl
        temp_ampl = temp_ampl.replace('\\\\', '\\').replace('\\', ' \\ ').replace('%', '')
        
        for symbol, pattern in self.pattern_operators.items():
            temp_ampl = pattern.sub(f' {symbol} ', temp_ampl)
        
        temp_ampl = re.sub(r' {2,}', ' ', temp_ampl)
        return [token for token in self.split_expression(temp_ampl) if token]

    def tgt_tokenize(self, sqampl):
        """Tokenize target expression."""
        sqampl = self.remove_whitespace(sqampl)
        temp_sqampl = sqampl
        
        for symbol, pattern in self.pattern_operators.items():
            temp_sqampl = pattern.sub(f' {symbol} ', temp_sqampl)
        
        for pattern in [self.pattern_reg_prop, self.pattern_mass, self.pattern_s]:
            temp_sqampl = pattern.sub(lambda match: f' {match.group(0)} ', temp_sqampl)
        
        temp_sqampl = re.sub(r' {2,}', ' ', temp_sqampl)
        return [token for token in self.split_expression(temp_sqampl) if token]
