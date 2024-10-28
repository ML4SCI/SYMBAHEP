from collections import Counter, OrderedDict
from itertools import cycle
import re
import random
from torchtext.vocab import vocab
from tqdm import tqdm
import warnings

class Tokenizer:
    def __init__(self, df, index_token_pool_size, momentum_token_pool_size, special_symbols, UNK_IDX, to_replace, is_normal=False, is_old=False):
        self.amps = df.amp.tolist()
        self.sqamps = df.sqamp.tolist()
        self.is_old = is_old

        if index_token_pool_size < 100:
            warnings.warn(f"Index token pool size ({index_token_pool_size}) is less. Considering increasing it.", UserWarning)
        if momentum_token_pool_size < 100:
            warnings.warn(f"Momentum token pool size ({momentum_token_pool_size}) is less. Considering increasing it", UserWarning)
            
        self.tokens_pool = [f"INDEX_{i}" for i in range(index_token_pool_size)]
        self.momentum_pool = [f"MOMENTUM_{i}" for i in range(momentum_token_pool_size)]
        
        # Updated regex patterns
        self.pattern_momentum = re.compile(r'\b[ijkl]_\d{1,}\b')
        self.pattern_num_123 = re.compile(r'\b(?![ps]_)\w+_\d{1,}\b')
        self.pattern_special = re.compile(r'\b\w+_+\w+\b\\')
        self.pattern_prop = re.compile(r'Prop')
        self.pattern_int = re.compile(r'int\{')
        self.pattern_plus = re.compile(r'\+')
        self.pattern_minus = re.compile(r'-')
        self.pattern_times = re.compile(r'\*')
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
        
        self.to_replace = to_replace
        self.is_normal = is_normal

    @staticmethod
    def remove_whitespace(expression):
        """Remove all forms of whitespace from the expression."""
        cleaned_expression = re.sub(r'\s+', '', expression)
        return cleaned_expression

    @staticmethod
    def split_expression(expression):
        """Split the expression by space delimiter."""
        return re.split(r' ', expression)

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

    def build_src_vocab(self,seed):
        """Build vocabulary for source sequences."""
        counter = Counter()
        for diag in tqdm(self.amps, desc='Processing src vocab'):
            tokens = self.src_tokenize(diag,seed)
            counter += Counter(tokens)
        if self.is_old:
            counter+=Counter(self.tokens_pool + self.momentum_pool)
        ordered_dict = OrderedDict(counter)
        voc = vocab(
            ordered_dict, specials=self.special_symbols[:], special_first=True)
        voc.set_default_index(self.UNK_IDX)
        return voc
    
    def src_replace(self,ampl,seed):
        
        ampl = self.remove_whitespace(ampl)
        random.seed(seed)
        if self.is_normal:
            random_tokens = self.tokens_pool
        else:
            random_tokens = random.sample(self.tokens_pool, len(self.tokens_pool))    
        token_cycle = cycle(random_tokens)
        
        random.seed(seed)
        if self.is_normal:
            random_momentum = self.momentum_pool
        else:
            random_momentum = random.sample(
            self.momentum_pool, len(self.momentum_pool))
        momentum_cycle = cycle(random_momentum)

        momentum_mapping = {}

        temp_ampl = ampl
        for match in set(self.pattern_momentum.findall(ampl)):
            if match not in momentum_mapping:
                momentum_mapping[match] = next(momentum_cycle)
            temp_ampl = temp_ampl.replace(match, momentum_mapping[match])

        num_123_mapping = {}
        for match in set(self.pattern_num_123.findall(ampl)):
            if match not in num_123_mapping:
                num_123_mapping[match] = next(token_cycle)
            temp_ampl = temp_ampl.replace(match, num_123_mapping[match])
        
        return temp_ampl
    
    def src_tokenize(self, ampl, seed):
        temp_ampl = ampl
        if self.to_replace:
            temp_ampl = self.src_replace(ampl,seed)
            
        temp_ampl = temp_ampl.replace("\\\\","\\")
        temp_ampl = self.pattern_special.sub(
            lambda match: ' ' + match.group(0) + ' ', temp_ampl)
        temp_ampl = self.pattern_prop.sub(' Prop( ', temp_ampl)
        temp_ampl = self.pattern_int.sub(' int{ ', temp_ampl)
        temp_ampl = re.sub(r"\^\(\*\)", "_CONJ ", temp_ampl)

        temp_ampl = self.pattern_plus.sub(' + ', temp_ampl)
        temp_ampl = self.pattern_minus.sub(' - ', temp_ampl)
        temp_ampl = self.pattern_times.sub(' * ', temp_ampl)
        temp_ampl = self.pattern_comma.sub(' , ', temp_ampl)
        temp_ampl = self.pattern_percent.sub(' % ', temp_ampl)
        temp_ampl = self.pattern_exponent.sub(' ^ ', temp_ampl)
        temp_ampl = self.pattern_right_bracket.sub(' } ', temp_ampl)
        temp_ampl = self.pattern_right_parentheses.sub(' ) ', temp_ampl)
        temp_ampl = self.pattern_left_parentheses.sub(' ( ', temp_ampl)
        temp_ampl = re.sub(r'   ', ' ', temp_ampl)
        temp_ampl = re.sub(r'  ', ' ', temp_ampl)
        temp_ampl = re.sub(r'_\\', ' _\\ ', temp_ampl)
        temp_ampl = re.sub(r'\\', ' \\ ', temp_ampl)
        temp_ampl = re.sub(r"_\{","_{ ",temp_ampl)

        # ADDED NEW
        temp_ampl = temp_ampl.replace("\\","")
        temp_ampl = temp_ampl.replace("%","")

        

        ampl_tokens = self.split_expression(temp_ampl)
        ampl_tokens = [item for item in ampl_tokens if item != '']

        return ampl_tokens

    def tgt_tokenize(self, sqampl):
        sqampl = self.remove_whitespace(sqampl)

        temp_sqampl = sqampl
        temp_sqampl = self.pattern_plus.sub(' + ', temp_sqampl)
        temp_sqampl = self.pattern_minus.sub(' - ', temp_sqampl)
        temp_sqampl = self.pattern_times.sub(' * ', temp_sqampl)
        temp_sqampl = self.pattern_exponent.sub(' ^ ', temp_sqampl)
        temp_sqampl = self.pattern_left_parentheses.sub(' ( ', temp_sqampl)
        temp_sqampl = self.pattern_right_parentheses.sub(' ) ', temp_sqampl)
        temp_sqampl = self.pattern_reg_prop.sub(
            lambda match: ' ' + match.group(0) + ' ', temp_sqampl)
        temp_sqampl = self.pattern_mass.sub(
            lambda match: ' ' + match.group(0) + ' ', temp_sqampl)
        temp_sqampl = self.pattern_s.sub(
            lambda match: ' ' + match.group(0) + ' ', temp_sqampl)
        
        temp_sqampl = temp_sqampl.replace("reg_prop"," reg_prop ")
        temp_sqampl = re.sub(r'   ', ' ', temp_sqampl)
        temp_sqampl = re.sub(r'  ', ' ', temp_sqampl)

        sqampl_tokens = self.split_expression(temp_sqampl)
        sqampl_tokens = [item for item in sqampl_tokens if item != '']

        return sqampl_tokens