import glob
import os
import re
from typing import Literal

import regex
from sacremoses import MosesPunctNormalizer
import ftfy
from six import text_type
import nltk
from sacremoses.corpus import Perluniprops
from sacremoses.corpus import NonbreakingPrefixes
from sacremoses.util import is_cjk
from sacremoses.indic import VIRAMAS, NUKTAS

import logging

logger = logging.getLogger(__name__)

perluniprops = Perluniprops()
nonbreaking_prefixes = NonbreakingPrefixes()


mpn = MosesPunctNormalizer()


def fix_double_quotes(text):
    n_quotes = text.count('"')
    if n_quotes == 0 or (n_quotes % 2) == 1 or '""' in text or '" "' in text:
        return text

    original_text = text

    i, i_quote, n_changes = 0, 0, 0
    while i < len(text):
        if text[i] != '"':
            i += 1
            continue

        if (i_quote % 2) == 0:
            if i > 0 and text[i - 1] != ' ':
                text = text[:i] + ' ' + text[i:]
                i += 1
                n_changes += 1
            if i + 1 < len(text) and text[i + 1] == ' ':
                text = text[:i + 1] + text[i + 2:]
                n_changes += 1
        else:
            if i > 0 and text[i - 1] == ' ':
                text = text[:i - 1] + text[i:]
                i -= 1
                n_changes += 1
            if i + 1 < len(text) and text[i + 1].isalnum():
                text = text[:i + 1] + ' ' + text[i + 1:]
                n_changes += 1

        i_quote += 1
        i += 1

    # too much changes, let's return the original text to play it safe
    if n_changes > 2 and n_changes > n_quotes * 2 / 3:
        return original_text
    return text


def normalize_abbreviations(text):
    text = text.replace(" n't ", "n't ")
    text = text.replace(" N'T ", "N'T ")
    text = text.replace(" 'll ", "'ll ")
    text = text.replace(" 'LL ", "'LL ")
    text = text.replace(" 're ", "'re ")
    text = text.replace(" 'RE ", "'RE ")
    text = text.replace(" 've ", "'ve ")
    text = text.replace(" 'VE ", "'VE ")
    text = text.replace(" 'm ", "'m ")
    text = text.replace(" 'M ", "'M ")
    text = text.replace(" 's ", "'s ")
    text = text.replace(" 'S ", "'S ")
    text = text.replace(" 'd ", "'d ")
    text = text.replace(" 'D ", "'D ")

    text = text.replace(" n't,", "n't,")
    text = text.replace(" N'T,", "N'T,")
    text = text.replace(" 'll,", "'ll,")
    text = text.replace(" 'LL,", "'LL,")
    text = text.replace(" 're,", "'re,")
    text = text.replace(" 'RE,", "'RE,")
    text = text.replace(" 've,", "'ve,")
    text = text.replace(" 'VE,", "'VE,")
    text = text.replace(" 'm,", "'m,")
    text = text.replace(" 'M,", "'M,")
    text = text.replace(" 's,", "'s,")
    text = text.replace(" 'S,", "'S,")
    text = text.replace(" 'd,", "'d,")
    text = text.replace(" 'D,", "'D,")

    text = text.replace(" n't.", "n't.")
    text = text.replace(" N'T.", "N'T.")
    text = text.replace(" 'll.", "'ll.")
    text = text.replace(" 'LL.", "'LL.")
    text = text.replace(" 're.", "'re.")
    text = text.replace(" 'RE.", "'RE.")
    text = text.replace(" 've.", "'ve.")
    text = text.replace(" 'VE.", "'VE.")
    text = text.replace(" 'm.", "'m.")
    text = text.replace(" 'M.", "'M.")
    text = text.replace(" 's.", "'s.")
    text = text.replace(" 'S.", "'S.")
    text = text.replace(" 'd.", "'d.")
    text = text.replace(" 'D.", "'D.")
    return text


def clean(text, minimal=False):
    if not minimal:
        text = add_whitespace(text)
        text = normalize_abbreviations(text)
        text = fix_double_quotes(text)
        text = mpn.normalize(text)

    text = ftfy.fix_text(text)
    text = text.strip()
    return text


def add_whitespace(text):
    text = ' '.join(text.replace('\n', "<<NEWLINE/>>").split()).replace("<<NEWLINE/>>", '\n')  # remove excess whitespace
    for i in range(len(text)-2, -1, -1):
        if text[i] == '.' and (text[i + 1].isupper() or text[i + 1] in ['‘', '(', '[', '{']):
            text = text[:i+1] + ' ' + text[i+1:]
        elif text[i] in ['?', '!', '…', '’'] and (text[i + 1].isalnum() or text[i + 1] in ['‘', '(', '[', '{']):
            text = text[:i+1] + ' ' + text[i+1:]
        elif i > 2 and text[i] == '.' and text[i - 1] == '.' and text[i - 2] == '.' and text[i + 1] != ' ':
            text = text[:i+1] + ' ' + text[i+1:]
        elif i > 2 and text[i] == '.' and text[i - 1] == '.' and text[i - 2] == '.' and text[i + 1] != ' ':
            text = text[:i+1] + ' ' + text[i+1:]
        elif text[i] == ',' and (text[i + 1].isalpha() or text[i + 1] in ['‘', '(', '[', '{']):
            text = text[:i+1] + ' ' + text[i+1:]
        elif text[i] in [';', ')', ']', '}', '%'] and (text[i + 1].isalnum() or text[i + 1] in ['‘', '(', '[', '{']):
            text = text[:i+1] + ' ' + text[i+1:]
        elif text[i] == ':' and (text[i + 1] in ['‘', '(', '[', '{'] or (text[i + 1].isalnum() and (not text[i + 1].isnumeric() or i - 1 < 0 or not text[i - 1].isnumeric()))):
            text = text[:i+1] + ' ' + text[i+1:]
        elif text[i] in ['(', '[', '{'] and text[i + 1] == ' ':
            text = text[:i+1] + text[i+2:]
        elif text[i] == ' ' and text[i+1] in ['.', ';', ':', '?', '!', '…', ',', '’', ')', ']', '}']:
            text = text[:i] + text[i+1:]
        elif i > 0 and text[i] == ' ' and text[i - 1] in ['$', '£', '€'] and text[i + 1].isnumeric():
            text = text[:i] + text[i+1:]
        elif i > 0 and text[i] == ' ' and text[i - 1].isnumeric() and text[i + 1] == '%':
            text = text[:i] + text[i+1:]

    return text

def _cleanup_qed(text):
    punctuation_ex = re.compile(r'([.!?]\s*)')
    unimportant_chars_ex = re.compile(r'\(.*?\)|[.!?]')
    lines = []
    for line in text.splitlines():
        nchars = len(line)
        if nchars > 0:
            line_body = unimportant_chars_ex.sub('', line)
            f_upper = sum(c.isupper() for c in line_body) / len(line_body)
            if f_upper >= 0.5: # Mostly uppercase characters
                split_on_punctuation = punctuation_ex.split(line.replace('l', 'I'))
                line = ''.join([sentence.capitalize() for sentence in split_on_punctuation])
        lines.append(line.strip())
    return '\n'.join(lines)

def _cleanup_extra_spaces(text):
    multiple_spaces_ex = re.compile(r'[ \t\u00A0]+')
    space_before_punctuation_ex = re.compile(r'[ \t\u00A0]([.,;!?])')
    text = multiple_spaces_ex.sub(' ', text)
    text = space_before_punctuation_ex.sub(r'\1', text)
    return text


class Cleanups:
    def __init__(self, data_dir: str, proc_dir: str, rx_format: Literal['*train', '*test', '*dev']):
        CLEANER = {
            "bnc_spoken": Cleanups.bnc_spoken,
            "childes": Cleanups.childes,
            "gutenberg": Cleanups.gutenberg,
            "open_subtitles": Cleanups.open_subtitles,
            "simple_wiki": Cleanups.simple_wikipedia,
            "switchboard": Cleanups.switchboard,
        }

        if not os.path.exists(data_dir):
            logger.warning(f"Dir not found: {data_dir}. Downliad the data from https://osf.io/ad7qg/")
            raise FileNotFoundError(f"Dir not found: {data_dir}")

        mode = rx_format[1:]
        if not proc_dir.endswith(mode) and not proc_dir.endswith(mode + '/'):
            proc_dir = os.path.join(proc_dir, mode)

        if not os.path.exists(proc_dir):
            os.makedirs(proc_dir)

        files = glob.glob(data_dir + rx_format)

        for file in files:
            cleaner = None
            file_name = os.path.basename(file)
            for key in CLEANER.keys():
                if key in file_name:
                    cleaner = CLEANER[key]
                    break
            if cleaner is None:
                logger.warning(f"File {file_name} not supported")
                continue
            with open(file, 'r') as f:
                text = f.read()
            text = cleaner(text)
            idx = file_name.find('.')
            file_name = file_name[:idx] + '_proc.txt'
            with open(os.path.join(proc_dir, file_name), 'w') as f:
                f.write(text)

    @staticmethod
    def aochildes(text: str) -> str:
        logger.info("Cleaning up AOCHILDES text")
        _len_i = len(text)
        logger.info(f"[AOCHILDES] Original text length: {_len_i}")

        text = _cleanup_extra_spaces(text)
        text = '\n'.join(('"'+clean(line[0].upper() + line[1:])+'"' for line in text.split('\n')))

        _len_f = len(text)
        logger.info(f"[AOCHILDES] Final text length: {_len_f}")
        logger.info(f"[AOCHILDES] Text length reduction: {_len_i - _len_f} or {100* (_len_i - _len_f) / _len_i:.3f}%")
        return text

    @staticmethod
    def childes(text: str) -> str:
        logger.info("Cleaning up CHILDES text")
        _len_i = len(text)
        logger.info(f"[CHILDES] Original text length: {_len_i}")

        text = _cleanup_extra_spaces(text)

        def helper(line:str):
            if line.startswith("["):
                line = line[1:]
            if line.endswith("]"):
                line = line[:-1]
            if line.startswith("*"):
                idx = line.find(":")
                line = line[idx+1:]
            line = line.strip()
            return line

        texts = (helper(line) for line in text.split('\n'))
        texts = (line for line in texts if len(line.split()) > 2)
        text = '\n'.join((line for line in texts if len(line)>0))

        _len_f = len(text)
        logger.info(f"[CHILDES] Final text length: {_len_f}")
        logger.info(f"[CHILDES] Text length reduction: {_len_i - _len_f} or {100* (_len_i - _len_f) / _len_i:.3f}%")
        return text


    @staticmethod
    def bnc_spoken(text: str) -> str:
        logger.info("Cleaning up BNC Spoken text")
        _len_i = len(text)
        logger.info(f"[BNC Spoken] Original text length: {_len_i}")

        text = _cleanup_extra_spaces(text)
        prev_line = None

        def helper(txt):
            global prev_line
            txt = txt.strip()
            if len(txt) == 0:
                prev_line = None
                return ""
            if txt in [".", "!", "?"]:
                return ""
            txt = txt[0].upper() + txt[1:]
            txt = clean(txt)
            txt = f'"{txt}"'
            if prev_line is not None and prev_line == txt:
                return ""
            prev_line = txt
            return txt

        texts = (helper(line) for line in text.split('\n'))
        text = '\n'.join((txt for txt in texts if len(txt)>0))

        _len_f = len(text)
        logger.info(f"[BNC Spoken] Final text length: {_len_f}")
        logger.info(f"[BNC Spoken] Text length reduction: {_len_i - _len_f} or {100* (_len_i - _len_f) / _len_i:.3f}%")
        return text

    @staticmethod
    def cbt(text: str) -> str:
        logger.info("Cleaning up CBT text")
        _len_i = len(text)
        logger.info(f"[CBT] Original text length: {_len_i}")

        text = _cleanup_extra_spaces(text)

        def helper(line):
            line = line.strip()
            line = line.replace("-LRB-", "(")
            line = line.replace("-LCB-", "{")
            line = line.replace("-LSB-", "[")
            line = line.replace("-RRB-", ")")
            line = line.replace("-RCB-", "}")
            line = line.replace("-RSB-", "]")
            line = line.replace("`` ", '"')
            line = line.replace("``", '"')
            line = line.replace(" ''", '"')
            line = line.replace("''", '"')

            if len(line) == 0:
                return ""
            line = clean(line)
            return line

        texts = (helper(line) for line in text.split('\n'))
        text = '\n'.join((txt for txt in texts if len(txt)>0))

        _len_f = len(text)
        logger.info(f"[CBT] Final text length: {_len_f}")
        logger.info(f"[CBT] Text length reduction: {_len_i - _len_f} or {100* (_len_i - _len_f) / _len_i:.3f}%")
        return text

    @staticmethod
    def children_stories(text: str) -> str:
        logger.info("Cleaning up Children Stories text")
        _len_i = len(text)
        logger.info(f"[Children Stories] Original text length: {_len_i}")

        lines = text.split('\n')
        num_non_blank_lines = 0

        def helper(line):
            global num_non_blank_lines
            if len(line.strip()) == 0:
                if num_non_blank_lines > 1:
                    num_non_blank_lines = 0
                    return ""
            if line.startswith(" "*4):
                line = f"[TAB] {' '.join(line.strip().split())}"
            else:
                line = ' '.join(line.strip().split())

            num_non_blank_lines += 1
            line = clean(line, minimal=True)

            return line

        texts = (helper(line) for line in lines)
        text = '\n'.join((txt for txt in texts if len(txt)>0))

        _len_f = len(text)
        logger.info(f"[Children Stories] Final text length: {_len_f}")
        logger.info(f"[Children Stories] Text length reduction: {_len_i - _len_f} or {100* (_len_i - _len_f) / _len_i:.3f}%")
        return text

    @staticmethod
    def gutenberg(text: str) -> str:
        logger.info("Cleaning up Gutenberg text")
        _len_i = len(text)
        logger.info(f"[Gutenberg] Original text length: {_len_i}")

        texts = []

        num_blank_lines = 0
        accumulated_line = []
        for line in text.split('\n'):
            line = ' '.join(line.strip().split())
            line = clean(line, minimal=True)
            if len(line) == 0:
                if len(accumulated_line) > 0:
                    texts.append(' '.join(accumulated_line))
                    last_num_non_blank_lines = len(accumulated_line)
                accumulated_line = []
                num_blank_lines += 1
                continue
            num_blank_lines = 0
            accumulated_line.append(line)

        text = '\n'.join(texts)

        _len_f = len(text)
        logger.info(f"[Gutenberg] Final text length: {_len_f}")
        logger.info(f"[Gutenberg] Text length reduction: {_len_i - _len_f} or {100* (_len_i - _len_f) / _len_i:.3f}%")
        return text

    @staticmethod
    def open_subtitles(text: str) -> str:
        logger.info("Cleaning up Open Subtitles text")
        _len_i = len(text)
        logger.info(f"[Open Subtitles] Original text length: {_len_i}")

        prev_line = None
        texts = []
        for line in text.split('\n'):
            line = ' '.join(line.strip().split())
            line = clean(line, minimal=True)

            if line.startswith("- "):
                line = line[2:]
            elif line.startswith("-"):
                line = line[1:]

            if len(line) == 0:
                prev_line = line
                continue
            if not line.endswith(":") and not line.startswith('"') and not line.endswith('"') and not (
                    line.startswith("(") and line.endswith(")")) and not (
                    line.startswith("[") and line.endswith("]")) and not (line.startswith("{") and line.endswith("}")):
                line = f'"{line}"'
            if prev_line is not None and prev_line == line:
                continue
            texts.append(line)
            prev_line = line

        text = '\n'.join(texts)
        _len_f = len(text)
        logger.info(f"[Open Subtitles] Final text length: {_len_f}")
        logger.info(f"[Open Subtitles] Text length reduction: {_len_i - _len_f} or {100* (_len_i - _len_f) / _len_i:.3f}%")
        return text

    @staticmethod
    def qed(text: str) -> str:
        logger.info("Cleaning up Switchboard text")
        _len_i = len(text)
        logger.info(f"[Switchboard] Original text length: {_len_i}")

        def helper(line):
            line = ' '.join(line.strip().split())
            if line.startswith("- "):
                line = line[2:]
            elif line.startswith("-"):
                line = line[1:]
            line = clean(line, minimal=True)

            if len(line) == 0:
                return ""
            if "&lt" in line and "&gt" in line:
                return ""
            if "->" in line:
                return ""

            line = line.replace("&gt;", ">")
            line = line.replace("&lt;", "<")
            line = line.replace("&amp;", "&")

            if line.endswith(":") and not any(c.isalpha() for c in line):
                return ""
            if not line.endswith(":") and not line.startswith('"') and not line.endswith('"') and not (
                    line.startswith("(") and line.endswith(")")) and not (
                    line.startswith("[") and line.endswith("]")) and not (line.startswith("{") and line.endswith("}")):
                line = f'"{line}"'
            return line

        texts = (helper(line) for line in text.split('\n'))
        text = '\n'.join((txt for txt in texts if len(txt)>0))
        text = _cleanup_qed(text)
        _len_f = len(text)
        logger.info(f"[Switchboard] Final text length: {_len_f}")
        logger.info(f"[Switchboard] Text length reduction: {_len_i - _len_f} or {100* (_len_i - _len_f) / _len_i:.3f}%")
        return text

    @staticmethod
    def segment(text: str) -> str:
        logger.info("Cleaning up Segment text")
        _len_i = len(text)
        logger.info(f"[Segment] Original text length: {_len_i}")

        lines = (nltk.sent_tokenize(line.strip()) for line in text.split('\n') if len(line.strip()) > 0)
        text = "\n".join('\n'.join(line) for line in lines)

        _len_f = len(text)
        logger.info(f"[Segment] Final text length: {_len_f}")
        logger.info(f"[Segment] Text length reduction: {_len_i - _len_f} or {100* (_len_i - _len_f) / _len_i:.3f}%")
        return text

    @staticmethod
    def simple_wikipedia(text: str) -> str:
        logger.info("Cleaning up Wikipedia text")
        _len_i = len(text)
        logger.info(f"[Wikipedia] Original text length: {_len_i}")

        texts = []
        prev_line = None
        for line in text.split('\n'):
            line = ' '.join(line.strip().split())
            line = regex.sub("", line)
            line = clean(line, minimal=True)

            if len(line) == 0:
                if prev_line is not None and prev_line != "":
                    texts.append(prev_line)
                prev_line = None
                continue
            if "is a commune. It is" in line and len(line) < 128:
                prev_line = None
                continue
            if "is a commune found" in line and len(line) < 128:
                prev_line = None
                continue
            if "is a city in" in line and len(line) < 128:
                prev_line = None
                continue
            if "is a village in" in line and len(line) < 128:
                prev_line = None
                continue
            if "is a municipality in" in line and len(line) < 128:
                prev_line = None
                continue
            if "is a town in" in line and len(line) < 128:
                prev_line = None
                continue
            line = line.replace("&gt;", ">")
            line = line.replace("&lt;", "<")
            line = line.replace("&amp;", "&")

            if prev_line is not None:
                texts.append(prev_line)
            prev_line = line

        text = '\n'.join(texts)

        _len_f = len(text)
        logger.info(f"[Wikipedia] Final text length: {_len_f}")
        logger.info(f"[Wikipedia] Text length reduction: {_len_i - _len_f} or {100* (_len_i - _len_f) / _len_i:.3f}%")
        return text

    @staticmethod
    def switchboard(text: str) -> str:
        logger.info("Cleaning up Switchboard text")
        _len_i = len(text)
        logger.info(f"[Switchboard] Original text length: {_len_i}")

        def helper(line):
            line = ' '.join(line.strip().split())
            if line.startswith("- "):
                line = line[2:]
            elif line.startswith("-"):
                line = line[1:]
            line = clean(line, minimal=True)
            return line

        texts = (helper(line) for line in text.split('\n'))
        text = '\n'.join((txt for txt in texts if len(txt)>0))

        _len_f = len(text)
        logger.info(f"[Switchboard] Final text length: {_len_f}")
        logger.info(f"[Switchboard] Text length reduction: {_len_i - _len_f} or {100* (_len_i - _len_f) / _len_i:.3f}%")
        return text

    @staticmethod
    def wikipedia(text: str) -> str:
        logger.info("Cleaning up Wikipedia text")
        _len_i = len(text)
        logger.info(f"[Wikipedia] Original text length: {_len_i}")

        regex_1 = re.compile(r"\[\d+\]")
        regex_2 = re.compile(r"\[\[([^\|\]]*)\|*[^\]]*\]\]")
        regex_3 = re.compile(r"= = = ([^\=]*) = = =")

        texts = []
        for line in text.split('\n'):
            line = ' '.join(line.strip().split())
            line = clean(line, minimal=True)
            line = regex_1.sub("", line)
            line = regex_2.sub(r"\1", line)
            line = regex_3.sub(r"\1", line)

            if line.startswith("[[Category:") or line.startswith("[[File:") or len(line) == 0:
                continue
            texts.append(line)

        text = '\n'.join(texts)

        _len_f = len(text)
        logger.info(f"[Wikipedia] Final text length: {_len_f}")
        logger.info(f"[Wikipedia] Text length reduction: {_len_i - _len_f} or {100* (_len_i - _len_f) / _len_i:.3f}%")
        return text



