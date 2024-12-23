"""Sanskrit Text Utility"""

###############################################################################

__author__ = """Hrishikesh Terdalkar"""
__email__ = "hrishikeshrt@linuxmail.org"
__version__ = "0.2.3"
__created_at__ = "Tue Apr 17 22:20:39 2018"

###############################################################################

import re
import logging

from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple

###############################################################################

LOGGER = logging.getLogger(__name__)

###############################################################################


def ord_unicode(ch: str) -> str:
    """Get Unicode 4-character-identifier corresponding to a character

    Parameters
    ----------
    ch : str
        Single character

    Returns
    -------
    str
        4-character unicode identifier
    """
    return hex(ord(ch)).split("x")[1].zfill(4)


def chr_unicode(u: str) -> str:
    """Get a Unicode character corresponding to 4-chracater identifier

    Parameters
    ----------
    u : str
        4-character unicode identifier

    Returns
    -------
    str
        Single character
    """

    return chr(int(u, 16))


###############################################################################

DESIGN = """
* len(SWARA) == len(MATRA) + 1  # 'अ' is extra at the beginning
* len(EXTENDED_SWARA) == len(EXTENDED_MATRA) + 1 # 'ऍ' is extra at the end
* It is unclear which of 'ॲ' or 'ऍ' should correspond to 'ॅ', current choice is
  the former. If that changes, the order in EXTENDED_MATRA would need to change
* ARTIFICIAL_MATRA contains absent vowel signs. Currently this is just for 'अ'.
  Any new sign should follow the pattern as hyphen ('-') followed by the vowel
  letter (e.g. '-अ').
* DIGITS, COMBINING_DIGIT_MARKS, PUNCTUATION and GENERAL_PUNCTUATION aren't
  part of the ALPHABET. Their inclusion needs more deliberation.
* VEDIC_MARKS are not used in syllabification functions currently.
"""

###############################################################################
# Alphabet of Sanskrit

SWARA = ["अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ॠ", "ऌ", "ॡ", "ए", "ऐ", "ओ", "औ"]
EXTENDED_SWARA = ["ऎ", "ऒ", "ॲ", "ऑ", "ऍ"]

MATRA = ["ा", "ि", "ी", "ु", "ू", "ृ", "ॄ", "ॢ", "ॣ", "े", "ै", "ो", "ौ"]
EXTENDED_MATRA = ["ॆ", "ॊ", "ॅ", "ॉ"]

HRASVA = ["अ", "इ", "उ", "ऋ", "ऌ"] + [MATRA[i] for i in [1, 3, 5, 7]]
DIRGHA = ["आ", "ई", "ऊ", "ॠ", "ॡ", "ए", "ऐ", "ओ", "औ"] + [MATRA[i] for i in [0, 2, 4, 6, 8, 9, 10, 11, 12]] 

KANTHYA = ["क", "ख", "ग", "घ", "ङ"]
TALAVYA = ["च", "छ", "ज", "झ", "ञ"]
MURDHANYA = ["ट", "ठ", "ड", "ढ", "ण"]
DANTYA = ["त", "थ", "द", "ध", "न"]
AUSHTHYA = ["प", "फ", "ब", "भ", "म"]
ANTAHSTHA = ["य", "र", "ल", "व"]
USHMA = ["श", "ष", "स", "ह"]
VISHISHTA = ["ळ"]
EXTENDED_VYANJANA = ["ऩ", "ऱ", "ऴ", "क़", "ख़", "ग़", "ज़", "ड़", "ढ़", "फ़", "य़"]

# --------------------------------------------------------------------------- #

ARTIFICIAL_MATRA_A = f"-{SWARA[0]}"

OM = "ॐ"
AVAGRAHA = "ऽ"

SWARITA = "॑"
DOUBLE_SWARITA = "᳚"
TRIPLE_SWARITA = "᳛"
ANUDATTA = "॒"
CHANDRABINDU = "ँ"
CHANDRABINDU_VIRAMA = "ꣳ"
CHANDRABINDU_SPACING = "ꣲ"
CHANDABINDU_TWO = "ꣵ"
CHANDRABINDU_THREE = "ꣶ"

ANUSWARA = "ं"
VISARGA = "ः"
ARDHAVISARGA = "ᳲ"
JIHVAAMULIYA = "ᳵ"
UPADHMANIYA = "ᳶ"

HALANTA = "्"
NUKTA = "़"  # unused
ABBREV = "॰"
DANDA = "।"
DOUBLE_DANDA = "॥"

# --------------------------------------------------------------------------- #
# Groups

VARGIYA = KANTHYA + TALAVYA + MURDHANYA + DANTYA + AUSHTHYA
VYANJANA = VARGIYA + ANTAHSTHA + USHMA + VISHISHTA

VARGA_PRATHAMA = [VARGIYA[i * 5] for i in range(5)]
VARGA_DWITIYA = [VARGIYA[i * 5 + 1] for i in range(5)]
VARGA_TRITIYA = [VARGIYA[i * 5 + 2] for i in range(5)]
VARGA_CHATURTHA = [VARGIYA[i * 5 + 3] for i in range(5)]
VARGA_PANCHAMA = [VARGIYA[i * 5 + 4] for i in range(5)]

LAGHU_SWARA = [SWARA[i] for i in [0, 2, 4, 6, 8]] + EXTENDED_SWARA[:2]
LAGHU_MATRA = [MATRA[i] for i in [1, 3, 5, 7]] + EXTENDED_MATRA[:2]

# --------------------------------------------------------------------------- #

AYOGAVAAHA_COMMON = [CHANDRABINDU, ANUSWARA, VISARGA]
AYOGAVAAHA = AYOGAVAAHA_COMMON + [JIHVAAMULIYA, UPADHMANIYA]

VEDIC_MARKS = [SWARITA, ANUDATTA, DOUBLE_SWARITA, TRIPLE_SWARITA]
SPECIAL = [
    AVAGRAHA,
    OM,
    CHANDRABINDU_VIRAMA,
    CHANDRABINDU_SPACING,
    CHANDABINDU_TWO,
    CHANDRABINDU_THREE,
]
OTHER = [HALANTA]

# --------------------------------------------------------------------------- #

ARTIFICIAL_MATRA = [ARTIFICIAL_MATRA_A]

ALL_SWARA = SWARA + EXTENDED_SWARA
ALL_VYANJANA = VYANJANA + EXTENDED_VYANJANA
ALL_MATRA = MATRA + EXTENDED_MATRA  # does NOT contain ARTIFICIAL_MATRA

VARNA = ALL_SWARA + ALL_VYANJANA
ALPHABET = VARNA + ALL_MATRA + AYOGAVAAHA + SPECIAL + OTHER + VEDIC_MARKS

# --------------------------------------------------------------------------- #

SPACES = [" ", "\t", "\n", "\r"]
PUNCTUATION = [DANDA, DOUBLE_DANDA, ABBREV]
GENERAL_PUNCTUATION = [".", ",", ";", "", '"', "'", "`"]

DIGITS = ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९"]
COMBINING_DIGIT_MARKS = ["꣠", "꣡", "꣢", "꣣", "꣤", "꣥", "꣦", "꣧", "꣨", "꣩"]

# --------------------------------------------------------------------------- #
# Special Sequences

KSHA = "क्ष"
JNA = "ज्ञ"

###############################################################################


HOW_TO_WRITE = """
Unicode characters chan be typed directly from the keyboard as follows,
[Ctrl+Shift+u] [4-digit-unicode-identifier] [space]

Some of the characters can also be typed using m17n-sanskrit-itrans keyboard
(Package: https://launchpad.net/ubuntu/+source/ibus-m17n)
(File: /usr/share/m17n/sa-itrans.mim)


Notable Unicodes and Shortcuts
---
1cf2 for Ardhavisarga
1cf5 for Jihvamuliya -- kH
1cf6 for Upadhmaniya -- pH
0951 for Swarita -- ''
0952 for Anudatta -- _
0901 for Chandrabindu -- .N
a8f2 for (stand-alone) Chandrabindu Spacing
093d for Avagraha -- .a
094d for Halanta -- .h

0950 for Om -- OM
a8e0 to a8e9 for Combining Devanagari Digits 0-9 (Swara Marks for Samaveda)
"""

###############################################################################

MAAHESHWARA_SUTRA = [
    ["अ", "इ", "उ", "ण्"],
    ["ऋ", "ऌ", "क्"],
    ["ए", "ओ", "ङ्"],
    ["ऐ", "औ", "च्"],
    ["ह", "य", "व", "र", "ट्"],
    ["ल", "ण्"],
    ["ञ", "म", "ङ", "ण", "न", "म्"],
    ["झ", "भ", "ञ्"],
    ["घ", "ढ", "ध", "ष्"],
    ["ज", "ब", "ग", "ड", "द", "श्"],
    ["ख", "फ", "छ", "ठ", "थ", "च", "ट", "त", "व्"],
    ["क", "प", "य्"],
    ["श", "ष", "स", "र्"],
    ["ह", "ल्"],
]

# --------------------------------------------------------------------------- #

MAAHESHWARA_KRAMA = [varna for sutra in MAAHESHWARA_SUTRA for varna in sutra]

# --------------------------------------------------------------------------- #

MAAHESHWARA_IDX = defaultdict(list)

idx = 0
for _sutra_idx, sutra in enumerate(MAAHESHWARA_SUTRA):
    for _internal_idx, varna in enumerate(sutra):
        if HALANTA in varna:
            _idx = -1
        else:
            _idx = idx
            idx += 1
        MAAHESHWARA_IDX[varna].append((_sutra_idx, _internal_idx, _idx))

###############################################################################


def form_pratyaahaara(letters: List[str]) -> str:
    """Form a pratyaahaara from a list of letters"""
    varna_idx = []
    ignored = []

    for varna in letters:
        if varna in MAAHESHWARA_IDX and HALANTA not in varna:
            varna_idx.append(MAAHESHWARA_IDX[varna])
        else:
            ignored.append(varna)

    if ignored:
        LOGGER.info(f"Ignored letters: {ignored}")

    varna_idxs = product(*varna_idx)
    for v_idx in varna_idxs:
        v_idx = sorted(v_idx, key=lambda x: x[2])
        _v_idx = [w[2] for w in v_idx]
        if _v_idx != list(range(_v_idx[0], _v_idx[-1] + 1)):
            continue
        else:
            break
    else:
        LOGGER.warning("Cannot form a pratyaahara due to discontinuity.")
        return None

    _aadi_idx = v_idx[0]
    _pre_antya_idx = v_idx[-1]

    if _pre_antya_idx[1] != len(MAAHESHWARA_SUTRA[_pre_antya_idx[0]]) - 2:
        LOGGER.warning("Cannot form a pratyaahara due to end position.")
        return None

    aadi = MAAHESHWARA_SUTRA[_aadi_idx[0]][_aadi_idx[1]]
    antya = MAAHESHWARA_SUTRA[_pre_antya_idx[0]][-1]
    return f"{aadi}{antya}"


def resolve_pratyaahaara(pratyaahaara: str) -> List[List[str]]:
    """Resolve pratyaahaara into all possible lists of characters"""
    aadi = pratyaahaara[0]
    antya = pratyaahaara[1:]

    possible_starts = []
    possible_ends = []

    for idx, varna in enumerate(MAAHESHWARA_KRAMA):
        if varna == aadi:
            possible_starts.append(idx)
        if varna == antya:
            possible_ends.append(idx)

    resolutions = [
        [
            MAAHESHWARA_KRAMA[idx]
            for idx in range(start, end)
            if HALANTA not in MAAHESHWARA_KRAMA[idx]
        ]
        for start in possible_starts
        for end in possible_ends
        if start < end
    ]
    return resolutions

###############################################################################


def clean(
    text: str,
    punct: bool = False,
    digits: bool = False,
    spaces: bool = True,
    allow: list = None,
) -> str:
    """Clean a line of Sanskrit (Devanagari) text

    Parameters
    ----------
    text : str
        Input string
    punct : bool, optional
        If True, the punctuations are kept.
        The default is False.
    digits : bool, optional
        If True, digits are kept.
        The default is False.
    spaces : bool, optional
        If False, spaces are removed.
        It is recommended to not change the default value
        unless it is specifically relevant to a use-case.
        The default is True.
    allow : list, optional
        List of characters to allow.
        The default is None.

    Returns
    -------
    str
        Clean version of the string
    """
    allow = allow or []
    alphabet = ALPHABET + allow
    if spaces:
        alphabet += SPACES
    if punct:
        alphabet += PUNCTUATION + GENERAL_PUNCTUATION
    if digits:
        alphabet += DIGITS
    answer = "".join(["" if c not in alphabet else c for c in text])
    answer = "\n".join(
        [" ".join(line.split()) for line in answer.split("\n") if line.strip()]
    )
    return answer


def split_lines(text: str, pattern=r"[।॥\r\n]+") -> List[str]:
    """Split a string into a list of strings using regular expression

    Parameters
    ----------
    text : str
        Input string
    pattern : regexp, optional
        Regular expression corresponding to the split points.
        The default is r'[।॥\\r\\n]+'.

    Returns
    -------
    List[str]
        List of strings
    """
    return list(filter(None, re.split(pattern, text)))

def split_into_padas(text: str, syllables_per_pada: int = 8) -> List[str]:
    """
    Split a Sanskrit text into padas based on syllable count.

    Parameters:
    ----------
    text : str
        Input Sanskrit text
    syllables_per_pada : int, optional
        Number of syllables per pada (default is 8 for Anushtup meter)

    Returns:
    -------
    List[str]
        List of padas, each containing the specified number of syllables
    """
    clean_text = clean(text, spaces=True)
    syllables = get_syllables_flat_improved(clean_text)
    
    padas = []
    current_pada = []
    syllable_count = 0

    for syllable in syllables:
        current_pada.append(syllable)
        syllable_count += 1

        if syllable_count == syllables_per_pada:
            padas.append(''.join(current_pada))
            current_pada = []
            syllable_count = 0

    # Add any remaining syllables as the last pada
    if current_pada:
        padas.append(''.join(current_pada))

    return padas

###############################################################################


def trim_matra(line: str) -> str:
    """Trim matra from the end of a string"""
    # TODO: If there is no general utility, consider removing this function.
    answer = line
    if line[-1] in [ANUSWARA, HALANTA, VISARGA]:
        answer = line[:-1]
    if answer[-1] in ALL_MATRA:
        answer = answer[:-1]
    return answer


###############################################################################


def is_laghu(syllable: str, next_syllable: str = "") -> bool:
    """
    Determine if a syllable is laghu (short) or guru (long).
    
    Parameters:
    ----------
    syllable : str
        The syllable to check
    next_syllable : str, optional
        The following syllable, used to check for conjunct consonants
    
    Returns:
    -------
    bool
        True if the syllable is laghu, False if it's guru
    """
    # Check for anusvara or visarga
    if any(char in syllable for char in [ANUSWARA, VISARGA]):
        return False
    
    # # Check for long vowels
    # if any(char in syllable for char in MATRA[1:] + EXTENDED_MATRA):
    #     return False
    
    if any(char in syllable for char in [MATRA[i] for i in [0, 2, 4, 6, 8, 9, 10, 11, 12]] + EXTENDED_MATRA):
        return False
    
    if any(char in syllable for char in [SWARA[i] for i in [1, 3, 5, 7, 9, 10, 11, 12, 13]] + EXTENDED_SWARA):
        return False

    # Check if the syllable ends with halanta
    if syllable.endswith(HALANTA):
        return False
    
    # Check if the next syllable starts with a conjunct consonant
    if next_syllable:
        if any(char in VYANJANA for char in next_syllable[:2]):
            return False
    
    return True


def toggle_matra(syllable: str) -> str:
    """Change the Laghu syllable to Guru and Guru to Laghu (if possible)"""
    if syllable[-1] in MATRA:
        index = MATRA.index(syllable[-1])
        if index in [2, 4, 6, 8]:
            return syllable[:-1] + MATRA[index - 1]
        if index in [1, 3, 5, 7]:
            return syllable[:-1] + MATRA[index + 1]

    if syllable in SWARA:
        index = SWARA.index(syllable)
        if index in [0, 2, 4, 6, 8]:
            return SWARA[index + 1]
        if index in [1, 3, 5, 7, 9]:
            return SWARA[index - 1]


###############################################################################


def marker_to_swara(m: str) -> str:
    """Convert a Matra to corresponding Swara"""
    if m in ARTIFICIAL_MATRA:
        return m[1:]

    if m in MATRA:
        m_idx = MATRA.index(m)
        return SWARA[m_idx + 1]
    elif m in EXTENDED_MATRA:
        m_idx = EXTENDED_MATRA.index(m)
        return EXTENDED_SWARA[m]
    return None


def swara_to_marker(s: str) -> str:
    """Convert a Swara to correponding Matra"""
    if s == SWARA[0]:
        return f"-{s}"

    if s in SWARA:
        s_idx = SWARA.index(s)
        return MATRA[s_idx - 1]
    if s in EXTENDED_SWARA[:-1]:
        s_idx = EXTENDED_SWARA.index(s)
        return EXTENDED_MATRA[s_idx]
    return None


###############################################################################


def get_anunaasika(ch: str) -> str:
    """Get the appropriate anunaasika from the character's group"""
    MA = AUSHTHYA[4]
    if ch == "":
        return MA
    if ch in VYANJANA:
        i = VYANJANA.index(ch)
        if i < 25:
            return VYANJANA[int(i / 5) * 5 + 4]
        else:
            return ANUSWARA
    else:
        return ANUSWARA


def fix_anuswara(text: str) -> str:
    """
    Check every anuswaara in the text and change to anunaasika if applicable
    """
    output_chars = []
    if text:
        for idx in range(len(text) - 1):
            char = text[idx]
            next_char = text[idx + 1]
            if char == ANUSWARA and next_char in VARGIYA:
                anunasika = get_anunaasika(next_char)
                output_chars.append(anunasika)
                output_chars.append(HALANTA)
            else:
                output_chars.append(char)
        output_chars.append(text[-1])
    return "".join(output_chars)


###############################################################################


def get_syllables_word(word: str, technical: bool = False) -> List[str]:
    """Get syllables from a Sanskrit (Devanagari) word

    Parameters
    ----------
    word : str
        Sanskrit (Devanagari) word to get syllables from.
        Spaces, if present, are ignored.
    technical : bool, optional
        If True, ensures that each element contains at most
        one Swara or Vyanjana.
        The default is False.

    Returns
    -------
    List[str]
        List of syllables
    """
    word = clean(word, spaces=False)
    wlen = len(word)
    word_syllables = []

    current = ""
    i = 0
    while i < wlen:
        curr_ch = word[i]
        current += curr_ch
        i += 1
        # words split to start at START_CHARS
        start_chars = VARNA + SPECIAL
        if technical:
            start_chars += AYOGAVAAHA_COMMON
        while i < wlen and word[i] not in start_chars:
            current += word[i]
            i += 1
        if current[-1] != HALANTA or i == wlen or technical:
            word_syllables.append(current)
            current = ""
    return word_syllables


def get_syllables_flat_improved(text: str, technical: bool = False) -> List[str]:
    """Get syllables from a Sanskrit (Devanagari) text with improved handling of halanta and avagraha.

    Parameters
    ----------
    text : str
        Sanskrit (Devanagari) text to get syllables from
    technical : bool, optional
        If True, ensures that each element contains at most
        one Swara or Vyanjana.
        The default is False.

    Returns
    -------
    List[str]
        List of syllables
    """
    lines = split_lines(text.strip())
    syllables = []
    for line in lines:
        words = line.split()
        for word in words:
            word_syllables = get_syllables_word(word, technical)
            
            # Merge the last syllable with the second last if it ends with halanta
            if len(word_syllables) > 1 and word_syllables[-1].endswith(HALANTA):
                word_syllables[-2] += word_syllables[-1]
                word_syllables.pop()
            
            # Remove avagraha from syllables
            word_syllables = [syl for syl in word_syllables if syl != AVAGRAHA]
            
            syllables.extend(word_syllables)
    return syllables

# Helper function to count syllables
def count_syllables(text: str) -> int:
    return len(get_syllables_flat_improved(text))

def get_syllables(text: str, technical: bool = False) -> List[List[List[str]]]:
    """Get syllables from a Sanskrit (Devanagari) text

    Parameters
    ----------
    text : str
        Sanskrit (Devanagari) text to get syllables from
    technical : bool, optional
        If True, ensures that each element contains at most
        one Swara or Vyanjana.
        The default is False.

    Returns
    -------
    List[List[List[str]]]
        List of syllables in a nested list format
        Nesting Levels: Text -> Lines -> Words
    """
    lines = split_lines(text.strip())
    syllables = []
    for line in lines:
        words = line.split()
        line_syllables = []
        for word in words:
            word_syllables = get_syllables_word(word, technical)
            line_syllables.append(word_syllables)
        syllables.append(line_syllables)
    return syllables


###############################################################################


def split_varna_word(word: str, technical: bool = True) -> List[str]:
    """Obtain the Varna decomposition of a Sanskrit (Devanagari) word

    Parameters
    ----------
    word : str
        Sanskrit (Devanagari) word to be split.
    technical : bool, optional
        If True, a split, vowels and vowel signs are treated independently
        which is more useful for analysis,
        The default is True.

    Returns
    -------
    List[str]
        List of Varna
    """
    word_syllables = get_syllables_word(word, True)
    word_viccheda = []
    for syllable in word_syllables:
        if syllable[0] in ALL_SWARA:
            word_viccheda.append(syllable[0])
            if len(syllable) > 1:
                word_viccheda.append(syllable[1])
            # TODO: Will this ever be the case?
            if len(syllable) > 2:
                LOGGER.warning(f"Long SWARA: {syllable}")
                word_viccheda.append(syllable[2:])
        elif syllable[0] in ALL_VYANJANA:
            word_viccheda.append(syllable[0] + HALANTA)
            if len(syllable) == 1:
                word_viccheda.append(ARTIFICIAL_MATRA_A)
            if len(syllable) > 1:
                if syllable[1] in AYOGAVAAHA_COMMON:
                    word_viccheda.append(ARTIFICIAL_MATRA_A)
                if syllable[1] != HALANTA:
                    word_viccheda.append(syllable[1])
            # TODO: Will this ever be the case?
            if len(syllable) > 2:
                LOGGER.warning(f"Long VYANJANA: {syllable}")
                word_viccheda.append(syllable[2:])
        else:
            word_viccheda.append(syllable)

    if not technical:
        real_word_viccheda = []
        for varna in word_viccheda:
            if varna in ARTIFICIAL_MATRA + ALL_MATRA:
                real_word_viccheda.append(marker_to_swara(varna))
            elif varna in AYOGAVAAHA_COMMON:
                real_word_viccheda[-1] += varna
            else:
                real_word_viccheda.append(varna)
        word_viccheda = real_word_viccheda
    return word_viccheda


def split_varna(
    text: str, technical: bool = True, flat: bool = False
) -> List[List[List[str]]] or List[str]:
    """Obtain the Varna decomposition of a Sanskrit (Devanagari) text

    Parameters
    ----------
    word : str
        Sanskrit (Devanagari) text to be split.
    technical : bool, optional
        If True, a split, vowels and vowel signs are treated independently
        which is more useful for analysis,
        The default is True.
    flat : bool, optional
        If True, a single list is returned instead of nested lists.
        The default is False.

    Returns
    -------
    List[List[List[str]]] or List[str]

        Varna decomposition of the text in a nested list format.
        Nesting Levels: Text -> Lines -> Words

        - Varna decomposition of each word is a List[char].
        - List of Varna decomposition of each word from a line.
        - List of Varna decomposition of each line from the text.

        If `flat=True`, Varna decomposition of the entire text is presented
        as a single list, also containing whitespace markers.
        Lines are separated by a newline character '\\n' and words are
        separated by a space character ' '.
    """

    lines = split_lines(text.strip())
    viccheda = []
    num_lines = len(lines)
    for line_idx, line in enumerate(lines):
        words = line.split()
        line_viccheda = []
        num_words = len(words)
        for word_idx, word in enumerate(words):
            word_viccheda = split_varna_word(word, technical)
            if flat:
                line_viccheda.extend(word_viccheda)
                if word_idx != num_words - 1:
                    line_viccheda.append(" ")
            else:
                line_viccheda.append(word_viccheda)
        if flat:
            viccheda.extend(line_viccheda)
            if line_idx != num_lines - 1:
                viccheda.append("\n")
        else:
            viccheda.append(line_viccheda)
    return viccheda


def join_varna(viccheda: str, technical: bool = True) -> str:
    """
    Join Varna decomposition to form a Sanskrit (Devanagari) word

    Parameters
    ----------
    viccheda : list
        Viccheda output obtained by `split_varna_word` with `technical=True`
        (or output of `split_varna` with `technical=True` and `flat=True`)
        IMPORTANT: `technical=True` is necessary.
    technical : bool
        WARNING: Currently unused.
        Value of the same parameter passed to `split_varna_word`


    NOTE
    ----
        Currently only works for the viccheda generated with `technical=True`

    Returns
    -------
    s : str
        Sanskrit word
    """
    word = []
    i = 0
    while i < len(viccheda):
        curr_varna = viccheda[i]
        next_varna = ""
        if i < len(viccheda) - 1:
            next_varna = viccheda[i + 1]

        i += 1

        if curr_varna in [" ", "\n"]:
            word.append(curr_varna)
            continue

        if curr_varna[0] in ALL_SWARA + SPECIAL:
            word.append(curr_varna[0])
            if curr_varna[-1] in AYOGAVAAHA_COMMON:
                word.append(curr_varna[-1])
        if curr_varna[-1] == HALANTA:
            if next_varna in [" ", "\n"]:
                word.append(curr_varna)
                continue
            if next_varna == "":
                word.append(curr_varna)
                break
            if next_varna[-1] == HALANTA:
                word.append(curr_varna)
            if next_varna[0] in ALL_SWARA:
                i += 1
                word.append(curr_varna[:-1])
                if next_varna[0] != SWARA[0]:
                    word.append(marker_to_swara(next_varna[0]))
                if next_varna[-1] == VISARGA:
                    # NOTE: This was mostly meant to handle version with
                    # `technical=False`
                    LOGGER.warning(
                        f"Next Varna is SWARA + VISARGA: {next_varna}"
                    )
                    word.append(next_varna[-1])
            if next_varna in AYOGAVAAHA_COMMON:
                i += 1
                word.append(curr_varna[:-1] + next_varna)
            if next_varna in ARTIFICIAL_MATRA + ALL_MATRA:
                i += 1
                word.append(curr_varna[:-1])
                if next_varna != ARTIFICIAL_MATRA_A:
                    word.append(next_varna)
        if curr_varna in ARTIFICIAL_MATRA + ALL_MATRA + AYOGAVAAHA_COMMON:
            word.append(curr_varna)

    return "".join(word)


###############################################################################

###############################################################################
# Ucchaarana Sthaana Module
# ------------------------

STHAANA = {
    "S_K": ["अ", "आ"] + KANTHYA + ["ह"] + [VISARGA],
    "S_T": ["इ", "ई"] + TALAVYA + ["य", "श"],
    "S_M": ["ऋ", "ॠ"] + MURDHANYA + ["र", "ष"],
    "S_D": ["ऌ", "ॡ"] + DANTYA + ["ल", "स"],
    "S_O": ["उ", "ऊ"] + AUSHTHYA + [UPADHMANIYA],
    "S_N": VARGA_PANCHAMA + [ANUSWARA],
    "S_KT": ["ए", "ऐ"],
    "S_KO": ["ओ", "औ"],
    "S_DO": ["व"],
    "S_JM": [JIHVAAMULIYA],
}

STHAANA_NAMES = {
    "S_K": "कण्ठः",
    "S_T": "तालु",
    "S_M": "मूर्धा",
    "S_D": "दन्ताः",
    "S_O": "ओष्ठौ",
    "S_N": "नासिका",
    "S_KT": "कण्ठतालु",
    "S_KO": "कण्ठौष्ठम्",
    "S_DO": "दन्तौष्ठम्",
    "S_JM": "जिह्वामूलम्",
}

###############################################################################

AABHYANTARA = {
    "A_SP": VARGIYA,
    "A_ISP": ANTAHSTHA,
    "A_IVVT": USHMA + [JIHVAAMULIYA, UPADHMANIYA],
    "A_VVT": SWARA[1:] + [CHANDRABINDU, ANUSWARA, VISARGA],
    "A_SVT": SWARA[:1],
}

AABHYANTARA_NAMES = {
    "A_SP": "स्पृष्टः",
    "A_ISP": "ईषत्स्पृष्टः",
    "A_IVVT": "ईषद्विवृतः",
    "A_VVT": "विवृतः",
    "A_SVT": "संवृतः",
}

###############################################################################

BAAHYA = {
    "B_VVR": resolve_pratyaahaara("खर्")[0],
    "B_SVR": resolve_pratyaahaara("हश्")[0] + SWARA,
    "B_SW": resolve_pratyaahaara("खर्")[0],
    "B_ND": resolve_pratyaahaara("हश्")[0] + SWARA,
    "B_GH": resolve_pratyaahaara("हश्")[0] + SWARA,
    "B_AGH": resolve_pratyaahaara("खर्")[0],
    "B_AP": (
        VARGA_PRATHAMA
        + VARGA_TRITIYA
        + VARGA_PANCHAMA
        + resolve_pratyaahaara("यण्")[0]
    )
    + SWARA
    + [CHANDRABINDU, ANUSWARA],
    "B_MP": (VARGA_DWITIYA + VARGA_CHATURTHA + resolve_pratyaahaara("शल्")[0])
    + [VISARGA, JIHVAAMULIYA, UPADHMANIYA],
    "B_U": SWARA,
    "B_ANU": [s + ANUDATTA for s in SWARA],
    "B_SWA": [s + SWARITA for s in SWARA],
}

BAAHYA_NAMES = {
    "B_VVR": "विवारः",
    "B_SVR": "संवारः",
    "B_SW": "श्वासः",
    "B_ND": "नादः",
    "B_GH": "घोषः",
    "B_AGH": "अघोषः",
    "B_AP": "अल्पप्राणः",
    "B_MP": "महाप्राणः",
    "B_U": "उदात्तः",
    "B_ANU": "अनुदात्तः",
    "B_SWA": "स्वरितः",
}

###############################################################################

UCCHAARANA = dict(**STHAANA, **AABHYANTARA, **BAAHYA)
UCCHAARANA_NAMES = dict(**STHAANA_NAMES, **AABHYANTARA_NAMES, **BAAHYA_NAMES)

###############################################################################


def get_ucchaarana_vector(letter: str, abbrev=False) -> Dict[str, int]:
    """
    Get ucchaarana sthaana and prayatna based vector of a letter

    Parameters
    ----------
    letter : str
        Sanskrit letter
    abbrev : bool
        If True, the output will contain English abbreviations
        otherwise, the output will contain Sanskrit names.
        The default is False.

    Returns
    -------
    vector : Dict[str, int]
        One-hot vector indicating utpatti sthaana, aabhyantara prayatna and
        baahya prayatna of a letter
    """
    varna = letter.replace(HALANTA, "") if letter.endswith(HALANTA) else letter
    if abbrev:

        def ucchaarana_name(s):
            return s

    else:

        def ucchaarana_name(s):
            return UCCHAARANA_NAMES[s]

    vector = {ucchaarana_name(k): 0 for k in UCCHAARANA}
    for s, varna_list in UCCHAARANA.items():
        if varna in varna_list:
            vector[ucchaarana_name(s)] = 1

    return vector


def get_ucchaarana_vectors(
    word: str, abbrev: bool = False
) -> List[Tuple[str, Dict[str, int]]]:
    """
    Get ucchaarana sthaana and prayatna based vector of a word or text

    Parameters
    ----------
    word : str
        Sanskrit word (or text)
    abbrev : bool
        If True, the output will contain English abbreviations
        otherwise, the output will contain Sanskrit names.
        The default is False.

    Returns
    -------
    vectors : List[Tuple[str, Dict[str, int]]]
        List of (letter, vector)
    """
    letters = []
    for letter in split_varna_word(word, technical=False):
        if [v for v in AYOGAVAAHA_COMMON if v in letter]:
            letters.extend(letter)
        else:
            letters.append(letter)
    return [
        (letter, get_ucchaarana_vector(letter, abbrev)) for letter in letters
    ]


###############################################################################


def get_signature_letter(letter: str, abbrev: bool = False) -> Dict[str, str]:
    """
    Get ucchaarana sthaana and prayatna based signature of a letter

    Parameters
    ----------
    letter : str
        Sanskrit letter
    abbrev : bool
        If True, the output will contain English abbreviations
        otherwise, the output will contain Sanskrit names.
        The default is False.

    Returns
    -------
    signature : Dict[str, str]
        utpatti sthaana, aabhyantara prayatna and baahya prayatna of a letter
    """
    sthaana = get_ucchaarana_letter(letter, dimension=0, abbrev=abbrev)
    aabhyantara = get_ucchaarana_letter(letter, dimension=1, abbrev=abbrev)
    baahya = get_ucchaarana_letter(letter, dimension=2, abbrev=abbrev)

    signature = {
        "sthaana": sthaana,
        "aabhyantara": aabhyantara,
        "baahya": baahya,
    }
    return signature


def get_signature_word(
    word: str, abbrev: bool = False
) -> List[Tuple[str, Dict[str, str]]]:
    """
    Get ucchaarana sthaana and prayatna based signature of a word

    Parameters
    ----------
    word : str
        Sanskrit word (or text)
        Caution: If multiple words are provided,
        the spaces are not included in the output list.
    abbrev : bool
        If True, the output will contain English abbreviations
        otherwise, the output will contain Sanskrit names.
        The default is False.

    Returns
    -------
    List[Tuple[str, Dict[str, str]]]
        List of (letter, signature)

    """
    letters = []
    for letter in split_varna_word(word, technical=False):
        if [v for v in AYOGAVAAHA_COMMON if v in letter]:
            letters.extend(letter)
        else:
            letters.append(letter)
    return [
        (letter, get_signature_letter(letter, abbrev)) for letter in letters
    ]


def get_signature(
    text: str, abbrev: bool = False
) -> List[List[List[Tuple[str, Dict[str, str]]]]]:
    """
    Get ucchaarana list of a Sanskrit text

    Parameters
    ----------
    text : str
        Sanskrit text (can contain newlines, spaces)
    abbrev : bool
        If True, the output will contain English abbreviations
        otherwise, the output will contain Sanskrit names.
        The default is False.

    Returns
    -------
    List[List[List[Tuple[str, Dict[str, str]]]]]
        List of (letter, signature) for words in a nested list format
        Nesting Levels: Text -> Lines -> Words
    """
    lines = split_lines(text.strip())
    signature = []
    for line in lines:
        words = line.split()
        line_signature = []
        for word in words:
            word_signature = get_signature_word(word, abbrev)
            line_signature.append(word_signature)
        signature.append(line_signature)
    return signature


###############################################################################


def get_ucchaarana_letter(
    letter: str, dimension: int = 0, abbrev: bool = False
) -> str:
    """
    Get ucchaarana sthaana or prayatna of a letter

    Parameters
    ----------
    letter : str
        Sanskrit letter
    dimension : int
        - 0: sthaana
        - 1: aabhyantara prayatna
        - 2: baahya prayatna

        The default is 0.
    abbrev : bool
        If True,
            The output will contain English abbreviations
        Otherwise,
            The output will contain Sanskrit names

        The default is False.

    Returns
    -------
    str
        ucchaarana sthaana or prayatna of a letter
    """
    varna = letter.replace(HALANTA, "") if letter.endswith(HALANTA) else letter
    ucchaarana = []
    _UCCHAARANA = [STHAANA, AABHYANTARA, BAAHYA]
    _NAMES = [STHAANA_NAMES, AABHYANTARA_NAMES, BAAHYA_NAMES]

    if abbrev:

        def ucchaarana_name(s):
            return s

        join_str = "-"
    else:

        def ucchaarana_name(s):
            return _NAMES[dimension][s]

        join_str = " "

    for s, varna_list in _UCCHAARANA[dimension].items():
        if varna in varna_list:
            ucchaarana.append(ucchaarana_name(s))

    if len(ucchaarana) > 1 and not abbrev:
        ucchaarana.append("च")

    return join_str.join(ucchaarana)


def get_ucchaarana_word(
    word: str, dimension: int = 0, abbrev: bool = False
) -> List[Tuple[str, str]]:
    """
    Get ucchaarana of a word

    Parameters
    ----------
    word : str
        Sanskrit word (or text)

        **Caution**: If multiple words are provided, the spaces are not included in
        the output list
    dimension : int
        - 0: sthaana
        - 1: aabhyantara prayatna
        - 2: baahya prayatna

        The default is 0.
    abbrev : bool
        If True,
            The output will contain English abbreviations
        Otherwise,
            The output will contain Sanskrit names

        The default is False.

    Returns
    -------
    List[Tuple[str, str]]
        List of (letter, ucchaarana)

    """
    letters = []
    for letter in split_varna_word(word, technical=False):
        if [v for v in AYOGAVAAHA_COMMON if v in letter]:
            letters.extend(letter)
        else:
            letters.append(letter)
    return [
        (letter, get_ucchaarana_letter(letter, dimension, abbrev))
        for letter in letters
    ]


def get_ucchaarana(
    text: str, dimension: int = 0, abbrev: bool = False
) -> List[List[List[Tuple[str, str]]]]:
    """
    Get ucchaarana list of a Sanskrit text

    Parameters
    ----------
    text : str
        Sanskrit text (can contain newlines, spaces)
    dimension : int
        - 0: sthaana
        - 1: aabhyantara prayatna
        - 2: baahya prayatna

        The default is 0.
    abbrev : bool
        If True,
            The output will contain English abbreviations
        Otherwise,
            The output will contain Sanskrit names

        The default is False.

    Returns
    -------
    List[List[List[Tuple[str, str]]]]
        List of (letter, ucchaarana) for words in a nested list format
        Nesting Levels: Text -> Lines -> Words
    """
    lines = split_lines(text.strip())
    ucchaarana = []
    for line in lines:
        words = line.split()
        line_ucchaarana = []
        for word in words:
            word_ucchaarana = get_ucchaarana_word(word, dimension, abbrev)
            line_ucchaarana.append(word_ucchaarana)
        ucchaarana.append(line_ucchaarana)
    return ucchaarana


###############################################################################


def get_sthaana_letter(letter: str, abbrev: bool = False):
    """Wrapper for get_ucchaarana_letter for sthaana"""
    return get_ucchaarana_letter(letter, dimension=0, abbrev=abbrev)


def get_sthaana_word(word: str, abbrev: bool = False):
    """Wrapper for get_ucchaarana_word for sthaana"""
    return get_ucchaarana_word(word, dimension=0, abbrev=abbrev)


def get_sthaana(text: str, abbrev: bool = False):
    """Wrapper for get_ucchaarana for sthaana"""
    return get_ucchaarana(text, dimension=0, abbrev=abbrev)


# --------------------------------------------------------------------------- #


def get_aabhyantara_letter(letter: str, abbrev: bool = False):
    """Wrapper for get_ucchaarana_letter for aabhyantara"""
    return get_ucchaarana_letter(letter, dimension=1, abbrev=abbrev)


def get_aabhyantara_word(word: str, abbrev: bool = False):
    """Wrapper for get_ucchaarana_word for aabhyantara"""
    return get_ucchaarana_word(word, dimension=1, abbrev=abbrev)


def get_aabhyantara(text: str, abbrev: bool = False):
    """Wrapper for get_ucchaarana for aabhyantara"""
    return get_ucchaarana(text, dimension=1, abbrev=abbrev)


# --------------------------------------------------------------------------- #


def get_baahya_letter(letter: str, abbrev: bool = False):
    """Wrapper for get_ucchaarana_letter for baahya"""
    return get_ucchaarana_letter(letter, dimension=2, abbrev=abbrev)


def get_baahya_word(word: str, abbrev: bool = False):
    """Wrapper for get_ucchaarana_word for baahya"""
    return get_ucchaarana_word(word, dimension=2, abbrev=abbrev)


def get_baahya(text: str, abbrev: bool = False):
    """Wrapper for get_ucchaarana for baahya"""
    return get_ucchaarana(text, dimension=2, abbrev=abbrev)



###############################################################################
from tqdm import tqdm
import os
import re
from typing import List, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BeamScorer, BeamSearchScorer, LogitsProcessor, LogitsProcessorList
import torch
import logging
from logging.handlers import RotatingFileHandler

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "anushtup_generation.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler) 

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")

model_name = "facebook/nllb-200-distilled-600M"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model.to(mps_device)

def split_into_padas(text):
    clean_text = clean(text, spaces=True)
    syllables = get_syllables_flat_improved(clean_text)
    padas = []
    for i in range(0, len(syllables), 8):
        padas.append(''.join(syllables[i:i+8]))
    return padas


def get_characters(word: str, technical: bool = False) -> List[str]:
   """Get syllables from a Sanskrit (Devanagari) word


   Parameters
   ----------
   word : str
       Sanskrit (Devanagari) word to get syllables from.
       Spaces, if present, are ignored.
   technical : bool, optional
       If True, ensures that each element contains at most
       one Swara or Vyanjana.
       The default is False.


   Returns
   -------
   List[str]
       List of syllables
   """
   word = clean(word, spaces=False)
   wlen = len(word)
   word_syllables = []


   current = ""
   i = 0
   while i < wlen:
       curr_ch = word[i]
       current += curr_ch
       i += 1
       # words split to start at START_CHARS
       start_chars = VARNA + SPECIAL
       while i < wlen and word[i] not in start_chars:
           current += word[i]
           i += 1
       # if current[-1] != HALANTA or i == wlen or technical:
       #     word_syllables.append(current)
       #     current = ""
       word_syllables.append(current)
       current = ""
   return word_syllables


def get_sanskrit_token_ids(tokenizer, target_lang="san_Deva"):
    all_token_ids = list(range(tokenizer.vocab_size))
    all_tokens = tokenizer.convert_ids_to_tokens(all_token_ids)
    sanskrit_token_ids = [
        token_id for token_id, token in zip(all_token_ids, all_tokens)
        if token.startswith(target_lang) or any(char in token for char in ALPHABET)
    ]
    return sanskrit_token_ids

def check_strict_anushtup_constraint(syllables):
    num_syllables = len(syllables)
    if num_syllables > 32:
        return False
    
    complete_padas = num_syllables // 8
    remaining_syllables = num_syllables % 8
    
    for i in range(complete_padas):
        pada_syllables = syllables[i*8:(i+1)*8]
        # 5th syllable must be laghu
        if len(pada_syllables) >= 5 and not is_laghu(pada_syllables[4]):
            return False
        # 6th syllable must be guru
        if len(pada_syllables) >= 6 and is_laghu(pada_syllables[5]):
            return False
    
    # Check incomplete pada if it exists
    if remaining_syllables > 0:
        last_pada_syllables = syllables[complete_padas*8:]
        if remaining_syllables >= 5 and not is_laghu(last_pada_syllables[4]):
            return False
        if remaining_syllables >= 6 and is_laghu(last_pada_syllables[5]):
            return False
    
    return True

def evaluate_strict_anushtup_fit(syllables):
    score = 1.0
    violations = []
    num_syllables = len(syllables)
    complete_padas = num_syllables // 8
    remaining_syllables = num_syllables % 8

    for i in range(complete_padas):
        pada_syllables = syllables[i*8:(i+1)*8]
        if len(pada_syllables) >= 5 and not is_laghu(pada_syllables[4]):
            score -= 0.2
            violations.append(f"Pada {i+1}, 5th syllable not laghu")
        if len(pada_syllables) >= 6 and is_laghu(pada_syllables[5]):
            score -= 0.2
            violations.append(f"Pada {i+1}, 6th syllable not guru")

    if remaining_syllables > 0:
        last_pada_syllables = syllables[complete_padas*8:]
        if remaining_syllables >= 5 and not is_laghu(last_pada_syllables[4]):
            score -= 0.2
            violations.append(f"Last pada, 5th syllable not laghu")
        if remaining_syllables >= 6 and is_laghu(last_pada_syllables[5]):
            score -= 0.2
            violations.append(f"Last pada, 6th syllable not guru")

    # Penalize for incorrect syllable count
    if num_syllables < 32:
        score -= (32 - num_syllables) * 0.1
        violations.append(f"Undergeneration: {num_syllables}/32 syllables")
    elif num_syllables > 32:
        score -= (num_syllables - 32) * 0.2
        violations.append(f"Overgeneration: {num_syllables}/32 syllables")

    return max(0, score), violations

class ProgressTrackingLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.progress_bar = tqdm(total=32, desc="Generating verse", unit="syllables")
        self.tokenizer = tokenizer
        self.current_syllables = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        new_syllables = count_syllables(text)
        if new_syllables > self.current_syllables:
            self.progress_bar.update(new_syllables - self.current_syllables)
            self.current_syllables = new_syllables
        return scores

    def close(self):
        self.progress_bar.close()

class SanskritAnushtupLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, num_beams: int, sanskrit_token_ids):
        self.tokenizer = tokenizer
        self.num_beams = num_beams
        self.sanskrit_token_ids = set(sanskrit_token_ids)
        self.generation_step = 0

    def is_anushtup_critical_position(self, syllable_index: int) -> bool:
        pada_position = syllable_index % 8
        return pada_position in [4, 5, 6]  # 5th, 6th, or 7th position in pada

    def get_required_syllable_type(self, syllable_index: int) -> bool:
        pada_position = syllable_index % 8
        pada_number = syllable_index // 8

        if pada_position == 4:  # 5th syllable
            return True  # must be laghu
        elif pada_position == 5:  # 6th syllable
            return False  # must be guru
        elif pada_position == 6:  # 7th syllable
            return pada_number % 2 == 1  # laghu for 2nd and 4th padas, guru for 1st and 3rd
        else:
            return None  # No specific requirement
        
    

    def enforce_syllable_type(self, current_text: str, should_be_laghu: bool, scores: torch.FloatTensor) -> torch.FloatTensor:
        current_syllables = get_syllables_flat_improved(current_text)
        pada_position = len(current_syllables) % 8  # Calculate the position within the pada
        pada_number = len(current_syllables) // 8   # Calculate the current pada number (1-based index)
    

        for token in range(self.tokenizer.vocab_size):
            if token not in self.sanskrit_token_ids:
                continue
            next_token = self.tokenizer.decode([token])
            candidate_text = current_text + next_token
            candidate_syllables = get_syllables_flat_improved(candidate_text)
            
            if len(candidate_syllables) == len(current_syllables) + 1:
                new_syllable = candidate_syllables[-1]
                next_syllable = ""  # We don't need to check the next syllable here

                # Check for halanta in the 6th position
                if pada_position == 5 and HALANTA in get_characters(new_syllable)[0]:
                    scores[token] = float('-inf')

                # Check for hrasva or deergha in the 7th position
                if pada_position == 6:
                    if pada_number % 2 == 1:  
                        if any(char in next_token for char in [MATRA[i] for i in [0, 2, 4, 6, 8, 9, 10, 11, 12]] + EXTENDED_MATRA + [SWARA[i] for i in [1, 3, 5, 7, 9, 10, 11, 12, 13]] + EXTENDED_SWARA):
                            scores[token] = float('-inf')
                            continue
                    else:  
                        if any(char in next_token for char in [MATRA[i] for i in [1, 3, 5, 7]] + [SWARA[i] for i in [0, 2, 4, 6, 8]]):
                            scores[token] = float('-inf')
                            continue

                if should_be_laghu == is_laghu(new_syllable, next_syllable):
                    scores[token] = 0
                else:
                    scores[token] = float('-inf')
        
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.generation_step += 1
        batch_size, seq_len = input_ids.shape
        
        for batch_idx in range(batch_size):
            current_text = self.tokenizer.decode(input_ids[batch_idx], skip_special_tokens=True)
            current_syllables = get_syllables_flat_improved(current_text)
            
            logger.info(f"\n=== Generation Step {self.generation_step}, Batch {batch_idx} ===")
            logger.info(f"Current text: {current_text}")
            logger.info(f"Current syllable count: {len(current_syllables)}")
            
            if len(current_syllables) >= 32:
                logger.info("Reached 32 syllables, allowing only EOS token")
                scores[batch_idx].fill_(float('-inf'))
                scores[batch_idx, self.tokenizer.eos_token_id] = 0
                continue
            
            is_critical_position = self.is_anushtup_critical_position(len(current_syllables))
            
            if is_critical_position:
                should_be_laghu = self.get_required_syllable_type(len(current_syllables))
                scores[batch_idx] = self.enforce_syllable_type(current_text, should_be_laghu, scores[batch_idx])
                logger.info(f"Enforced {'laghu' if should_be_laghu else 'guru'} syllable at position {len(current_syllables) + 1}")
            else:
                # For non-critical positions, we still ensure only Sanskrit tokens are used
                non_sanskrit_mask = torch.ones_like(scores[batch_idx], dtype=torch.bool)
                non_sanskrit_mask[list(self.sanskrit_token_ids)] = False
                scores[batch_idx][non_sanskrit_mask] = float('-inf')
            
            # Ensure we don't generate more than 32 syllables
            # if len(current_syllables) == 31:
            #     # For the last syllable, we need to ensure it completes the verse
            #     scores[batch_idx] = self.enforce_syllable_type(current_text, None, scores[batch_idx])
            #     logger.info("Enforcing final syllable to complete 32 syllables")
        
        return scores


def generate_anushtup_verse(model, tokenizer, input_text: str, target_lang: str = "san_Deva") -> str:
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    
    sanskrit_token_ids = get_sanskrit_token_ids(tokenizer, target_lang)
    
    num_beams = 10
    
    # Initialize our custom LogitsProcessor
    anushtup_processor = SanskritAnushtupLogitsProcessor(tokenizer, num_beams, sanskrit_token_ids)
    progress_processor = ProgressTrackingLogitsProcessor(tokenizer)

    logger.info(f"Generating Anushtup verse for input: {input_text}")
    
    # Generate
    output_ids = model.generate(
        input_ids,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
        max_length=100,
        num_beams=num_beams,
        num_return_sequences=1,
        logits_processor=LogitsProcessorList([anushtup_processor, progress_processor]),
        # early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # top_p=0.92,
        # do_sample=True,  # Ensure deterministic behavior
        length_penalty=1.0,  # Neutral length penalty
        repetition_penalty=1.0,  # No repetition penalty
    )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    logger.info(f"Generated text: {generated_text}")
    
    # Perform Anushtup meter check
    meter_analysis = check_anushtup_meter(generated_text)
    print_anushtup_analysis(meter_analysis)
    
    return generated_text, meter_analysis

def check_anushtup_meter(verse: str) -> dict:
    """
    Check if a given verse follows the Anushtup meter rules.
    
    Parameters:
    ----------
    verse : str
        The Sanskrit verse to check
    
    Returns:
    -------
    dict
        A dictionary containing the analysis results
    """
    syllables = get_syllables_flat_improved(verse)
    syllables_with_type = get_syllables_with_type(verse)
    
    result = {
        "is_anushtup": True,
        "total_syllables": len(syllables),
        "padas": [],
        "violations": []
    }
    
    # Check total syllable count
    if len(syllables) != 32:
        result["is_anushtup"] = False
        result["violations"].append(f"Total syllable count is {len(syllables)}, should be 32")
    
    # Analyze each pada
    for i in range(4):
        pada_start = i * 8
        pada_end = pada_start + 8
        pada_syllables = syllables_with_type[pada_start:pada_end]
        
        pada_analysis = {
            "pada_number": i + 1,
            "syllables": pada_syllables,
            "violations": []
        }
        
        # Check pada length
        if len(pada_syllables) != 8:
            result["is_anushtup"] = False
            pada_analysis["violations"].append(f"Pada {i+1} has {len(pada_syllables)} syllables, should be 8")
        
        # Check 5th syllable (should be laghu)
        if len(pada_syllables) >= 5 and pada_syllables[4][1] != 'L':
            result["is_anushtup"] = False
            pada_analysis["violations"].append(f"5th syllable is guru, should be laghu")
        
        # Check 6th syllable (should be guru)
        if len(pada_syllables) >= 6 and pada_syllables[5][1] != 'G':
            result["is_anushtup"] = False
            pada_analysis["violations"].append(f"6th syllable is laghu, should be guru")
        
        # Check 7th syllable (odd padas: guru, even padas: laghu)
        if len(pada_syllables) >= 7:
            expected_7th = 'G' if i % 2 == 0 else 'L'
            if pada_syllables[6][1] != expected_7th:
                result["is_anushtup"] = False
                pada_analysis["violations"].append(f"7th syllable is {'laghu' if pada_syllables[6][1] == 'L' else 'guru'}, should be {'guru' if expected_7th == 'G' else 'laghu'}")
        
        result["padas"].append(pada_analysis)
        result["violations"].extend([f"Pada {i+1}: {v}" for v in pada_analysis["violations"]])
    
    return result

def print_anushtup_analysis(analysis: dict):
    """
    Print a formatted analysis of the Anushtup meter check.
    
    Parameters:
    ----------
    analysis : dict
        The analysis dictionary returned by check_anushtup_meter
    """
    print(f"Anushtup Meter Analysis:")
    print(f"Total syllables: {analysis['total_syllables']}")
    print(f"Is valid Anushtup: {'Yes' if analysis['is_anushtup'] else 'No'}")
    
    if analysis['violations']:
        print("\nViolations:")
        for violation in analysis['violations']:
            print(f"- {violation}")
    
    print("\nPada Analysis:")
    for pada in analysis['padas']:
        print(f"\nPada {pada['pada_number']}:")
        for i, (syllable, type_) in enumerate(pada['syllables'], 1):
            print(f"  {i}. {syllable} ({type_})")
        if pada['violations']:
            print("  Violations:")
            for violation in pada['violations']:
                print(f"  - {violation}")

import json
with open("D:\BTP\constrained_decoding\data.json" ,'r') as f:
    data = json.load(f)

results = []


import json
from typing import List, Tuple

def get_syllables_with_type(text: str) -> List[Tuple[str, str]]:
    """Get syllables from text along with their type (laghu or guru)"""
    syllables = get_syllables_flat_improved(text)
    return [(syl, "L" if is_laghu(syl) else "G") for syl in syllables]

for verse in data['verses']:
    anushtup_gen = {}
    eng = verse['english']
    sans = verse['sanskrit']
    sanskrit_verse, meter_analysis = generate_anushtup_verse(model, tokenizer, eng)
    
    print("################SANSKRIT VERSE###############")
    print(sanskrit_verse)
    
    anushtup_gen['english'] = eng
    anushtup_gen['ground_truth'] = sans
    anushtup_gen['anushtup_generated'] = sanskrit_verse
    # anushtup_gen['meter_analysis'] = meter_analysis
    
    results.append(anushtup_gen)

with open("./constrained_decoding_full_anushtup.json", 'w') as file:
    json.dump(results, file, ensure_ascii=False, indent=4)

print(f"Results written to {file}")