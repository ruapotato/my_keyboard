from pygame.locals import *

# Display settings
WIDTH = 1024
HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Audio settings
SAMPLE_RATE = 44100
BASE_FREQ = 440.0  # A4

# Extended just intonation ratios
RATIOS = {
    'A': (1, 1),      # unison/tonic
    'B': (9, 8),      # major second
    'C': (5, 4),      # major third
    'D': (4, 3),      # perfect fourth
    'E': (3, 2),      # perfect fifth
    'F': (5, 3),      # major sixth
    'G': (15, 8),     # major seventh
    'H': (16, 15),    # minor second
    'I': (6, 5),      # minor third
    'J': (7, 5),      # augmented fourth
    'K': (8, 5),      # minor sixth
    'L': (9, 5),      # minor seventh
}

# Keyboard note ratios
NOTE_RATIOS = {
    K_q: 2.0,     # Octave up
    K_w: 1.5,     # Perfect fifth
    K_e: 1.25,    # Major third
    K_r: 1.125,   # Major second
    K_t: 1.0,     # Unison
    K_a: 0.833,   # Minor sixth
    K_s: 0.75,    # Perfect fourth
    K_d: 0.667,   # Perfect fifth below
    K_f: 0.625,   # Major third below
    K_g: 0.5,     # Octave down
    K_z: 0.375,   # Octave + fifth down
    K_x: 0.3125,  # Two octaves down
    K_c: 0.25,    # Two octaves down
    K_v: 0.125,   # Three octaves down
    K_b: 0.0625   # Four octaves down
}

# Keys for selecting ratios
RATIO_KEYS = {
    K_h: (1, 1),    # unison
    K_j: (9, 8),    # major whole tone
    K_k: (5, 4),    # major third
    K_l: (4, 3),    # perfect fourth
    K_SEMICOLON: (3, 2),  # perfect fifth
    K_QUOTE: (5, 3),      # major sixth
    K_RETURN: (15, 8),    # major seventh
    K_u: (16, 15),  # minor second
    K_i: (6, 5),    # minor third
    K_o: (7, 5),    # tritone
    K_p: (8, 5),    # minor sixth
    K_LEFTBRACKET: (9, 5),  # minor seventh
}

# Each ratio gets one letter
RATIO_TO_LETTER = {
    (1, 1): 'A',    # unison
    (9, 8): 'B',    # major whole tone
    (5, 4): 'C',    # major third
    (4, 3): 'D',    # fourth
    (3, 2): 'E',    # fifth
    (5, 3): 'F',    # sixth
    (15, 8): 'G',   # seventh
    (16, 15): 'H',  # minor second
    (6, 5): 'I',    # minor third
    (7, 5): 'J',    # tritone
    (8, 5): 'K',    # minor sixth
    (9, 5): 'L',    # minor seventh
}

# Display text for ratios
RATIO_TEXT = [
    "A: 1/1",  "B: 9/8",   "C: 5/4", 
    "D: 4/3",  "E: 3/2",   "F: 5/3",
    "G: 15/8", "H: 16/15", "I: 6/5",
    "J: 7/5",  "K: 8/5",   "L: 9/5"
]

# Visualization colors
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),    # Dark Red
    (0, 128, 0)     # Dark Green
]
