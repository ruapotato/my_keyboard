import pygame
import numpy as np
from typing import Dict
from constants import WIDTH, WHITE, BLACK, COLORS, RATIO_TO_LETTER, NOTE_RATIOS

class Visualizer:
    def __init__(self, width, height, y_offset):
        self.width = width
        self.height = height
        self.y_offset = y_offset
        self.colors = COLORS
        self.font = pygame.font.Font(None, 36)
        
        # Display text for ratios
        self.ratio_text = [
            "A: 1/1",  "B: 9/8",   "C: 5/4", 
            "D: 4/3",  "E: 3/2",   "F: 5/3",
            "G: 15/8", "H: 16/15", "I: 6/5",
            "J: 7/5",  "K: 8/5",   "L: 9/5"
        ]
        
    def get_note_name(self, key, selected_ratio):
        if key in NOTE_RATIOS:
            key_letter = RATIO_TO_LETTER[selected_ratio]
            position = list(NOTE_RATIOS.keys()).index(key)
            return f"{key_letter}{position + 1}"
        return ""
        
    def draw(self, screen, frequencies: Dict[int, float], selected_ratio, mode: str, status_text=None):
        base_freq = min(frequencies.values()) if frequencies else 440.0
        
        # Draw reference text in two columns
        col_width = WIDTH // 3
        for i, text in enumerate(self.ratio_text[:6]):
            text_surface = self.font.render(text, True, BLACK)
            screen.blit(text_surface, (WIDTH - col_width + 10, 20 + i * 25))
        
        for i, text in enumerate(self.ratio_text[6:]):
            text_surface = self.font.render(text, True, BLACK)
            screen.blit(text_surface, (WIDTH - col_width/2 + 10, 20 + i * 25))
            
        # Draw dividing line
        pygame.draw.line(screen, BLACK, (0, 400), (WIDTH, 400), 2)
            
        # Draw the waves
        for idx, (key, freq) in enumerate(frequencies.items()):
            color = self.colors[idx % len(self.colors)]
            points = []
            for x in range(self.width):
                t = x / self.width * 2
                freq_ratio = freq / base_freq
                phase = 2 * np.pi * (freq_ratio - 1) * t
                y = int(self.height/4 * np.sin(2 * np.pi * freq * t / 100 + phase))
                points.append((x, self.y_offset + self.height/2 + y))
                
            if len(points) > 1:
                pygame.draw.lines(screen, color, False, points, 2)
        
        # Draw active note numbers
        note_x = 20
        for key, freq in frequencies.items():
            color = self.colors[list(frequencies.keys()).index(key) % len(self.colors)]
            note_text = self.font.render(self.get_note_name(key, selected_ratio), True, color)
            screen.blit(note_text, (note_x, self.y_offset + self.height + 20))
            note_x += 40
            
    def draw_help(self, screen: pygame.Surface, mode: str):
        """Draw help text for current mode."""
        if mode == 'keyboard':
            help_texts = [
                "Left hand (playable notes): QWERT ASDFG ZXCVB",
                "Right hand (ratios): HJKL;' UIOP[",
                "Right hand (octave): 1-7",
                "Press ESC to return to menu"
            ]
        else:  # MIDI mode
            help_texts = [
                "Space: Stop playback and select new file",
                "R: Reset playback",
                "Press ESC to return to menu"
            ]
            
        for i, text in enumerate(help_texts):
            surface = self.font.render(text, True, BLACK)
            screen.blit(surface, (20, 20 + i * 40))
