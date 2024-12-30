import pygame
import os
import sys
from pygame.locals import *
from constants import (WIDTH, HEIGHT, WHITE, BLACK, 
                     NOTE_RATIOS, RATIO_KEYS)
from visualizer import Visualizer
from sound_manager import SoundManager

class Button:
    def __init__(self, x, y, width, height, text, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.color = (200, 200, 200)
        self.hover_color = (180, 180, 180)
        self.text_color = BLACK
        self.is_hovered = False

    def draw(self, screen):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)  # Border
        
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == MOUSEBUTTONDOWN:
            if self.is_hovered:
                return True
        return False

class MidiSelector:
    def __init__(self, x, y, width, height, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.font = font
        self.songs = []
        self.selected_index = 0
        self.visible_songs = 5  # Number of songs visible at once
        self.scroll_offset = 0
        self.load_songs()
        
        button_height = 40
        self.up_button = Button(x + width - 50, y, 50, button_height, "▲", font)
        self.down_button = Button(x + width - 50, y + height - button_height, 50, button_height, "▼", font)
        
    def load_songs(self):
        if os.path.exists("songs"):
            self.songs = [f for f in os.listdir("songs") 
                         if f.endswith(('.mid', '.midi'))]
        self.songs.sort()

    def draw(self, screen):
        if not self.songs:
            text = self.font.render("No MIDI files found in ./songs/", True, BLACK)
            screen.blit(text, (self.rect.x, self.rect.y))
            return

        visible_songs = self.songs[self.scroll_offset:self.scroll_offset + self.visible_songs]
        for i, song in enumerate(visible_songs):
            color = (180, 180, 180) if i + self.scroll_offset == self.selected_index else (220, 220, 220)
            song_rect = pygame.Rect(self.rect.x, self.rect.y + i * 50, self.rect.width - 60, 45)
            pygame.draw.rect(screen, color, song_rect)
            pygame.draw.rect(screen, BLACK, song_rect, 2)
            
            text = self.font.render(song[:25], True, BLACK)
            screen.blit(text, (song_rect.x + 10, song_rect.y + 10))

        self.up_button.draw(screen)
        self.down_button.draw(screen)

    def handle_event(self, event):
        if self.up_button.handle_event(event):
            self.scroll_offset = max(0, self.scroll_offset - 1)
        elif self.down_button.handle_event(event):
            max_offset = max(0, len(self.songs) - self.visible_songs)
            self.scroll_offset = min(max_offset, self.scroll_offset + 1)
        
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            for i in range(min(self.visible_songs, len(self.songs) - self.scroll_offset)):
                song_rect = pygame.Rect(self.rect.x, self.rect.y + i * 50, 
                                      self.rect.width - 60, 45)
                if song_rect.collidepoint(event.pos):
                    self.selected_index = i + self.scroll_offset
                    return True
        return False

    def get_selected_song(self):
        if self.songs:
            return os.path.join("songs", self.songs[self.selected_index])
        return None

class MusicApp:
    def __init__(self):
        """Initialize the music application."""
        pygame.init()
        pygame.mixer.init(44100, -16, 2, 2048)
        
        # Create display
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Just Intonation Music Player")
        
        # Initialize state variables
        self.mode = None  # 'keyboard', 'midi_corrected', or 'midi_uncorrected'
        self.selected_ratio = (1, 1)  # Start with unison ratio
        self.selected_octave = 4
        self.running = True
        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 36)
        
        # Create buttons for menu
        btn_width, btn_height = 400, 60
        x_pos = WIDTH//2 - btn_width//2
        self.menu_buttons = [
            Button(x_pos, HEIGHT//2 - 90, btn_width, btn_height, 
                  "1. Keyboard Mode", self.font),
            Button(x_pos, HEIGHT//2, btn_width, btn_height, 
                  "2. MIDI (Just Intonation)", self.font),
            Button(x_pos, HEIGHT//2 + 90, btn_width, btn_height, 
                  "3. MIDI (Original Tuning)", self.font)
        ]
        
        # Create MIDI selector
        self.midi_selector = MidiSelector(WIDTH//4, HEIGHT//4, WIDTH//2, HEIGHT//2, 
                                        self.small_font)
        
        # Initialize components
        self.sound_manager = None
        self.visualizer = Visualizer(WIDTH, 300, 450)  # width, height, y_offset - matching original values
        
    def draw_menu(self):
        """Draw the mode selection menu."""
        self.screen.fill(WHITE)
        
        # Draw title
        title = self.font.render("Just Intonation Music Player", True, BLACK)
        title_rect = title.get_rect(center=(WIDTH/2, HEIGHT/4))
        self.screen.blit(title, title_rect)
        
        # Draw buttons
        for button in self.menu_buttons:
            button.draw(self.screen)
        
        # Draw key mappings
        key_info = [
            "Keyboard Mode: QWERT ASDFG ZXCVB for notes",
            "Ratio selection: HJKL;'",
            "Octave selection: 1-7"
        ]
        
        y_start = HEIGHT*3/4
        for i, info in enumerate(key_info):
            text = self.small_font.render(info, True, BLACK)
            rect = text.get_rect(center=(WIDTH/2, y_start + i * 40))
            self.screen.blit(text, rect)
        
        pygame.display.flip()

    def draw_midi_selector(self):
        """Draw the MIDI file selector screen."""
        self.screen.fill(WHITE)
        
        title = self.font.render("Select MIDI File", True, BLACK)
        title_rect = title.get_rect(center=(WIDTH/2, 50))
        self.screen.blit(title, title_rect)
        
        self.midi_selector.draw(self.screen)
        
        # Draw instructions
        text = self.small_font.render("Press ESC to return to menu", True, BLACK)
        self.screen.blit(text, (20, HEIGHT - 40))
        
        pygame.display.flip()

    def handle_keyboard_input(self, event):
        """Handle keyboard mode input events."""
        if event.type == KEYDOWN:
            # Handle ratio selection
            if event.key in RATIO_KEYS:
                self.selected_ratio = RATIO_KEYS[event.key]
            # Handle octave selection
            elif event.key in range(K_1, K_8):
                self.selected_octave = event.key - K_1 + 1
            # Handle note playing
            elif event.key in NOTE_RATIOS:
                freq = self.sound_manager.get_frequency(
                    self.selected_ratio,
                    NOTE_RATIOS[event.key],
                    self.selected_octave
                )
                self.sound_manager.play_note(freq, event.key)
        elif event.type == KEYUP:
            if event.key in NOTE_RATIOS:
                self.sound_manager.stop_note(event.key)

    def run(self):
        """Main application loop."""
        while self.running:
            # Mode selection menu
            if self.mode is None:
                self.draw_menu()
                for event in pygame.event.get():
                    if event.type == QUIT:
                        self.running = False
                        
                    # Handle mouse events for buttons
                    for i, button in enumerate(self.menu_buttons):
                        if button.handle_event(event):
                            if i == 0:
                                self.mode = 'keyboard'
                                self.sound_manager = SoundManager(use_just_intonation=True)
                            elif i == 1:
                                self.mode = 'midi_select'
                                self.sound_manager = SoundManager(use_just_intonation=True)
                            elif i == 2:
                                self.mode = 'midi_select_uncorrected'
                                self.sound_manager = SoundManager(use_just_intonation=False)
                                
                    # Handle keyboard shortcuts
                    if event.type == KEYDOWN:
                        if event.key == K_1:
                            self.mode = 'keyboard'
                            self.sound_manager = SoundManager(use_just_intonation=True)
                        elif event.key == K_2:
                            self.mode = 'midi_select'
                            self.sound_manager = SoundManager(use_just_intonation=True)
                        elif event.key == K_3:
                            self.mode = 'midi_select_uncorrected'
                            self.sound_manager = SoundManager(use_just_intonation=False)
                continue
                
            # MIDI selection screen
            if self.mode in ['midi_select', 'midi_select_uncorrected']:
                self.draw_midi_selector()
                for event in pygame.event.get():
                    if event.type == QUIT:
                        self.running = False
                    elif event.type == KEYDOWN and event.key == K_ESCAPE:
                        self.mode = None
                    elif self.midi_selector.handle_event(event):
                        song_path = self.midi_selector.get_selected_song()
                        if song_path:
                            self.mode = ('midi_corrected' if self.mode == 'midi_select' 
                                       else 'midi_uncorrected')
                            self.sound_manager.play_midi_file(song_path)
                continue
                
            # Main loop for keyboard or MIDI mode
            self.screen.fill(WHITE)
            
            # Process events
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.mode = None
                        if self.sound_manager:
                            self.sound_manager.stop_midi()
                            for key in list(self.sound_manager.active_sounds.keys()):
                                self.sound_manager.stop_note(key)
                        continue
                    
                    # Handle mode-specific input
                    if self.mode == 'keyboard':
                        self.handle_keyboard_input(event)
                    elif self.mode in ['midi_corrected', 'midi_uncorrected']:
                        if event.key == K_SPACE:
                            self.sound_manager.stop_midi()
                            self.mode = 'midi_select' if self.mode == 'midi_corrected' else 'midi_select_uncorrected'
                        elif event.key == K_r:
                            self.sound_manager.stop_midi()
                elif event.type == KEYUP and self.mode == 'keyboard':
                    self.handle_keyboard_input(event)
                    
            # Draw status for keyboard mode
            if self.mode == 'keyboard':
                status_text = [
                    f"Base ratio: {self.selected_ratio}",
                    f"Octave: {self.selected_octave}",
                    f"Active notes: {len(self.sound_manager.active_sounds)}"
                ]
                for i, text in enumerate(status_text):
                    surface = self.small_font.render(text, True, BLACK)
                    self.screen.blit(surface, (20, 140 + i * 40))
                    
            # Process any pending MIDI events
            if self.mode in ['midi_corrected', 'midi_uncorrected'] and self.sound_manager:
                self.sound_manager.process_midi_events()
                
            # Update visualization
            if self.sound_manager:
                self.visualizer.draw(
                    self.screen,
                    self.sound_manager.active_frequencies,
                    self.selected_ratio,
                    self.mode
                )
            
            # Draw help text
            self.visualizer.draw_help(self.screen, self.mode)
            
            pygame.display.flip()
            pygame.time.wait(10)
            
        # Cleanup
        if self.sound_manager:
            self.sound_manager.cleanup()
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    app = MusicApp()
    app.run()
