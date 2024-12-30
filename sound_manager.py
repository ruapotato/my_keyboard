import pygame
import numpy as np
import scipy.signal
from typing import Dict, List, Tuple
import mido
import threading
import queue
import time
from constants import SAMPLE_RATE, BASE_FREQ, RATIOS
from threading import Lock

class ModulationSystem:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.lfo_shapes = {
            'sine': lambda x: np.sin(2 * np.pi * x),
            'triangle': lambda x: 2 * np.abs(2 * (x - np.floor(x + 0.5))) - 1,
            'square': lambda x: np.sign(np.sin(2 * np.pi * x)),
            'random': lambda x: np.random.random(len(x)) * 2 - 1
        }
    
    def generate_lfo(self, freq: float, shape: str, duration: float, phase: float = 0) -> np.ndarray:
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)
        lfo = self.lfo_shapes[shape](freq * t + phase)
        return self.smooth_transitions(lfo)
    
    def smooth_transitions(self, signal: np.ndarray, window_size: int = 64) -> np.ndarray:
        window = np.hanning(window_size)
        smoothed = np.convolve(signal, window/window.sum(), mode='same')
        return smoothed

class HarmonicGenerator:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.ratio_harmonics = {
            'A': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # Perfect harmonics for tonic
            'B': [1.0, 1.125, 2.25, 3.375, 4.5],   # Based on 9:8 ratio
            'C': [1.0, 1.25, 2.5, 3.75, 5.0],      # Based on 5:4 ratio
            'D': [1.0, 1.333, 2.667, 4.0],         # Based on 4:3 ratio
            'E': [1.0, 1.5, 3.0, 4.5],             # Based on 3:2 ratio
            'F': [1.0, 1.667, 3.333, 5.0],         # Based on 5:3 ratio
            'G': [1.0, 1.875, 3.75, 5.625],        # Based on 15:8 ratio
            'H': [1.0, 1.067, 2.133, 3.2],         # Based on 16:15 ratio
            'I': [1.0, 1.2, 2.4, 3.6, 4.8],        # Based on 6:5 ratio
            'J': [1.0, 1.4, 2.8, 4.2],             # Based on 7:5 ratio
            'K': [1.0, 1.6, 3.2, 4.8],             # Based on 8:5 ratio
            'L': [1.0, 1.8, 3.6, 5.4],             # Based on 9:5 ratio
        }

    def generate_harmonic_series(self, frequency: float, num_harmonics: int, note_name: str = 'A') -> List[Tuple[float, float]]:
        series = []
        ratio_multipliers = self.ratio_harmonics.get(note_name, self.ratio_harmonics['A'])
        series.append((1.0, frequency))
        
        for i, multiplier in enumerate(ratio_multipliers[1:], start=2):
            if i > num_harmonics:
                break
            base_amplitude = 1.0 / (i ** 1.5)
            ratio_alignment = 1.0 + (0.2 if multiplier in ratio_multipliers else 0.0)
            amplitude = base_amplitude * ratio_alignment
            
            detune_amount = 0.001 * (1 + (i-1) * 0.1)
            detune_freq = frequency * multiplier
            
            series.append((amplitude * 0.5, detune_freq * (1 + detune_amount)))
            series.append((amplitude * 0.5, detune_freq * (1 - detune_amount)))
            
        return series

class SoundManager:
    def __init__(self, use_just_intonation=True):
        self.sample_rate = SAMPLE_RATE
        pygame.mixer.init(self.sample_rate, -16, 2, 512)
        self.modulation = ModulationSystem(self.sample_rate)
        self.harmonics = HarmonicGenerator(self.sample_rate)
        
        self.active_sounds = {}
        self.active_frequencies = {}
        self.use_just_intonation = use_just_intonation
        self.note_history = []
        self.last_frequency = None
        self.midi_playing = False
        self.midi_thread = None
        self.event_queue = queue.Queue()
        
        self.mod_matrix = {
            'vibrato': {'freq': 5.0, 'depth': 0.003, 'shape': 'sine'},
            'tremolo': {'freq': 4.0, 'depth': 0.15, 'shape': 'triangle'},
            'harmonic': {'freq': 0.5, 'depth': 0.2, 'shape': 'sine'}
        }

    def generate_complex_wave(self, frequency: float, duration: float = 0.5, velocity: float = 1.0) -> np.ndarray:
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)
        
        # Generate harmonic content
        harmonics = self.harmonics.generate_harmonic_series(frequency, 12)
        wave = np.zeros_like(t)
        
        # Generate modulation signals
        vibrato = self.modulation.generate_lfo(
            self.mod_matrix['vibrato']['freq'],
            self.mod_matrix['vibrato']['shape'],
            duration
        )
        tremolo = self.modulation.generate_lfo(
            self.mod_matrix['tremolo']['freq'],
            self.mod_matrix['tremolo']['shape'],
            duration
        )
        
        # Apply modulation
        for amp, freq in harmonics:
            mod_freq = freq * (1 + vibrato * self.mod_matrix['vibrato']['depth'])
            phase = np.cumsum(mod_freq) * 2 * np.pi / self.sample_rate
            harmonic = amp * np.sin(phase)
            wave += harmonic
            
        wave *= 1 + tremolo * self.mod_matrix['tremolo']['depth']
        return wave

    def get_frequency(self, ratio: Tuple[int, int], note_ratio: float, octave: int) -> float:
        if self.use_just_intonation:
            numerator, denominator = ratio
            freq = BASE_FREQ * (numerator/denominator) * note_ratio
        else:
            freq = BASE_FREQ * note_ratio
        octave_adjust = octave - 4
        return freq * (2 ** octave_adjust)

    def play_note(self, freq: float, key: int, velocity: float = 1.0):
        if key in self.active_sounds:
            self.active_sounds[key].stop()
            
        wave = self.generate_complex_wave(freq, velocity=velocity)
        
        # Add simple envelope
        fade_samples = int(0.02 * self.sample_rate)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        # Apply envelope before converting to int16
        wave[:fade_samples] *= fade_in
        wave[-fade_samples:] *= fade_out
        
        # Convert to stereo and int16 after envelope
        wave = np.int16(wave * 32767)
        stereo_wave = np.column_stack((wave, wave))
        
        sound = pygame.sndarray.make_sound(stereo_wave)
        sound.play(-1)
        self.active_sounds[key] = sound
        self.active_frequencies[key] = freq

    def stop_note(self, key: int):
        if key in self.active_sounds:
            self.active_sounds[key].fadeout(100)
            if key in self.active_frequencies:
                del self.active_frequencies[key]

    def stop_midi(self):
        """Stop MIDI playback."""
        self.midi_playing = False
        if self.midi_thread:
            self.midi_thread.join()
        for key in list(self.active_sounds.keys()):
            self.stop_note(key)

    def play_midi_file(self, midi_path: str):
        """Start MIDI file playback in a separate thread."""
        def midi_player():
            mid = mido.MidiFile(midi_path)
            tempo = 500000  # Default tempo (microseconds per beat)
            
            for msg in mid.play():
                if not self.midi_playing:
                    break
                    
                if msg.type == 'note_on' and msg.velocity > 0:
                    freq = 440 * (2 ** ((msg.note - 69) / 12))
                    velocity = msg.velocity / 127.0
                    self.event_queue.put(('note_on', freq, msg.note, velocity))
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    self.event_queue.put(('note_off', msg.note))
                    
                time.sleep(msg.time)
            self.midi_playing = False

        self.midi_playing = True
        self.midi_thread = threading.Thread(target=midi_player)
        self.midi_thread.start()

    def process_midi_events(self):
        """Process pending MIDI events."""
        while not self.event_queue.empty():
            event = self.event_queue.get()
            if event[0] == 'note_on':
                _, freq, note, velocity = event
                self.play_note(freq, note, velocity)
            elif event[0] == 'note_off':
                _, note = event
                self.stop_note(note)
