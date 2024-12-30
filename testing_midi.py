import pygame
import numpy as np
import scipy.signal
import scipy.interpolate
from typing import Dict, List, Tuple
import time
import queue
import argparse
import mido
from threading import Lock

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

def midi_to_freq(note: int) -> float:
    return 440 * (2 ** ((note - 69) / 12))

class AudioBuffer:
    def __init__(self, sample_rate: int, buffer_size: int = 4096):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.buffer = np.zeros(buffer_size * 2)  # Double buffer for smooth transitions
        self.write_pos = 0
        self.lock = Lock()
        
    def write(self, data: np.ndarray) -> None:
        with self.lock:
            # Crossfade at buffer boundaries to prevent clicks
            fade_len = min(1024, len(data))
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            
            # Apply crossfade at buffer wrap points
            if self.write_pos + len(data) > len(self.buffer):
                first_part_len = len(self.buffer) - self.write_pos
                second_part_len = len(data) - first_part_len
                
                # Crossfade at wrap point
                self.buffer[self.write_pos:] *= fade_out[-first_part_len:]
                self.buffer[:second_part_len] *= fade_out[:second_part_len]
                
                self.buffer[self.write_pos:] += data[:first_part_len] * fade_in[:first_part_len]
                self.buffer[:second_part_len] += data[first_part_len:] * fade_in[first_part_len:]
                
                self.write_pos = second_part_len
            else:
                self.buffer[self.write_pos:self.write_pos + len(data)] += data
                self.write_pos = (self.write_pos + len(data)) % len(self.buffer)

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
        """Generate LFO with given shape and frequency"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)
        lfo = self.lfo_shapes[shape](freq * t + phase)
        return self.smooth_transitions(lfo)
    
    def smooth_transitions(self, signal: np.ndarray, window_size: int = 64) -> np.ndarray:
        """Smooth sudden transitions in modulation signals"""
        window = np.hanning(window_size)
        smoothed = np.convolve(signal, window/window.sum(), mode='same')
        return smoothed

class HarmonicGenerator:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.harmonic_cache = {}
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
        """Generate a rich harmonic series with harmonics based on the note's ratio"""
        series = []
        
        # Get the ratio-specific harmonics
        ratio_multipliers = self.ratio_harmonics.get(note_name, self.ratio_harmonics['A'])
        
        # Add fundamental
        series.append((1.0, frequency))
        
        # Generate harmonics based on the note's ratio
        for i, multiplier in enumerate(ratio_multipliers[1:], start=2):
            if i > num_harmonics:
                break
                
            # Calculate amplitude with ratio-aware rolloff
            base_amplitude = 1.0 / (i ** 1.5)
            
            # Enhance harmonics that align with the note's ratio
            ratio_alignment = 1.0 + (0.2 if multiplier in ratio_multipliers else 0.0)
            amplitude = base_amplitude * ratio_alignment
            
            # Add slight detuning for chorus effect
            detune_amount = 0.001 * (1 + (i-1) * 0.1)
            detune_freq = frequency * multiplier
            
            series.append((amplitude * 0.5, detune_freq * (1 + detune_amount)))
            series.append((amplitude * 0.5, detune_freq * (1 - detune_amount)))
            
        return series

class SoundManager:
    def __init__(self, use_just_intonation: bool):
        self.sample_rate = 44100
        pygame.mixer.init(self.sample_rate, -16, 2, 512)  # Smaller buffer size for lower latency
        self.audio_buffer = AudioBuffer(self.sample_rate)
        self.modulation = ModulationSystem(self.sample_rate)
        self.harmonics = HarmonicGenerator(self.sample_rate)
        
        self.active_sounds = {}
        self.use_just_intonation = use_just_intonation
        self.note_history = []
        self.last_frequency = None
        
        # Initialize modulation parameters
        self.mod_matrix = {
            'vibrato': {'freq': 5.0, 'depth': 0.003, 'shape': 'sine'},
            'tremolo': {'freq': 4.0, 'depth': 0.15, 'shape': 'triangle'},
            'harmonic': {'freq': 0.5, 'depth': 0.2, 'shape': 'sine'}
        }
        
    def find_closest_ratio(self, freq: float) -> tuple[str, int]:
        """Find the closest just intonation ratio for a given frequency"""
        ratio = freq / BASE_FREQ
        octave = int(np.log2(ratio))
        
        ratio_in_octave = ratio / (2 ** octave)
        if ratio_in_octave < 1:
            ratio_in_octave *= 2
            octave -= 1
            
        min_diff = float('inf')
        closest_ratio = None
        
        for name, (num, den) in RATIOS.items():
            r = num / den
            diff = abs(ratio_in_octave - r)
            if diff < min_diff:
                min_diff = diff
                closest_ratio = name
        
        return closest_ratio, octave + 4
        
    def get_frequency(self, note: int) -> tuple[float, str, int]:
        """Convert MIDI note to frequency with optional just intonation"""
        freq = midi_to_freq(note)
        if self.use_just_intonation:
            note_name, octave = self.find_closest_ratio(freq)
            num, den = RATIOS[note_name]
            just_freq = BASE_FREQ * (num/den) * (2 ** (octave - 4))
            return just_freq, note_name, octave
        return freq, 'X', 4

    def generate_complex_wave(self, frequency: float, duration: float, velocity: float) -> np.ndarray:
        """Generate a complex waveform with rich harmonics and modulation"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)
        
        # Get just intonation ratios if enabled
        if self.use_just_intonation:
            note_name, octave = self.find_closest_ratio(frequency)
            num, den = RATIOS[note_name]
            base_freq = BASE_FREQ * (num/den) * (2 ** (octave - 4))
        else:
            base_freq = frequency
            
        # Generate harmonic content
        harmonics = self.harmonics.generate_harmonic_series(base_freq, 12)
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
        harmonic_mod = self.modulation.generate_lfo(
            self.mod_matrix['harmonic']['freq'],
            self.mod_matrix['harmonic']['shape'],
            duration
        )
        
        # Apply frequency modulation to each harmonic
        for amp, freq in harmonics:
            # Apply vibrato
            mod_freq = freq * (1 + vibrato * self.mod_matrix['vibrato']['depth'])
            # Phase accumulation for accurate frequency modulation
            phase = np.cumsum(mod_freq) * 2 * np.pi / self.sample_rate
            
            # Generate and add harmonic with modulation
            harmonic = amp * np.sin(phase)
            harmonic *= 1 + harmonic_mod * self.mod_matrix['harmonic']['depth']
            wave += harmonic
            
        # Apply amplitude modulation (tremolo)
        wave *= 1 + tremolo * self.mod_matrix['tremolo']['depth']
        
        return wave

    def generate_adaptive_envelope(self, duration: float, samples: int, velocity: float) -> np.ndarray:
        """Generate a sophisticated ADSR envelope with advanced modulation"""
        velocity_factor = np.power(velocity / 127, 1.5)
        
        # Dynamic envelope timing based on velocity and frequency
        attack = int(0.008 * samples * (1.0 - 0.3 * velocity_factor))
        decay = int(0.15 * samples * (1.0 + 0.2 * velocity_factor))
        release = int(0.3 * samples * (1.0 - 0.2 * velocity_factor))
        sustain = samples - attack - decay - release
        
        # Generate envelope segments with advanced curves
        attack_env = self._generate_curve(attack, 'exponential', velocity_factor)
        decay_env = self._generate_curve(decay, 'logarithmic', velocity_factor)
        sustain_level = 0.8 * velocity_factor
        sustain_env = np.ones(sustain) * sustain_level
        release_env = self._generate_curve(release, 'logarithmic', velocity_factor) * sustain_level
        
        # Combine segments
        envelope = np.concatenate([attack_env, decay_env, sustain_env, release_env])
        
        # Add complex modulation
        t = np.linspace(0, duration, len(envelope))
        
        # Primary modulation
        mod1 = 0.02 * np.sin(2 * np.pi * 5.0 * t)
        # Secondary slower modulation
        mod2 = 0.01 * np.sin(2 * np.pi * 2.5 * t)
        # Random micro-variations
        noise = 0.005 * (np.random.random(len(t)) - 0.5)
        
        # Combine modulations with envelope
        envelope *= (1.0 + mod1 + mod2 + noise)
        
        # Advanced smoothing with multiple passes
        envelope = self._multi_stage_smoothing(envelope)
        
        return envelope
        
    def _generate_curve(self, length: int, curve_type: str, factor: float) -> np.ndarray:
        """Generate various envelope curves with controllable shape"""
        x = np.linspace(0, 1, length)
        
        if curve_type == 'exponential':
            curve = 1 - np.exp(-5 * x * factor)
        elif curve_type == 'logarithmic':
            curve = np.log(1 + 9 * x) / np.log(10)
        elif curve_type == 'sigmoid':
            curve = 1 / (1 + np.exp(-10 * (x - 0.5)))
        else:
            curve = x
            
        return curve / np.max(curve)
        
    def _multi_stage_smoothing(self, signal: np.ndarray) -> np.ndarray:
        """Apply multi-stage smoothing to prevent artifacts"""
        # First stage: Remove high-frequency content
        window_size = 32
        signal = scipy.signal.savgol_filter(signal, window_size, 3)
        
        # Second stage: Smooth transitions
        window = np.hanning(64)
        signal = scipy.signal.convolve(signal, window/window.sum(), mode='same')
        
        # Third stage: Ensure no sudden changes
        gradient = np.gradient(signal)
        max_gradient = np.max(np.abs(gradient))
        if max_gradient > 0.1:
            gradient = np.clip(gradient, -0.1, 0.1)
            signal = np.cumsum(gradient) / len(signal)
            
        # Final stage: Normalize
        return signal / np.max(np.abs(signal))

    def apply_stereo_enhancement(self, wave: np.ndarray, frequency: float) -> np.ndarray:
        """Enhanced stereo processing with frequency-dependent width and modulation"""
        # Calculate frequency-dependent parameters
        stereo_width = np.clip(1.0 - (frequency - 440) / 2000, 0.4, 0.9)
        mod_freq = 0.5 + frequency / 2000  # Higher frequencies get faster modulation
        
        # Generate stereo modulation
        t = np.linspace(0, len(wave)/self.sample_rate, len(wave))
        mod_left = self.modulation.generate_lfo(mod_freq, 'sine', len(wave)/self.sample_rate)
        mod_right = self.modulation.generate_lfo(mod_freq, 'sine', len(wave)/self.sample_rate, np.pi)
        
        # Apply Haas effect for frequencies below 1000 Hz
        if frequency < 1000:
            delay_samples = int(0.01 * self.sample_rate * (1000 - frequency) / 1000)
            right_channel = np.roll(wave, delay_samples)
        else:
            right_channel = wave.copy()
            
        # Create stereo field
        left_channel = wave * (1 + mod_left * 0.02)
        right_channel = right_channel * (1 + mod_right * 0.02)
        
        # Advanced stereo processing
        mid = (left_channel + right_channel) * 0.5
        side = (left_channel - right_channel) * 0.5
        
        # Frequency-dependent stereo enhancement
        enhanced_side = scipy.signal.filtfilt(
            *scipy.signal.butter(2, frequency/2, 'lowpass', fs=self.sample_rate),
            side
        )
        
        # Recombine with enhanced width
        left = mid + enhanced_side * stereo_width
        right = mid - enhanced_side * stereo_width
        
        # Normalize
        max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
        if max_val > 0:
            left = left / max_val
            right = right / max_val
            
        return np.column_stack((left, right))

    def play_note(self, note: int, velocity: int, duration: float):
        """Enhanced note playback with anti-click protection"""
        freq, note_name, octave = self.get_frequency(note)
        
        # Generate the basic waveform with all modulations
        wave = self.generate_complex_wave(freq, duration, velocity)
        
        # Apply envelope
        envelope = self.generate_adaptive_envelope(duration, len(wave), velocity)
        wave *= envelope
        
        # Apply stereo enhancement
        stereo_wave = self.apply_stereo_enhancement(wave, freq)
        
        # Anti-click protection: ensure zero-crossings at boundaries
        fade_samples = 64
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        stereo_wave[:fade_samples] *= fade_in[:, np.newaxis]
        stereo_wave[-fade_samples:] *= fade_out[:, np.newaxis]
        
        # Find zero crossings near boundaries and adjust
        for i in range(min(100, len(stereo_wave))):
            if abs(stereo_wave[i][0]) < 0.001:
                stereo_wave[:i] = 0
                break
                
        for i in range(1, min(100, len(stereo_wave))):
            if abs(stereo_wave[-i][0]) < 0.001:
                stereo_wave[-i:] = 0
                break
        
        # Ensure smooth transition from previous note
        if note in self.active_sounds:
            # Crossfade with previous note
            self.active_sounds[note].fadeout(30)
            
        # Create and play the sound
        sound = pygame.sndarray.make_sound(np.int16(stereo_wave * 32767))
        
        # Apply velocity-sensitive volume
        volume = self._calculate_velocity_curve(velocity)
        sound.set_volume(volume)
        
        # Start playback
        sound.play()
        self.active_sounds[note] = sound
        
        # Update note history
        self.note_history.append((freq, time.time()))
        self.last_frequency = freq
        
        # Clean up old history
        self._clean_note_history()
        
    def _calculate_velocity_curve(self, velocity: int) -> float:
        """Calculate volume using an optimized velocity curve"""
        # Non-linear velocity response curve
        normalized = velocity / 127
        return 0.2 + 0.8 * (np.power(normalized, 1.5))
        
    def _clean_note_history(self):
        """Clean up note history and manage resources"""
        current_time = time.time()
        self.note_history = [(f, t) for f, t in self.note_history if current_time - t < 2.0]
        
    def process_events(self):
        """Process cleanup events for note management"""
        for event in pygame.event.get(pygame.USEREVENT):
            if hasattr(event, 'dict') and 'note' in event.dict:
                note = event.dict['note']
                if note in self.active_sounds:
                    del self.active_sounds[note]

    def stop_note(self, note: int):
        """Stop note with fadeout"""
        if note in self.active_sounds:
            self.active_sounds[note].fadeout(100)
            # Create a custom event for cleanup
            cleanup_event = pygame.event.Event(pygame.USEREVENT, {'note': note})
            pygame.event.post(cleanup_event)
            pygame.time.set_timer(pygame.USEREVENT, 150)  # Cleanup after fadeout

def play_midi_file(midi_path: str, use_just_intonation: bool):
    """Play a MIDI file with enhanced sound generation"""
    pygame.init()
    sound_manager = SoundManager(use_just_intonation)
    mid = mido.MidiFile(midi_path)
    expected_length = mid.length
    print(f"Expected length: {expected_length:.1f}s")
    
    tempo = 500000  # Default tempo (microseconds per beat)
    
    for msg in mid.play():
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        elif msg.type == 'note_on' and msg.velocity > 0:
            # Calculate note duration based on tempo
            ticks_per_beat = mid.ticks_per_beat
            seconds_per_tick = tempo / (ticks_per_beat * 1000000)
            duration = max(0.2, seconds_per_tick * 480)  # Minimum duration of 200ms
            
            sound_manager.play_note(msg.note, msg.velocity, duration)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            sound_manager.stop_note(msg.note)
            
        # Process any cleanup events
        sound_manager.process_events()
        time.sleep(msg.time)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('midi_file', help='Path to MIDI file')
    parser.add_argument('--original', action='store_true', help='Use original frequencies instead of just intonation')
    args = parser.parse_args()
    
    try:
        play_midi_file(args.midi_file, not args.original)
    except KeyboardInterrupt:
        pygame.quit()

if __name__ == '__main__':
    main()
