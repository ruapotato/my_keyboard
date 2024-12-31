# My Keyboard

My Keyboard is an innovative application for exploring and playing music using Just Intonation tuning and an ergonomic keyboard layout. Developed by David Hamner and released under the GPL3 license, it provides a unique way to interact with musical notes and harmonies.

## Features

- **Just Intonation Tuning**: My Keyboard uses Just Intonation tuning, which is based on small whole number ratios between frequencies. This results in more harmonically pure intervals compared to the standard Equal Temperament tuning.

- **Ergonomic Keyboard Layout**: The application features an ergonomic keyboard layout optimized for playing and exploring musical notes. The keys are arranged in a way that facilitates easy access to related notes and chords.

- **Interactive Visualization**: My Keyboard includes an interactive visualization that displays the waveforms of the notes being played. It provides a visual representation of the harmonic relationships between the notes.

- **MIDI Support**: In addition to the keyboard mode, My Keyboard also supports playing MIDI files. It can load and play MIDI files using either the Just Intonation tuning or the original tuning of the MIDI file.


## Keyboard Layout and Note Frequencies

The following table shows the ratios associated with each letter:

| Letter | Ratio  | Interval        |
|--------|--------|-----------------|
| A      | 1/1    | Unison          |
| B      | 9/8    | Major Second    |
| C      | 5/4    | Major Third     |
| D      | 4/3    | Perfect Fourth  |
| E      | 3/2    | Perfect Fifth   |
| F      | 5/3    | Major Sixth     |
| G      | 15/8   | Major Seventh   |
| H      | 16/15  | Minor Second    |
| I      | 6/5    | Minor Third     |
| J      | 7/5    | Tritone         |
| K      | 8/5    | Minor Sixth     |
| L      | 9/5    | Minor Seventh   |

## Usage

1. Launch the My Keyboard application.
2. In the main menu, select the desired mode:
   - Keyboard Mode: Play notes using the computer keyboard.
   - MIDI (Just Intonation): Load and play a MIDI file using Just Intonation tuning.
   - MIDI (Original Tuning): Load and play a MIDI file using its original tuning.
3. If in Keyboard Mode:
   - Use the keys QWERT ASDFG ZXCVB to play notes.
   - Use the keys HJKL;' to select different frequency ratios.
   - Use the number keys 1-7 to select octaves.
4. If in MIDI Mode:
   - Select a MIDI file from the file browser.
   - The MIDI file will start playing automatically.
   - Press Space to stop playback and select a new file.
   - Press R to reset playback.
5. Observe the interactive visualization to see the waveforms of the notes being played.
6. Press Esc to return to the main menu.

## Requirements

- Python 3.x
- Pygame
- NumPy
- SciPy
- Mido

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/davidhamner/my-keyboard.git
   ```
2. Navigate to the project directory:
   ```
   cd my-keyboard
   ```
   ```
3. Run the application:
   ```
   python main.py
   ```

## License

My Keyboard is released under the GPL3 License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- The Just Intonation tuning ratios are based on the work of Harry Partch and other microtonal music theorists.
- The sound generation and processing techniques are inspired by various research papers and articles on digital sound synthesis and audio effects.
- Claude for answering my stupid music theory questions.
- This is just me playing around with sound. Don't expect any updates on this repository. :)

## Contact

David Hamner: ke7oxh@gmail.com
