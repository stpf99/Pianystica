# ================================
# pianistica.py — Unified Music Engine with MixedModes
# ================================
import sys
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QPushButton, QComboBox, QGroupBox, QHBoxLayout, QSpinBox
)
from PyQt6.QtCore import QTimer

import rtmidi

# ================================
# GLOBAL MUSIC ENGINE
# ================================
@dataclass
class MusicNote:
    pitch: int
    velocity: int
    time: float
    duration: float
    origin: str = "generated"
    coherence_score: float = 0.0

@dataclass
class MusicAnalysis:
    scale_coherence: float = 0.0
    tempo_stability: float = 0.0
    pitch_variety: float = 0.0
    velocity_consistency: float = 0.0
    suggested_key: int = 0

@dataclass
class ModeConfig:
    name: str
    scale_type: str  # "major", "minor", "pentatonic", "whole_tone", "chromatic"
    notes_per_second: float
    pitch_deviation: int
    velocity_range: Tuple[int, int]
    duration_range: Tuple[float, float]
    structure: str  # "structured" (motif-based), "chaos" (random-to-ordered), "ambient" (sparse)
    rhythm_quantum: float  # Time grid in seconds
    max_notes_per_second: int
    harmonic_complexity: str  # "simple" (diatonic), "complex" (jazz), "minimal" (ambient)
    weight: float  # For MixedModes selection

@dataclass
class GenreConfig:
    name: str
    scale_modifier: str  # Optional scale override
    pitch_dev_modifier: float  # Multiplier for pitch deviation
    velocity_modifier: Tuple[int, int]  # Adjustment to velocity range
    duration_modifier: Tuple[float, float]  # Adjustment to duration range
    rhythm_offset: float  # Offset for syncopation (0 for none)

class GlobalMusicEngine:
    def __init__(self):
        self.current_key = random.randint(0, 11)
        self.history: List[List[MusicNote]] = []
        self.generation_count = 0
        self.last_notes: List[MusicNote] = []
        self.mode_configs = {
            "Song Mode": ModeConfig(
                name="Song Mode",
                scale_type="major",
                notes_per_second=5.0,
                pitch_deviation=4,
                velocity_range=(60, 90),
                duration_range=(0.2, 0.4),
                structure="structured",
                rhythm_quantum=0.2,
                max_notes_per_second=6,
                harmonic_complexity="simple",
                weight=0.20
            ),
            "Jazz Theory Mode": ModeConfig(
                name="Jazz Theory Mode",
                scale_type="chromatic",
                notes_per_second=6.0,
                pitch_deviation=6,
                velocity_range=(50, 100),
                duration_range=(0.15, 0.3),
                structure="structured",
                rhythm_quantum=0.1667,
                max_notes_per_second=7,
                harmonic_complexity="complex",
                weight=0.15
            ),
            "Relaxation Mode": ModeConfig(
                name="Relaxation Mode",
                scale_type="pentatonic",
                notes_per_second=3.0,
                pitch_deviation=3,
                velocity_range=(40, 70),
                duration_range=(0.5, 1.5),
                structure="ambient",
                rhythm_quantum=0.3333,
                max_notes_per_second=4,
                harmonic_complexity="minimal",
                weight=0.25
            ),
            "Like Author Mode": ModeConfig(
                name="Like Author Mode",
                scale_type="major",
                notes_per_second=5.0,
                pitch_deviation=4,
                velocity_range=(60, 90),
                duration_range=(0.2, 0.4),
                structure="structured",
                rhythm_quantum=0.2,
                max_notes_per_second=6,
                harmonic_complexity="simple",
                weight=0.25
            ),
            "Simplify from Chaos": ModeConfig(
                name="Simplify from Chaos",
                scale_type="major",
                notes_per_second=5.0,
                pitch_deviation=5,
                velocity_range=(60, 90),
                duration_range=(0.2, 0.2),
                structure="chaos",
                rhythm_quantum=0.2,
                max_notes_per_second=5,
                harmonic_complexity="simple",
                weight=0.15
            ),
            "MixedModes": ModeConfig(
                name="MixedModes",
                scale_type="major",
                notes_per_second=5.0,
                pitch_deviation=4,
                velocity_range=(60, 90),
                duration_range=(0.2, 0.4),
                structure="mixed",
                rhythm_quantum=0.2,
                max_notes_per_second=6,
                harmonic_complexity="simple",
                weight=1.0
            )
        }
        self.genre_configs = {
            "Classical": GenreConfig(
                name="Classical",
                scale_modifier="major",
                pitch_dev_modifier=0.7,
                velocity_modifier=(-10, -5),
                duration_modifier=(1.0, 1.2),
                rhythm_offset=0.0
            ),
            "Jazz": GenreConfig(
                name="Jazz",
                scale_modifier="chromatic",
                pitch_dev_modifier=1.2,
                velocity_modifier=(-5, 5),
                duration_modifier=(0.8, 1.0),
                rhythm_offset=0.05
            ),
            "Ambient": GenreConfig(
                name="Ambient",
                scale_modifier="pentatonic",
                pitch_dev_modifier=0.5,
                velocity_modifier=(-15, -10),
                duration_modifier=(1.5, 2.0),
                rhythm_offset=0.0
            ),
            "Pop": GenreConfig(
                name="Pop",
                scale_modifier="major",
                pitch_dev_modifier=0.8,
                velocity_modifier=(0, 0),
                duration_modifier=(1.0, 1.0),
                rhythm_offset=0.0
            )
        }
        self.current_mode = "Song Mode"
        self.current_genre = "Classical"
        self.motif: List[MusicNote] = []
        self.section_index = 0
        self.repeat_counter = 0

    def set_mode(self, mode: str, genre: str = "Classical"):
        self.current_mode = mode if mode in self.mode_configs else "Song Mode"
        self.current_genre = genre if genre in self.genre_configs else "Classical"
        self.generation_count = 0
        self.last_notes = []
        self.motif = []
        self.section_index = 0
        self.repeat_counter = 0
        if self.current_mode != "Simplify from Chaos":
            self.current_key = random.randint(0, 11)

    def _get_scale(self, scale_type: str) -> List[int]:
        if scale_type == "major":
            return [(self.current_key + i) % 12 for i in [0, 2, 4, 5, 7, 9, 11]]
        elif scale_type == "minor":
            return [(self.current_key + i) % 12 for i in [0, 2, 3, 5, 7, 8, 10]]
        elif scale_type == "pentatonic":
            return [(self.current_key + i) % 12 for i in [0, 2, 4, 7, 9]]
        elif scale_type == "whole_tone":
            return [(self.current_key + i * 2) % 12 for i in range(6)]
        else:  # chromatic
            return list(range(12))

    def _nearest_scale_pitch(self, pitch: int, scale: List[int]) -> int:
        pitch_class = pitch % 12
        closest_pc = min(scale, key=lambda x: min(abs(x - pitch_class), abs(x - pitch_class + 12)))
        octave = pitch // 12
        return max(21, min(108, octave * 12 + closest_pc))

    def _apply_genre_modifiers(self, config: ModeConfig, genre: str) -> ModeConfig:
        genre_config = self.genre_configs[genre]
        modified = ModeConfig(
            name=config.name,
            scale_type=genre_config.scale_modifier or config.scale_type,
            notes_per_second=config.notes_per_second,
            pitch_deviation=int(config.pitch_deviation * genre_config.pitch_dev_modifier),
            velocity_range=(
                max(1, config.velocity_range[0] + genre_config.velocity_modifier[0]),
                min(127, config.velocity_range[1] + genre_config.velocity_modifier[1])
            ),
            duration_range=(
                config.duration_range[0] * genre_config.duration_modifier[0],
                config.duration_range[1] * genre_config.duration_modifier[1]
            ),
            structure=config.structure,
            rhythm_quantum=config.rhythm_quantum,
            max_notes_per_second=config.max_notes_per_second,
            harmonic_complexity=config.harmonic_complexity,
            weight=config.weight
        )
        return modified

    def _generate_motif(self, config: ModeConfig, duration: float) -> List[MusicNote]:
        notes = []
        t = 0.0
        last_pitch = self.last_notes[-1].pitch if self.last_notes else random.randint(60, 72)
        scale = self._get_scale(config.scale_type)
        genre_config = self.genre_configs[self.current_genre]
        while t < duration:
            slot = int(t / config.rhythm_quantum)
            if slot >= int(duration / config.rhythm_quantum):
                break
            pitch = last_pitch + random.randint(-config.pitch_deviation, config.pitch_deviation)
            pitch = self._nearest_scale_pitch(pitch, scale)
            last_pitch = pitch
            velocity = random.randint(*config.velocity_range)
            note_duration = random.uniform(*config.duration_range)
            note_time = round(t / config.rhythm_quantum) * config.rhythm_quantum + genre_config.rhythm_offset
            notes.append(MusicNote(
                pitch=pitch,
                velocity=velocity,
                time=note_time,
                duration=note_duration,
                origin="motif"
            ))
            t += config.rhythm_quantum
        return sorted(notes, key=lambda x: x.time)

    def _generate_chaos(self, config: ModeConfig, duration: float) -> List[MusicNote]:
        notes = []
        notes_per_slot = [0] * int(duration / config.rhythm_quantum + 1)
        scale = self._get_scale(config.scale_type)
        last_pitch = self.last_notes[-1].pitch if self.last_notes else random.randint(60, 72)
        genre_config = self.genre_configs[self.current_genre]
        t = 0.0
        while t < duration:
            slot = int(t / config.rhythm_quantum)
            if slot >= len(notes_per_slot) or notes_per_slot[slot] >= config.max_notes_per_second:
                t += config.rhythm_quantum
                continue
            if random.random() < config.notes_per_second * config.rhythm_quantum:
                pitch = last_pitch + random.randint(-config.pitch_deviation, config.pitch_deviation)
                pitch = self._nearest_scale_pitch(pitch, scale)
                last_pitch = pitch
                velocity = random.randint(*config.velocity_range)
                note_duration = config.duration_range[0]
                note_time = round(t / config.rhythm_quantum) * config.rhythm_quantum + genre_config.rhythm_offset
                notes.append(MusicNote(
                    pitch=pitch,
                    velocity=velocity,
                    time=note_time,
                    duration=note_duration,
                    origin="chaos"
                ))
                notes_per_slot[slot] += 1
            t += config.rhythm_quantum
        return sorted(notes, key=lambda x: x.time)

    def _simplify_notes(self, notes: List[MusicNote], config: ModeConfig) -> List[MusicNote]:
        scale = self._get_scale(config.scale_type)
        simplified = []
        notes_per_slot = [0] * int(max(n.time for n in notes) / config.rhythm_quantum + 1) if notes else []
        last_pitch = self.last_notes[-1].pitch if self.last_notes else random.randint(60, 72)
        genre_config = self.genre_configs[self.current_genre]
        for note in notes:
            pitch = self._nearest_scale_pitch(note.pitch, scale)
            velocity = max(config.velocity_range[0], min(config.velocity_range[1], 70 + int((note.velocity - 70) * 0.6)))
            note_time = round(note.time / config.rhythm_quantum) * config.rhythm_quantum + genre_config.rhythm_offset
            if abs(pitch - last_pitch) > config.pitch_deviation:
                pitch = self._nearest_scale_pitch(last_pitch + random.choice([-1, 1]) * config.pitch_deviation, scale)
            last_pitch = pitch
            simplified_note = MusicNote(
                pitch=pitch,
                velocity=velocity,
                time=note_time,
                duration=config.duration_range[0],
                origin="simplified",
                coherence_score=0.8
            )
            slot = int(note_time / config.rhythm_quantum)
            if slot < len(notes_per_slot) and notes_per_slot[slot] < config.max_notes_per_second:
                simplified.append(simplified_note)
                notes_per_slot[slot] += 1
        return sorted(simplified, key=lambda x: x.time)

    def _apply_variations(self, notes: List[MusicNote], config: ModeConfig, section: str) -> List[MusicNote]:
        out = [MusicNote(n.pitch, n.velocity, n.time, n.duration, n.origin) for n in notes]
        if config.structure == "structured" and random.random() < 0.3:
            semitones = random.randint(-2, 2)
            scale = self._get_scale(config.scale_type)
            out = [MusicNote(self._nearest_scale_pitch(n.pitch + semitones, scale), n.velocity, n.time, n.duration, n.origin) for n in out]
        if section in ("chorus", "outro"):
            out = [MusicNote(n.pitch, int(n.velocity * 1.1), n.time, n.duration / 2, n.origin) for n in out]
        return sorted(out, key=lambda x: x.time)

    def _analyze_notes(self, notes: List[MusicNote]) -> MusicAnalysis:
        analysis = MusicAnalysis()
        if not notes:
            return analysis
        scale = self._get_scale(self.mode_configs[self.current_mode].scale_type)
        pitch_classes = [n.pitch % 12 for n in notes]
        in_scale_count = sum(1 for pc in pitch_classes if pc in scale)
        analysis.scale_coherence = in_scale_count / len(pitch_classes) if pitch_classes else 0
        time_intervals = [notes[i+1].time - notes[i].time for i in range(len(notes)-1) if notes[i+1].time > notes[i].time]
        if time_intervals:
            avg_interval = sum(time_intervals) / len(time_intervals)
            variance = sum((t - avg_interval) ** 2 for t in time_intervals) / len(time_intervals)
            analysis.tempo_stability = 1.0 / (1.0 + variance)
        analysis.pitch_variety = len(set(n.pitch for n in notes)) / len(notes) if notes else 0
        velocities = [n.velocity for n in notes]
        if velocities:
            vel_range = max(velocities) - min(velocities)
            analysis.velocity_consistency = 1.0 - (vel_range / (self.mode_configs[self.current_mode].velocity_range[1] - self.mode_configs[self.current_mode].velocity_range[0]))
        analysis.suggested_key = self.current_key
        return analysis

    def _select_mixed_mode(self) -> ModeConfig:
        modes = [m for m in self.mode_configs.values() if m.name != "MixedModes"]
        weights = [m.weight for m in modes]
        return random.choices(modes, weights=weights, k=1)[0]

    def next_iteration(self, duration: float = 8.0) -> Dict:
        self.generation_count += 1
        config = self.mode_configs[self.current_mode]
        genre = random.choice(list(self.genre_configs.keys()))
        section = ["intro", "verse", "chorus", "verse", "chorus", "outro"][min(self.section_index, 5)]
        
        if self.current_mode == "MixedModes":
            base_config = self._select_mixed_mode()
            config = self._apply_genre_modifiers(base_config, genre)
        else:
            config = self._apply_genre_modifiers(config, self.current_genre)

        if config.structure == "structured":
            if not self.motif or self.generation_count % 6 == 0:
                self.motif = self._generate_motif(config, duration)
            notes = self._apply_variations(self.motif, config, section)
        elif config.structure == "chaos":
            if self.generation_count == 1:
                notes = self._generate_chaos(config, duration)
            else:
                notes = self._generate_chaos(config, duration)
                notes = self._simplify_notes(notes, config)
        else:  # ambient
            notes = self._generate_chaos(config, duration)
            notes = self._simplify_notes(notes, config)

        analysis = self._analyze_notes(notes)
        self.last_notes = notes
        self.history.append(notes.copy())
        
        midi_notes = [(n.pitch, n.velocity, n.time) for n in notes]
        info = f"{config.name} ({genre}) | Gen{self.generation_count} | Key:{self.current_key} | Notes:{len(notes)}"
        
        if config.structure == "structured":
            self.repeat_counter += 1
            max_rep = {"intro": 1, "verse": 2, "chorus": 2, "outro": 1}.get(section, 1)
            if self.repeat_counter >= max_rep:
                self.section_index += 1
                self.repeat_counter = 0
                if self.section_index >= 6:
                    self.section_index = 0
                    self.motif = []
                    if random.random() < 0.2:
                        self.current_key = random.randint(0, 11)
        
        return {
            "type": config.name,
            "name": info,
            "notes": midi_notes,
            "analysis": analysis
        }

# ================================
# MUSIC PLAYER
# ================================
class MusicPlayer:
    def __init__(self, parent):
        self.parent = parent
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next)
        self.is_playing = False
        self.engine = GlobalMusicEngine()
        
    def start(self, mode: str):
        self.stop()
        genre = self.parent.genre_combo.currentText()
        self.engine.set_mode(mode, genre)
        self.is_playing = True
        config = self.engine.mode_configs[mode]
        # Update config with GUI values
        config.notes_per_second = self.parent.notes_per_sec.value()
        config.rhythm_quantum = 1.0 / config.notes_per_second
        config.max_notes_per_second = max(1, int(config.notes_per_second))
        config.pitch_deviation = self.parent.pitch_dev.value()
        config.duration_range = (config.rhythm_quantum, config.rhythm_quantum)
        if mode == "MixedModes":
            config.weight = 1.0
            for m in self.engine.mode_configs.values():
                if m.name != "MixedModes":
                    m.weight = self.parent.mode_weights[m.name].value() / 100.0
        self.play_next()
        self.timer.start(8000)
    
    def stop(self):
        self.is_playing = False
        try:
            self.timer.stop()
        except Exception:
            pass
    
    def play_next(self):
        if not self.is_playing:
            return
        iteration_data = self.engine.next_iteration(duration=8.0)
        self.parent.question_label.setText(iteration_data['name'])
        config = self.engine.mode_configs[self.engine.current_mode]
        for (note, vel, t) in iteration_data['notes']:
            delay_ms = int(t * 1000)
            QTimer.singleShot(delay_ms, lambda n=note, v=vel: self._note_on(n, v))
            QTimer.singleShot(delay_ms + int(config.duration_range[0] * 1000), lambda n=note: self._note_off(n))
    
    def _note_on(self, n, v):
        if self.parent.rtmidi_output:
            self.parent.rtmidi_output.send_message([0x90, n, v])
    
    def _note_off(self, n):
        if self.parent.rtmidi_output:
            self.parent.rtmidi_output.send_message([0x80, n, 0])

# ================================
# APLIKACJA GUI
# ================================
class PianoLearningApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Piano Learning with Unified Music Engine & MixedModes")
        self.resize(900, 600)
        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        self.question_label = QLabel("Ready")
        layout.addWidget(self.question_label)

        # --- MIDI OUT ---
        self.rtmidi_output = None
        self.midi = rtmidi.MidiOut()
        ports = self.midi.get_ports()
        self.midi_combo = QComboBox()
        self.midi_combo.addItems(ports if ports else ["No MIDI ports found"])
        self.midi_connect_btn = QPushButton("Connect")
        self.midi_connect_btn.clicked.connect(self.connect_midi)
        layout.addWidget(QLabel("MIDI Out:"))
        layout.addWidget(self.midi_combo)
        layout.addWidget(self.midi_connect_btn)

        # --- Mode combo ---
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Song Mode", "Jazz Theory Mode", "Relaxation Mode", "Like Author Mode", "Simplify from Chaos", "MixedModes"])
        self.mode_combo.currentTextChanged.connect(self.switch_mode)
        layout.addWidget(QLabel("Mode:"))
        layout.addWidget(self.mode_combo)

        # --- Genre combo ---
        self.genre_combo = QComboBox()
        self.genre_combo.addItems(["Classical", "Jazz", "Ambient", "Pop"])
        layout.addWidget(QLabel("Genre:"))
        layout.addWidget(self.genre_combo)

        # --- Controls ---
        self.controls = QGroupBox("Mode Settings")
        ctrl_layout = QHBoxLayout(self.controls)
        self.bpm = QSpinBox(); self.bpm.setRange(20, 300); self.bpm.setValue(120); self.bpm.setSuffix(" BPM")
        self.notes_per_sec = QSpinBox(); self.notes_per_sec.setRange(1, 10); self.notes_per_sec.setValue(5)
        self.pitch_dev = QSpinBox(); self.pitch_dev.setRange(1, 12); self.pitch_dev.setValue(4)
        for w in [
            ("BPM", self.bpm),
            ("Notes/s", self.notes_per_sec),
            ("Pitch Dev", self.pitch_dev)
        ]:
            ctrl_layout.addWidget(QLabel(w[0]))
            ctrl_layout.addWidget(w[1])
        layout.addWidget(self.controls)

        # --- Mode weights for MixedModes ---
        self.weights_group = QGroupBox("MixedModes Weights (%)")
        weights_layout = QHBoxLayout(self.weights_group)
        self.mode_weights = {
            "Song Mode": QSpinBox(),
            "Jazz Theory Mode": QSpinBox(),
            "Relaxation Mode": QSpinBox(),
            "Like Author Mode": QSpinBox(),
            "Simplify from Chaos": QSpinBox()
        }
        for mode, spinbox in self.mode_weights.items():
            spinbox.setRange(0, 100)
            spinbox.setValue(int(self._get_default_weight(mode) * 100))
            weights_layout.addWidget(QLabel(mode))
            weights_layout.addWidget(spinbox)
        layout.addWidget(self.weights_group)
        self.weights_group.setVisible(False)

        self.player = None

    def _get_default_weight(self, mode: str) -> float:
        weights = {
            "Song Mode": 0.20,
            "Jazz Theory Mode": 0.15,
            "Relaxation Mode": 0.25,
            "Like Author Mode": 0.25,
            "Simplify from Chaos": 0.15
        }
        return weights.get(mode, 0.20)

    def connect_midi(self):
        idx = self.midi_combo.currentIndex()
        try:
            self.rtmidi_output = rtmidi.MidiOut()
            self.rtmidi_output.open_port(idx)
            self.statusBar().showMessage(f"Connected to MIDI Out: {self.midi_combo.currentText()}")
        except Exception as e:
            self.statusBar().showMessage(f"MIDI connection failed: {e}")
            self.rtmidi_output = None

    def switch_mode(self, mode: str):
        if self.player:
            self.player.stop()
        self.player = MusicPlayer(self)
        self.weights_group.setVisible(mode == "MixedModes")
        self.player.start(mode)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PianoLearningApp()
    win.show()
    sys.exit(app.exec())