# ================================
# pianistica.py — Unified Music Engine with MixedModes, Virtual Arranger, and Music Thinker
# ================================
import sys
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable

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
    pleasantness: float = 0.0

@dataclass
class ModeConfig:
    name: str
    scale_type: str
    mean_notes_per_second: float  # Changed to mean
    mean_pitch_deviation: int     # Changed to mean
    mean_velocity_range: Tuple[int, int]  # Changed to mean
    mean_duration_range: Tuple[float, float]  # Changed to mean
    structure: str
    mean_rhythm_quantum: float    # Changed to mean
    mean_max_notes_per_second: int  # Changed to mean
    harmonic_complexity: str
    weight: float

@dataclass
class GenreConfig:
    name: str
    scale_modifier: str
    pitch_dev_modifier: float
    velocity_modifier: Tuple[int, int]
    duration_modifier: Tuple[float, float]
    rhythm_offset: float

@dataclass
class VariationScheme:
    name: str
    probability: float
    params: Dict

class VariationArranger:
    def __init__(self):
        self.variation_schemes = {
            "repetition": VariationScheme(
                name="repetition",
                probability=0.4,
                params={}
            ),
            "transposition": VariationScheme(
                name="transposition",
                probability=0.3,
                params={"semitones": [-2, -1, 1, 2]}
            ),
            "inversion": VariationScheme(
                name="inversion",
                probability=0.15,
                params={"pivot_pitch": 60}
            ),
            "retrograde": VariationScheme(
                name="retrograde",
                probability=0.1,
                params={}
            ),
            "rhythmic_shift": VariationScheme(
                name="rhythmic_shift",
                probability=0.05,
                params={"shift_amount": [0.1, -0.1, 0.2, -0.2]}
            )
        }

    def apply_variation(self, notes: List[MusicNote], config: ModeConfig, section: str, current_key: int, nearest_scale_pitch: Callable[[int, List[int]], int], scheme_name: Optional[str] = None) -> List[MusicNote]:
        if not notes:
            return notes
        scale = [(current_key + i) % 12 for i in self._get_scale(config.scale_type)]
        if scheme_name and scheme_name in self.variation_schemes:
            selected_scheme = self.variation_schemes[scheme_name]
        else:
            selected_scheme = random.choices(
                list(self.variation_schemes.values()),
                weights=[s.probability for s in self.variation_schemes.values()],
                k=1
            )[0]
        
        out_notes = [MusicNote(n.pitch, n.velocity, n.time, n.duration, n.origin) for n in notes]
        max_time = max(n.time for n in notes) if notes else 0

        if selected_scheme.name == "repetition":
            return out_notes
        elif selected_scheme.name == "transposition":
            semitones = random.choice(selected_scheme.params["semitones"])
            out_notes = [
                MusicNote(
                    nearest_scale_pitch(n.pitch + semitones, scale),
                    n.velocity,
                    n.time,
                    n.duration,
                    f"{n.origin}_transposed"
                ) for n in out_notes
            ]
        elif selected_scheme.name == "inversion":
            pivot = selected_scheme.params["pivot_pitch"]
            out_notes = [
                MusicNote(
                    nearest_scale_pitch(2 * pivot - n.pitch, scale),
                    n.velocity,
                    n.time,
                    n.duration,
                    f"{n.origin}_inverted"
                ) for n in out_notes
            ]
        elif selected_scheme.name == "retrograde":
            out_notes = [
                MusicNote(
                    n.pitch,
                    n.velocity,
                    max_time - n.time,
                    n.duration,
                    f"{n.origin}_retrograde"
                ) for n in out_notes[::-1]
            ]
        elif selected_scheme.name == "rhythmic_shift":
            shift = random.choice(selected_scheme.params["shift_amount"])
            out_notes = [
                MusicNote(
                    n.pitch,
                    n.velocity,
                    max(0, n.time + shift),
                    n.duration,
                    f"{n.origin}_shifted"
                ) for n in out_notes
            ]
        
        if section in ("chorus", "outro"):
            out_notes = [
                MusicNote(n.pitch, int(n.velocity * 1.1), n.time, n.duration * 0.9, n.origin)
                for n in out_notes
            ]
        
        return sorted(out_notes, key=lambda x: x.time)

    def _get_scale(self, scale_type: str) -> List[int]:
        if scale_type == "major":
            return [0, 2, 4, 5, 7, 9, 11]
        elif scale_type == "minor":
            return [0, 2, 3, 5, 7, 8, 10]
        elif scale_type == "pentatonic":
            return [0, 2, 4, 7, 9]
        elif scale_type == "whole_tone":
            return [0, 2, 4, 6, 8, 10]
        else:  # chromatic
            return list(range(12))

class MusicThinker:
    def __init__(self, engine: 'GlobalMusicEngine'):
        self.engine = engine
        self.lookahead_duration = 20.0
        self.num_proposals = 3

    def evaluate_proposal(self, notes: List[MusicNote], config: ModeConfig) -> float:
        analysis = self.engine._analyze_notes(notes)
        note_density = len(notes) / self.lookahead_duration
        max_density = config.mean_max_notes_per_second * self.lookahead_duration
        density_score = 1.0 - abs(note_density - max_density / 2) / (max_density / 2)
        pitch_jumps = [abs(notes[i+1].pitch - notes[i].pitch) for i in range(len(notes)-1)] if len(notes) > 1 else [0]
        avg_jump = sum(pitch_jumps) / len(pitch_jumps) if pitch_jumps else 0
        jump_score = 1.0 - min(avg_jump / config.mean_pitch_deviation, 1.0)
        analysis.pleasantness = 0.4 * analysis.scale_coherence + 0.3 * analysis.pitch_variety + 0.2 * density_score + 0.1 * jump_score
        return analysis.pleasantness

    def generate_proposals(self, config: ModeConfig, section: str, duration: float) -> List[List[MusicNote]]:
        proposals = []
        original_motif = self.engine.motif[:]
        variation_schemes = list(self.engine.arranger.variation_schemes.keys())

        for i in range(self.num_proposals):
            # Apply random variation around mean values
            modified_config = ModeConfig(
                name=config.name,
                scale_type=config.scale_type,
                mean_notes_per_second=config.mean_notes_per_second * random.uniform(0.8, 1.2),
                mean_pitch_deviation=int(config.mean_pitch_deviation * random.uniform(0.8, 1.2)),
                mean_velocity_range=(
                    max(1, config.mean_velocity_range[0] + random.randint(-5, 5)),
                    min(127, config.mean_velocity_range[1] + random.randint(-5, 5))
                ),
                mean_duration_range=(
                    config.mean_duration_range[0] * random.uniform(0.9, 1.1),
                    config.mean_duration_range[1] * random.uniform(0.9, 1.1)
                ),
                structure=config.structure,
                mean_rhythm_quantum=config.mean_rhythm_quantum * random.uniform(0.8, 1.2),
                mean_max_notes_per_second=max(1, int(config.mean_max_notes_per_second * random.uniform(0.8, 1.2))),
                harmonic_complexity=config.harmonic_complexity,
                weight=config.weight
            )
            
            if config.structure == "structured":
                self.engine.motif = self.engine._generate_motif(modified_config, duration)
                scheme = variation_schemes[i % len(variation_schemes)]
                notes = self.engine.arranger.apply_variation(
                    self.engine.motif, modified_config, section, self.engine.current_key, self.engine._nearest_scale_pitch, scheme
                )
            elif config.structure == "chaos":
                notes = self.engine._generate_chaos(modified_config, duration)
                if self.engine.generation_count > 1:
                    notes = self.engine._simplify_notes(notes, modified_config)
            else:  # ambient
                notes = self.engine._generate_chaos(modified_config, duration)
                notes = self.engine._simplify_notes(notes, modified_config)
            
            proposals.append(notes)
            self.engine.motif = original_motif[:]  # Restore original motif

        return proposals

    def select_best_proposal(self, config: ModeConfig, section: str, duration: float) -> List[MusicNote]:
        proposals = self.generate_proposals(config, section, duration)
        if not proposals:
            return []
        scores = [self.evaluate_proposal(proposal, config) for proposal in proposals]
        best_index = scores.index(max(scores))
        return proposals[best_index]

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
                mean_notes_per_second=5.0,
                mean_pitch_deviation=4,
                mean_velocity_range=(60, 90),
                mean_duration_range=(0.2, 0.4),
                structure="structured",
                mean_rhythm_quantum=0.2,
                mean_max_notes_per_second=6,
                harmonic_complexity="simple",
                weight=0.20
            ),
            "Jazz Theory Mode": ModeConfig(
                name="Jazz Theory Mode",
                scale_type="chromatic",
                mean_notes_per_second=6.0,
                mean_pitch_deviation=6,
                mean_velocity_range=(50, 100),
                mean_duration_range=(0.15, 0.3),
                structure="structured",
                mean_rhythm_quantum=0.1667,
                mean_max_notes_per_second=7,
                harmonic_complexity="complex",
                weight=0.15
            ),
            "Relaxation Mode": ModeConfig(
                name="Relaxation Mode",
                scale_type="pentatonic",
                mean_notes_per_second=3.0,
                mean_pitch_deviation=3,
                mean_velocity_range=(40, 70),
                mean_duration_range=(0.5, 1.5),
                structure="ambient",
                mean_rhythm_quantum=0.3333,
                mean_max_notes_per_second=4,
                harmonic_complexity="minimal",
                weight=0.25
            ),
            "Like Author Mode": ModeConfig(
                name="Like Author Mode",
                scale_type="major",
                mean_notes_per_second=5.0,
                mean_pitch_deviation=4,
                mean_velocity_range=(60, 90),
                mean_duration_range=(0.2, 0.4),
                structure="structured",
                mean_rhythm_quantum=0.2,
                mean_max_notes_per_second=6,
                harmonic_complexity="simple",
                weight=0.25
            ),
            "Simplify from Chaos": ModeConfig(
                name="Simplify from Chaos",
                scale_type="major",
                mean_notes_per_second=5.0,
                mean_pitch_deviation=5,
                mean_velocity_range=(60, 90),
                mean_duration_range=(0.2, 0.2),
                structure="chaos",
                mean_rhythm_quantum=0.2,
                mean_max_notes_per_second=5,
                harmonic_complexity="simple",
                weight=0.15
            ),
            "MixedModes": ModeConfig(
                name="MixedModes",
                scale_type="major",
                mean_notes_per_second=5.0,
                mean_pitch_deviation=4,
                mean_velocity_range=(60, 90),
                mean_duration_range=(0.2, 0.4),
                structure="mixed",
                mean_rhythm_quantum=0.2,
                mean_max_notes_per_second=6,
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
        self.arranger = VariationArranger()
        self.thinker = MusicThinker(self)

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
            mean_notes_per_second=config.mean_notes_per_second,
            mean_pitch_deviation=int(config.mean_pitch_deviation * genre_config.pitch_dev_modifier),
            mean_velocity_range=(
                max(1, config.mean_velocity_range[0] + genre_config.velocity_modifier[0]),
                min(127, config.mean_velocity_range[1] + genre_config.velocity_modifier[1])
            ),
            mean_duration_range=(
                config.mean_duration_range[0] * genre_config.duration_modifier[0],
                config.mean_duration_range[1] * genre_config.duration_modifier[1]
            ),
            structure=config.structure,
            mean_rhythm_quantum=config.mean_rhythm_quantum,
            mean_max_notes_per_second=config.mean_max_notes_per_second,
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
            slot = int(t / config.mean_rhythm_quantum)
            if slot >= int(duration / config.mean_rhythm_quantum):
                break
            pitch = last_pitch + random.randint(-config.mean_pitch_deviation, config.mean_pitch_deviation)
            pitch = self._nearest_scale_pitch(pitch, scale)
            last_pitch = pitch
            velocity = random.randint(*config.mean_velocity_range)
            note_duration = random.uniform(*config.mean_duration_range)
            note_time = round(t / config.mean_rhythm_quantum) * config.mean_rhythm_quantum + genre_config.rhythm_offset
            notes.append(MusicNote(
                pitch=pitch,
                velocity=velocity,
                time=note_time,
                duration=note_duration,
                origin="motif"
            ))
            t += config.mean_rhythm_quantum
        return sorted(notes, key=lambda x: x.time)

    def _generate_chaos(self, config: ModeConfig, duration: float) -> List[MusicNote]:
        notes = []
        notes_per_slot = [0] * int(duration / config.mean_rhythm_quantum + 1)
        scale = self._get_scale(config.scale_type)
        last_pitch = self.last_notes[-1].pitch if self.last_notes else random.randint(60, 72)
        genre_config = self.genre_configs[self.current_genre]
        t = 0.0
        while t < duration:
            slot = int(t / config.mean_rhythm_quantum)
            if slot >= len(notes_per_slot) or notes_per_slot[slot] >= config.mean_max_notes_per_second:
                t += config.mean_rhythm_quantum
                continue
            if random.random() < config.mean_notes_per_second * config.mean_rhythm_quantum:
                pitch = last_pitch + random.randint(-config.mean_pitch_deviation, config.mean_pitch_deviation)
                pitch = self._nearest_scale_pitch(pitch, scale)
                last_pitch = pitch
                velocity = random.randint(*config.mean_velocity_range)
                note_duration = config.mean_duration_range[0]
                note_time = round(t / config.mean_rhythm_quantum) * config.mean_rhythm_quantum + genre_config.rhythm_offset
                notes.append(MusicNote(
                    pitch=pitch,
                    velocity=velocity,
                    time=note_time,
                    duration=note_duration,
                    origin="chaos"
                ))
                notes_per_slot[slot] += 1
            t += config.mean_rhythm_quantum
        return sorted(notes, key=lambda x: x.time)

    def _simplify_notes(self, notes: List[MusicNote], config: ModeConfig) -> List[MusicNote]:
        scale = self._get_scale(config.scale_type)
        simplified = []
        notes_per_slot = [0] * int(max(n.time for n in notes) / config.mean_rhythm_quantum + 1) if notes else []
        last_pitch = self.last_notes[-1].pitch if self.last_notes else random.randint(60, 72)
        genre_config = self.genre_configs[self.current_genre]
        for note in notes:
            pitch = self._nearest_scale_pitch(note.pitch, scale)
            velocity = max(config.mean_velocity_range[0], min(config.mean_velocity_range[1], 70 + int((note.velocity - 70) * 0.6)))
            note_time = round(note.time / config.mean_rhythm_quantum) * config.mean_rhythm_quantum + genre_config.rhythm_offset
            if abs(pitch - last_pitch) > config.mean_pitch_deviation:
                pitch = self._nearest_scale_pitch(last_pitch + random.choice([-1, 1]) * config.mean_pitch_deviation, scale)
            last_pitch = pitch
            simplified_note = MusicNote(
                pitch=pitch,
                velocity=velocity,
                time=note_time,
                duration=config.mean_duration_range[0],
                origin="simplified",
                coherence_score=0.8
            )
            slot = int(note_time / config.mean_rhythm_quantum)
            if slot < len(notes_per_slot) and notes_per_slot[slot] < config.mean_max_notes_per_second:
                simplified.append(simplified_note)
                notes_per_slot[slot] += 1
        return sorted(simplified, key=lambda x: x.time)

    def _apply_variations(self, notes: List[MusicNote], config: ModeConfig, section: str) -> List[MusicNote]:
        return self.arranger.apply_variation(notes, config, section, self.current_key, self._nearest_scale_pitch)

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
            analysis.velocity_consistency = 1.0 - (vel_range / (self.mode_configs[self.current_mode].mean_velocity_range[1] - self.mode_configs[self.current_mode].mean_velocity_range[0]))
        analysis.suggested_key = self.current_key
        return analysis

    def _select_mixed_mode(self) -> ModeConfig:
        modes = [m for m in self.mode_configs.values() if m.name != "MixedModes"]
        weights = [m.weight for m in modes]
        return random.choices(modes, weights=weights, k=1)[0]

    def _apply_config_variation(self, config: ModeConfig) -> ModeConfig:
        return ModeConfig(
            name=config.name,
            scale_type=config.scale_type,
            mean_notes_per_second=config.mean_notes_per_second * random.uniform(0.8, 1.2),
            mean_pitch_deviation=int(config.mean_pitch_deviation * random.uniform(0.8, 1.2)),
            mean_velocity_range=(
                max(1, config.mean_velocity_range[0] + random.randint(-5, 5)),
                min(127, config.mean_velocity_range[1] + random.randint(-5, 5))
            ),
            mean_duration_range=(
                config.mean_duration_range[0] * random.uniform(0.9, 1.1),
                config.mean_duration_range[1] * random.uniform(0.9, 1.1)
            ),
            structure=config.structure,
            mean_rhythm_quantum=config.mean_rhythm_quantum * random.uniform(0.8, 1.2),
            mean_max_notes_per_second=max(1, int(config.mean_max_notes_per_second * random.uniform(0.8, 1.2))),
            harmonic_complexity=config.harmonic_complexity,
            weight=config.weight
        )

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
        
        # Apply random variation to config for this iteration
        config = self._apply_config_variation(config)

        if config.structure == "structured":
            if not self.motif or self.generation_count % 6 == 0:
                self.motif = self._generate_motif(config, duration)
            notes = self.thinker.select_best_proposal(config, section, self.thinker.lookahead_duration)
            notes = [n for n in notes if n.time < duration]
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
        config.mean_notes_per_second = self.parent.notes_per_sec.value()
        config.mean_rhythm_quantum = 1.0 / config.mean_notes_per_second
        config.mean_max_notes_per_second = max(1, int(config.mean_notes_per_second))
        config.mean_pitch_deviation = self.parent.pitch_dev.value()
        config.mean_duration_range = (config.mean_rhythm_quantum, config.mean_rhythm_quantum)
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
            QTimer.singleShot(delay_ms + int(config.mean_duration_range[0] * 1000), lambda n=note: self._note_off(n))
    
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

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Song Mode", "Jazz Theory Mode", "Relaxation Mode", "Like Author Mode", "Simplify from Chaos", "MixedModes"])
        self.mode_combo.currentTextChanged.connect(self.switch_mode)
        layout.addWidget(QLabel("Mode:"))
        layout.addWidget(self.mode_combo)

        self.genre_combo = QComboBox()
        self.genre_combo.addItems(["Classical", "Jazz", "Ambient", "Pop"])
        layout.addWidget(QLabel("Genre:"))
        layout.addWidget(self.genre_combo)

        self.controls = QGroupBox("Mode Settings")
        ctrl_layout = QHBoxLayout(self.controls)
        self.bpm = QSpinBox(); self.bpm.setRange(20, 300); self.bpm.setValue(50); self.bpm.setSuffix(" BPM")
        self.notes_per_sec = QSpinBox(); self.notes_per_sec.setRange(1, 10); self.notes_per_sec.setValue(3)
        self.pitch_dev = QSpinBox(); self.pitch_dev.setRange(1, 12); self.pitch_dev.setValue(2)
        for w in [
            ("BPM", self.bpm),
            ("Mean Notes/s", self.notes_per_sec),
            ("Mean Pitch Dev", self.pitch_dev)
        ]:
            ctrl_layout.addWidget(QLabel(w[0]))
            ctrl_layout.addWidget(w[1])
        layout.addWidget(self.controls)

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
