#!/usr/bin/env python3
"""
iconic_composer.py — Iconic Composer Fantasy Engine
Generative MIDI komposer w stylu mistrzów klasycznych.
Algorytm: Typ instrumentu + DNA emocjonalne kompozytora × wybór emoji-nastrojów
→ Wagi fantazji → Parametry aranżacji → Track MIDI
"""

import sys, random, math
from dataclasses import dataclass, field, fields
from typing import List, Dict, Tuple, Optional

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import numpy as np

try:
    import rtmidi
    RTMIDI_OK = True
except ImportError:
    RTMIDI_OK = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QGroupBox, QSpinBox,
    QScrollArea, QFrame, QGridLayout, QFileDialog,
    QTextEdit, QMessageBox, QCheckBox, QProgressBar, QSizePolicy,
    QSlider
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor

# ─────────────────────────────────────────────────────────────
# SCALE DEFINITIONS
# ─────────────────────────────────────────────────────────────

SCALE_INTERVALS: Dict[str, List[int]] = {
    "major":          [0, 2, 4, 5, 7, 9, 11],
    "minor":          [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "pentatonic":     [0, 2, 4, 7, 9],
    "whole_tone":     [0, 2, 4, 6, 8, 10],
    "chromatic":      list(range(12)),
    "dorian":         [0, 2, 3, 5, 7, 9, 10],
}

CHORD_INTERVALS: Dict[str, List[int]] = {
    "major":     [0, 4, 7],
    "minor":     [0, 3, 7],
    "dim":       [0, 3, 6],
    "major7":    [0, 4, 7, 11],
    "minor7":    [0, 3, 7, 10],
    "dominant7": [0, 4, 7, 10],
    "sus4":      [0, 5, 7],
    "add9":      [0, 4, 7, 14],
}

def build_scale_pitches(root: int, scale_type: str) -> List[int]:
    ivs = SCALE_INTERVALS.get(scale_type, SCALE_INTERVALS["major"])
    out = []
    for octave in range(0, 9):
        for iv in ivs:
            p = root + iv + octave * 12
            if 21 <= p <= 108:
                out.append(p)
    return out

def nearest_pitch(pitch: int, scale_pitches: List[int]) -> int:
    if not scale_pitches:
        return max(21, min(108, pitch))
    return min(scale_pitches, key=lambda p: abs(p - pitch))

def chord_pitches(root_pc: int, ctype: str, scale_pitches: List[int],
                  octave: int = 3) -> List[int]:
    base = root_pc + octave * 12
    ivs = CHORD_INTERVALS.get(ctype, [0, 4, 7])
    return [max(21, min(108, nearest_pitch(base + iv, scale_pitches))) for iv in ivs]

# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class EmotionState:
    emoji: str
    name: str
    valence: float       # –1…+1  (negatywna/pozytywna)
    arousal: float       # 0…1    (spokój/pobudzenie)
    tension: float       # 0…1    (rozwiązanie/napięcie)
    complexity: float    # 0…1
    tempo_factor: float
    velocity_factor: float
    scale_pref: str
    articulation: str    # legato / staccato / rubato / mixed

@dataclass
class NoteEvent:
    pitch:    int
    velocity: int
    time:     float    # sekundy od początku sekcji
    duration: float
    channel:  int  = 0
    voice:    str  = "melody"

@dataclass
class ComposerProfile:
    name: str
    emoji: str
    instrument: str
    midi_program: int          # GM number
    base_tempo: int
    emotional_dna: Dict[str, float]   # nazwy emocji → wagi
    scale_prefs: List[str]
    texture: str               # nocturne/sonata/classical/counterpoint/impressionist/virtuosic
    pitch_range: Tuple[int, int]
    vel_range: Tuple[int, int]
    rhythm_feel: str           # rubato/strict/dance/free
    ornament_density: float    # 0…1
    harm_complexity: float     # 0…1
    char_intervals: List[int]  # charakterystyczne interwały melodyczne

@dataclass
class GenParams:
    root_key:        int
    scale_type:      str
    scale_pitches:   List[int]
    tempo_bpm:       float
    vel_range:       Tuple[int, int]
    beat_dur:        float
    texture:         str
    articulation:    str
    rubato:          float
    ornaments:       float
    pitch_center:    int
    pitch_spread:    int
    phrase_len:      int        # takty (w ćwierćnutach)
    harm_complexity: float
    section_dur:     float

def update_params(p: GenParams, **kw) -> GenParams:
    d = {f.name: getattr(p, f.name) for f in fields(p)}
    d.update(kw)
    return GenParams(**d)

# ─────────────────────────────────────────────────────────────
# BAZY DANYCH: KOMPOZYTORZY & EMOCJE
# ─────────────────────────────────────────────────────────────

COMPOSERS: Dict[str, ComposerProfile] = {
    "Chopin": ComposerProfile(
        name="Chopin", emoji="🌙",
        instrument="Piano", midi_program=0,
        base_tempo=66,
        emotional_dna={"melancholy": 0.40, "passion": 0.30, "tenderness": 0.20, "drama": 0.10},
        scale_prefs=["minor", "harmonic_minor", "major"],
        texture="nocturne",
        pitch_range=(48, 96), vel_range=(28, 110),
        rhythm_feel="rubato", ornament_density=0.72, harm_complexity=0.76,
        char_intervals=[1, 2, 3, 4, 7, 12],
    ),
    "Beethoven": ComposerProfile(
        name="Beethoven", emoji="⚡",
        instrument="Piano", midi_program=0,
        base_tempo=112,
        emotional_dna={"heroic": 0.35, "struggle": 0.30, "triumph": 0.20, "lyrical": 0.15},
        scale_prefs=["minor", "major"],
        texture="sonata",
        pitch_range=(36, 96), vel_range=(20, 127),
        rhythm_feel="strict", ornament_density=0.22, harm_complexity=0.66,
        char_intervals=[0, 0, 0, 7],    # short-short-short-long
    ),
    "Mozart": ComposerProfile(
        name="Mozart", emoji="🌟",
        instrument="Piano", midi_program=0,
        base_tempo=132,
        emotional_dna={"elegance": 0.40, "playfulness": 0.30, "brightness": 0.20, "clarity": 0.10},
        scale_prefs=["major", "minor"],
        texture="classical",
        pitch_range=(48, 84), vel_range=(40, 100),
        rhythm_feel="dance", ornament_density=0.55, harm_complexity=0.44,
        char_intervals=[2, 4, 5, 7],
    ),
    "Bach": ComposerProfile(
        name="Bach", emoji="🎼",
        instrument="Harpsichord", midi_program=6,
        base_tempo=80,
        emotional_dna={"mathematical": 0.35, "spiritual": 0.30, "baroque": 0.25, "polyphony": 0.10},
        scale_prefs=["major", "minor", "dorian"],
        texture="counterpoint",
        pitch_range=(36, 84), vel_range=(50, 90),
        rhythm_feel="strict", ornament_density=0.80, harm_complexity=0.87,
        char_intervals=[2, 3, 4, 5, 7],
    ),
    "Debussy": ComposerProfile(
        name="Debussy", emoji="🌊",
        instrument="Piano", midi_program=0,
        base_tempo=69,
        emotional_dna={"impressionist": 0.40, "dreamy": 0.35, "coloristic": 0.15, "fluid": 0.10},
        scale_prefs=["whole_tone", "pentatonic", "chromatic"],
        texture="impressionist",
        pitch_range=(48, 96), vel_range=(15, 88),
        rhythm_feel="free", ornament_density=0.33, harm_complexity=0.91,
        char_intervals=[2, 6, 7, 9, 11],
    ),
    "Liszt": ComposerProfile(
        name="Liszt", emoji="🔥",
        instrument="Piano", midi_program=0,
        base_tempo=152,
        emotional_dna={"virtuosic": 0.40, "romantic": 0.30, "dramatic": 0.20, "lyrical": 0.10},
        scale_prefs=["minor", "chromatic", "major"],
        texture="virtuosic",
        pitch_range=(28, 108), vel_range=(20, 127),
        rhythm_feel="rubato", ornament_density=0.44, harm_complexity=0.82,
        char_intervals=[1, 2, 3, 5, 7, 12],
    ),
}

EMOTIONS: Dict[str, EmotionState] = {
    "😢": EmotionState("😢", "Sadness",    -0.8, 0.20, 0.30, 0.40, 0.72, 0.58, "minor",      "legato"),
    "❤️": EmotionState("❤️", "Love",       +0.9, 0.50, 0.20, 0.50, 0.83, 0.73, "major",      "legato"),
    "⚡": EmotionState("⚡", "Energy",     +0.6, 0.95, 0.70, 0.60, 1.40, 0.95, "major",      "staccato"),
    "🌙": EmotionState("🌙", "Mystery",    +0.0, 0.20, 0.50, 0.70, 0.58, 0.42, "chromatic",  "legato"),
    "🔥": EmotionState("🔥", "Passion",    +0.7, 0.85, 0.60, 0.70, 1.08, 0.88, "minor",      "rubato"),
    "😊": EmotionState("😊", "Joy",        +0.95,0.70, 0.10, 0.30, 1.22, 0.80, "major",      "staccato"),
    "😤": EmotionState("😤", "Struggle",   -0.5, 0.90, 0.85, 0.75, 1.12, 1.00, "minor",      "staccato"),
    "🌸": EmotionState("🌸", "Tenderness", +0.8, 0.30, 0.10, 0.30, 0.78, 0.53, "major",      "legato"),
    "🌊": EmotionState("🌊", "Flow",       +0.3, 0.40, 0.30, 0.60, 0.88, 0.63, "pentatonic", "legato"),
    "⚔️": EmotionState("⚔️", "Drama",     -0.3, 0.85, 0.90, 0.80, 1.03, 0.93, "minor",      "mixed"),
    "🕊️": EmotionState("🕊️", "Peace",     +0.8, 0.10, 0.00, 0.20, 0.62, 0.48, "pentatonic", "legato"),
    "🌈": EmotionState("🌈", "Wonder",     +0.9, 0.60, 0.20, 0.70, 0.97, 0.73, "whole_tone", "mixed"),
}

# ─────────────────────────────────────────────────────────────
# FANTASY ENGINE
# ─────────────────────────────────────────────────────────────

class FantasyEngine:
    """
    Blends Composer DNA (60%) with User Emotion (40%) into GenParams,
    then realises texture-specific note generation with
    internal quality testing + auto-correction (up to 3 passes).
    """
    QUALITY_THRESHOLD = 0.62
    MAX_RETRIES = 3

    # --- PARAMETER SYNTHESIS ---

    def compute_params(self, comp: ComposerProfile, emotions: List[EmotionState],
                       section_idx: int, total_sections: int,
                       section_dur_override: float = 8.0,
                       bpm_override: int = 0) -> GenParams:
        if not emotions:
            emotions = [EMOTIONS["❤️"]]
        n = len(emotions)
        avg = lambda attr: sum(getattr(e, attr) for e in emotions) / n

        valence   = avg("valence")
        arousal   = avg("arousal")
        tension   = avg("tension")
        complexity= avg("complexity")
        tempo_f   = avg("tempo_factor")
        vel_f     = avg("velocity_factor")

        # FANTASY BLEND: 60 % composer DNA weight
        # The composer's emotional_dna is remapped to musical intensity
        dna_intensity = sum(comp.emotional_dna.values()) / len(comp.emotional_dna)
        fantasy_tension = 0.60 * dna_intensity + 0.40 * tension

        # Scale selection via voting
        votes: Dict[str, float] = {}
        for e in emotions:
            votes[e.scale_pref] = votes.get(e.scale_pref, 0) + 0.40
        for sp in comp.scale_prefs[:2]:
            votes[sp] = votes.get(sp, 0) + 0.30
        # Chopin special: harmonic minor for high-tension minor
        chosen_scale = max(votes, key=votes.get)
        if chosen_scale == "minor" and comp.name == "Chopin" and tension > 0.5:
            chosen_scale = "harmonic_minor"

        # Root key moves through circle of fifths across the piece
        root_key = (section_idx * 7) % 12

        scale_pitches = build_scale_pitches(root_key, chosen_scale)

        # Tempo
        if bpm_override > 0:
            tempo_bpm = float(bpm_override)
        else:
            tempo_bpm = float(comp.base_tempo) * tempo_f
            tempo_bpm = max(comp.base_tempo * 0.50, min(comp.base_tempo * 2.20, tempo_bpm))

        beat_dur = 60.0 / tempo_bpm

        # Velocity range — proporcjonalne do vel_f, min span = 30
        vlo, vhi = comp.vel_range
        span = vhi - vlo
        elo = max(vlo, int(vlo + span * 0.15))
        ehi = max(elo + 30, min(vhi, int(vlo + span * max(0.35, vel_f))))
        ehi = min(vhi, ehi)

        # Articulation: composer feel dominates
        art_map = {"rubato": "rubato", "strict": "mixed", "dance": "staccato", "free": "legato"}
        art_votes: Dict[str, float] = {art_map[comp.rhythm_feel]: 0.60}
        for e in emotions:
            art_votes[e.articulation] = art_votes.get(e.articulation, 0) + 0.40
        articulation = max(art_votes, key=art_votes.get)

        rubato = 0.12 * arousal + 0.08 * tension if "rubato" in articulation or comp.rhythm_feel == "rubato" else 0.0

        # Pitch space
        plo, phi = comp.pitch_range
        pcr = 0.5 + 0.25 * valence - 0.15 * tension
        pitch_center = int(plo + (phi - plo) * max(0.2, min(0.8, pcr)))
        pitch_spread = int((phi - plo) * (0.30 + 0.38 * arousal))

        phrase_len = max(4, int(16 * (1.0 - 0.45 * arousal)))  # beats

        return GenParams(
            root_key=root_key, scale_type=chosen_scale, scale_pitches=scale_pitches,
            tempo_bpm=tempo_bpm, vel_range=(elo, ehi), beat_dur=beat_dur,
            texture=comp.texture, articulation=articulation, rubato=rubato,
            ornaments=comp.ornament_density * (0.5 + 0.5 * complexity),
            pitch_center=pitch_center, pitch_spread=pitch_spread,
            phrase_len=phrase_len, harm_complexity=comp.harm_complexity * (0.5 + 0.5 * complexity),
            section_dur=section_dur_override,
        )

    # --- TOP-LEVEL GENERATION WITH INTERNAL TESTS ---

    def generate_section(self, comp: ComposerProfile, params: GenParams,
                         prev_notes: Optional[List[NoteEvent]],
                         quality_threshold: float = 0.62) -> Tuple[List[NoteEvent], float]:
        best_notes: List[NoteEvent] = []
        best_score = 0.0
        for attempt in range(self.MAX_RETRIES):
            notes = self._realise(comp, params, prev_notes)
            score = self._quality(notes, params)
            if score > best_score:
                best_notes, best_score = notes, score
            if score >= quality_threshold:
                break
            # auto-correct: tighten spread and harmony
            params = update_params(params,
                                   pitch_spread=int(params.pitch_spread * 0.82),
                                   harm_complexity=params.harm_complexity * 0.88)
        return best_notes, best_score

    def _realise(self, comp: ComposerProfile, params: GenParams,
                 prev: Optional[List[NoteEvent]]) -> List[NoteEvent]:
        melody = self._melody(comp, params, prev)
        accomp = self._accompaniment(comp, params, melody)
        if params.ornaments > 0.25:
            melody = self._ornaments(melody, params)
        if params.rubato > 0.01:
            melody = self._rubato(melody, params.rubato)
        return melody + accomp

    # --- MELODY ---

    def _melody(self, comp: ComposerProfile, params: GenParams,
                prev: Optional[List[NoteEvent]]) -> List[NoteEvent]:
        notes: List[NoteEvent] = []
        t = 0.0
        dur = params.section_dur
        beat = params.beat_dur

        # Seed pitch: continue from last melody note
        if prev:
            prev_mel = [n for n in prev if n.voice == "melody"]
            cur = prev_mel[-1].pitch if prev_mel else params.pitch_center
        else:
            cur = params.pitch_center
        cur = nearest_pitch(cur, params.scale_pitches)

        motif = self._build_motif(comp, params)
        m_pos = 0
        phrase_beat_dur = params.phrase_len * beat

        while t < dur - 0.02:
            # Phrase position (0…1) for dynamic shaping
            phrase_t = (t % phrase_beat_dur) / phrase_beat_dur if phrase_beat_dur > 0 else 0
            phrase_pos = math.sin(math.pi * phrase_t)

            # Note duration
            if params.articulation == "staccato":
                ndur = beat * random.choice([0.25, 0.33, 0.50])
            elif params.articulation == "legato":
                ndur = beat * random.choice([0.50, 0.75, 1.00, 1.50])
            elif params.articulation == "rubato":
                ndur = beat * random.choice([0.50, 0.75, 1.00, 1.25, 1.50, 2.00])
            else:
                ndur = beat * random.choice([0.25, 0.50, 0.75, 1.00])
            ndur = min(ndur, dur - t)
            if ndur < 0.04:
                break

            # Pitch: motif-driven with range guarding
            interval = motif[m_pos % len(motif)]
            target = cur + interval
            if target > params.pitch_center + params.pitch_spread:
                target = cur - abs(interval)
            elif target < params.pitch_center - params.pitch_spread:
                target = cur + abs(interval)
            target = nearest_pitch(max(21, min(108, target)), params.scale_pitches)
            cur = target
            m_pos += 1

            # Velocity: swell at mid-phrase + slight humanisation
            vlo, vhi = params.vel_range
            vel = int(vlo + (vhi - vlo) * (0.4 + 0.6 * phrase_pos))
            vel = max(1, min(127, vel + random.randint(-8, 8)))

            # Breathing pause
            if random.random() < 0.09:
                t += beat * random.choice([0.25, 0.50])
                continue

            notes.append(NoteEvent(cur, vel, t, ndur, 0, "melody"))
            t += ndur

        return notes

    def _build_motif(self, comp: ComposerProfile, params: GenParams) -> List[int]:
        base = comp.char_intervals[:]
        # Texture expansions
        if params.texture == "virtuosic":
            extended = [iv * 2 for iv in base] + [-iv for iv in base] + base
        elif params.texture == "nocturne":
            extended = [iv for iv in base if abs(iv) <= 5] + [0, -1, 1, -2, 2, -3, 3]
        elif params.texture == "counterpoint":
            extended = [1, -1, 2, -2, 1, 3, -1, 2, -3, 1, -1, 2]
        elif params.texture == "impressionist":
            extended = base + [2, 2, -1, 2, 2, -1, 4, -2]  # whole-tone feel
        else:
            extended = base + [-iv for iv in base] + [0, 1, -1]
        random.shuffle(extended)
        return (extended * 3)[:16]

    # --- ACCOMPANIMENT TEXTURES ---

    def _accompaniment(self, comp: ComposerProfile, params: GenParams,
                       melody: List[NoteEvent]) -> List[NoteEvent]:
        dispatch = {
            "nocturne":      self._tex_nocturne,
            "sonata":        self._tex_sonata,
            "classical":     self._tex_alberti,
            "counterpoint":  self._tex_counterpoint,
            "impressionist": self._tex_impressionist,
            "virtuosic":     self._tex_virtuosic,
        }
        fn = dispatch.get(params.texture, self._tex_alberti)
        return fn(params, melody)

    def _progression(self, params: GenParams, section: str = "verse") -> List[int]:
        r = params.root_key
        if params.scale_type in ("minor", "harmonic_minor", "dorian"):
            prog = [r, (r+7)%12, (r+5)%12, (r+7)%12]
        else:
            prog = [r, (r+5)%12, (r+7)%12, r]
        return prog

    def _chord_type(self, params: GenParams) -> str:
        if params.scale_type in ("minor", "harmonic_minor"):
            return "minor7"
        elif params.scale_type == "whole_tone":
            return "dominant7"
        else:
            return "major7"

    def _tex_nocturne(self, params: GenParams, melody: List[NoteEvent]) -> List[NoteEvent]:
        """Chopin: arpegio LH wide span  low→mid→high→mid"""
        acc: List[NoteEvent] = []
        t, dur, beat = 0.0, params.section_dur, params.beat_dur
        prog = self._progression(params)
        vlo, vhi = params.vel_range
        base_v = int((vlo + vhi) / 2 * 0.70)
        i = 0
        while t < dur:
            pc = prog[i % len(prog)]
            ch = chord_pitches(pc, "minor" if "minor" in params.scale_type else "major",
                               params.scale_pitches, 2)
            ch = sorted(set(ch))
            while len(ch) < 3:
                ch.append(ch[-1] + 12)
            # LH arpeggio: low, mid, high, mid
            arp = [ch[0], ch[1], ch[2], ch[1]]
            step = beat / 2
            for j, p in enumerate(arp):
                if t + j * step >= dur:
                    break
                v = max(1, min(127, base_v + random.randint(-8, 8)))
                acc.append(NoteEvent(max(21, min(72, p)), v, t + j * step, step * 0.92, 1, "bass"))
            t += beat * 2
            i += 1
        return acc

    def _tex_sonata(self, params: GenParams, melody: List[NoteEvent]) -> List[NoteEvent]:
        """Beethoven: strong-beat bass octave + off-beat chord stab"""
        acc: List[NoteEvent] = []
        t, dur, beat = 0.0, params.section_dur, params.beat_dur
        prog = self._progression(params)
        vlo, vhi = params.vel_range
        i = 0
        while t < dur:
            pc = prog[i % len(prog)]
            bass = nearest_pitch(pc + 2 * 12, params.scale_pitches)
            bass_lo = nearest_pitch(pc + 12, params.scale_pitches)
            # Octave bass on beat 1
            for bp in [bass_lo, bass]:
                acc.append(NoteEvent(max(21, min(72, bp)), min(127, vhi - 5),
                                     t, beat * 0.88, 1, "bass"))
            # Chord stab off-beat
            if t + beat * 1.5 < dur:
                ch = chord_pitches(pc, "minor" if "minor" in params.scale_type else "major",
                                   params.scale_pitches, 3)
                for p in ch[1:3]:
                    acc.append(NoteEvent(max(21, min(108, p)),
                                         max(1, min(127, vlo + 10)),
                                         t + beat * 1.5, beat * 0.35, 1, "harmony"))
            t += beat * 2
            i += 1
        return acc

    def _tex_alberti(self, params: GenParams, melody: List[NoteEvent]) -> List[NoteEvent]:
        """Mozart: Alberti bass  low-high-mid-high"""
        acc: List[NoteEvent] = []
        t, dur, beat = 0.0, params.section_dur, params.beat_dur
        prog = self._progression(params)
        vlo, vhi = params.vel_range
        base_v = int((vlo + vhi) * 0.46)
        i = 0
        while t < dur:
            pc = prog[i % len(prog)]
            ch = sorted(chord_pitches(pc, "major" if params.scale_type == "major" else "minor",
                                      params.scale_pitches, 3))
            while len(ch) < 3:
                ch.append(ch[-1] + 12)
            pattern = [ch[0], ch[-1], ch[1], ch[-1]]
            step = beat / 2
            for j, p in enumerate(pattern):
                if t + j * step >= dur:
                    break
                v = max(1, min(127, base_v + random.randint(-6, 6)))
                acc.append(NoteEvent(max(21, min(84, p)), v, t + j * step, step * 0.88, 1, "bass"))
            t += beat * 2
            i += 1
        return acc

    def _tex_counterpoint(self, params: GenParams, melody: List[NoteEvent]) -> List[NoteEvent]:
        """Bach: contrary-motion counter-voice, offset by an eigth"""
        acc: List[NoteEvent] = []
        vlo, vhi = params.vel_range
        base_v = int((vlo + vhi) * 0.58)
        for note in melody:
            offset = random.choice([0, params.beat_dur / 4, params.beat_dur / 2])
            # Imitation at lower 5th with contrary motion tendency
            jump = random.choice([-7, -5, -4, -3, 5, 7])
            cp = nearest_pitch(note.pitch + jump, params.scale_pitches)
            cp = max(21, min(84, cp))
            v = max(1, min(127, base_v + random.randint(-10, 10)))
            acc.append(NoteEvent(cp, v, max(0, note.time + offset), note.duration, 1, "harmony"))
        return sorted(acc, key=lambda n: n.time)

    def _tex_impressionist(self, params: GenParams, melody: List[NoteEvent]) -> List[NoteEvent]:
        """Debussy: stacked parallel chords shifting slowly"""
        acc: List[NoteEvent] = []
        t, dur, beat = 0.0, params.section_dur, params.beat_dur
        vlo, vhi = params.vel_range
        base_v = int(vlo + (vhi - vlo) * 0.32)
        roots = [(params.root_key + iv) % 12 for iv in [0, 2, 4, 6, 8, 10]]  # whole-tone roots
        while t < dur:
            rpc = random.choice(roots)
            # Parallel 9th-based voicing
            pitches = [
                nearest_pitch(rpc + 24, params.scale_pitches),
                nearest_pitch(rpc + 36, params.scale_pitches),
                nearest_pitch(rpc + 41, params.scale_pitches),
                nearest_pitch(rpc + 45, params.scale_pitches),
                nearest_pitch(rpc + 50, params.scale_pitches),
            ]
            chord_dur = beat * random.choice([2, 3, 4])
            chord_dur = min(chord_dur, dur - t)
            for p in pitches:
                if 21 <= p <= 96:
                    v = max(1, min(127, base_v + random.randint(-12, 12)))
                    acc.append(NoteEvent(p, v, t, chord_dur * 0.96, 1, "harmony"))
            t += chord_dur
        return acc

    def _tex_virtuosic(self, params: GenParams, melody: List[NoteEvent]) -> List[NoteEvent]:
        """Liszt: LH alternates between bass octaves and rapid scale runs"""
        acc: List[NoteEvent] = []
        t, dur, beat = 0.0, params.section_dur, params.beat_dur
        vlo, vhi = params.vel_range
        while t < dur:
            if random.random() < 0.55:
                # Scale run
                lo = random.randint(28, 55)
                hi = lo + random.randint(14, 38)
                run = [p for p in params.scale_pitches if lo <= p <= hi]
                if random.random() < 0.5:
                    run = list(reversed(run))
                if run:
                    step = min(beat / max(1, len(run)), beat / 8)
                    for j, p in enumerate(run[:20]):
                        if t + j * step >= dur:
                            break
                        frac = j / max(1, len(run) - 1)
                        v = int(vlo + (vhi - vlo) * (0.55 + 0.45 * frac))
                        acc.append(NoteEvent(max(21, min(108, p)), max(1, min(127, v)),
                                             t + j * step, step * 0.88, 1, "bass"))
                t += beat * 2
            else:
                # Bass octave + chord
                pc = params.root_key
                bass_h = nearest_pitch(pc + 36, params.scale_pitches)
                bass_l = nearest_pitch(pc + 24, params.scale_pitches)
                for bp in [bass_l, bass_h]:
                    acc.append(NoteEvent(max(21, min(72, bp)),
                                         min(127, vhi - 8), t, beat * 0.88, 1, "bass"))
                ch = chord_pitches(pc, "minor" if "minor" in params.scale_type else "major",
                                   params.scale_pitches, 4)
                for p in ch:
                    acc.append(NoteEvent(max(21, min(108, p)),
                                         max(1, min(127, vlo + 12)), t + beat, beat * 0.4, 1, "harmony"))
                t += beat * 2
        return acc

    # --- ORNAMENTS & RUBATO ---

    def _ornaments(self, notes: List[NoteEvent], params: GenParams) -> List[NoteEvent]:
        result: List[NoteEvent] = []
        for note in notes:
            result.append(note)
            if (random.random() < params.ornaments * 0.28
                    and note.duration > params.beat_dur * 0.45):
                gp = nearest_pitch(note.pitch + random.choice([-2, -1, 1, 2]),
                                   params.scale_pitches)
                gd = params.beat_dur * 0.08
                result.append(NoteEvent(max(21, min(108, gp)),
                                         max(1, min(127, note.velocity - 18)),
                                         max(0.0, note.time - gd), gd, 0, "ornament"))
        return sorted(result, key=lambda n: n.time)

    def _rubato(self, notes: List[NoteEvent], amount: float) -> List[NoteEvent]:
        if not notes:
            return notes
        max_t = max(n.time for n in notes) + 0.01
        out = []
        for note in notes:
            phase = note.time / max_t * 2 * math.pi
            offset = amount * math.sin(phase + random.uniform(-0.3, 0.3)) * 0.5
            out.append(NoteEvent(note.pitch, note.velocity,
                                  max(0.0, note.time + offset),
                                  note.duration, note.channel, note.voice))
        return sorted(out, key=lambda n: n.time)

    # --- QUALITY SCORE (internal test) ---

    def _quality(self, notes: List[NoteEvent], params: GenParams) -> float:
        melody = [n for n in notes if n.voice == "melody"]
        if not melody:
            return 0.3

        # 1. Scale coherence
        in_scale = sum(1 for n in melody if n.pitch in params.scale_pitches)
        sc = in_scale / len(melody)

        # 2. Pitch range adherence
        plo = params.pitch_center - params.pitch_spread
        phi = params.pitch_center + params.pitch_spread
        in_range = sum(1 for n in melody if plo <= n.pitch <= phi)
        rc = in_range / len(melody)

        # 3. Note density fitness
        total_t = max(n.time + n.duration for n in melody) + 0.01
        density = len(melody) / total_t          # notes/s
        ideal = 60.0 / (params.beat_dur * 100)   # ≈ notes per second
        dc = 1.0 - min(1.0, abs(density - ideal * 3) / (ideal * 3 + 0.1))

        # 4. Interval smoothness (penalise jumps > octave)
        if len(melody) > 1:
            jumps = [abs(melody[i+1].pitch - melody[i].pitch) for i in range(len(melody)-1)]
            avg_j = sum(jumps) / len(jumps)
            ic = 1.0 - min(1.0, avg_j / 12.0)
        else:
            ic = 0.5

        # 5. Velocity variance (too flat = boring)
        vels = [n.velocity for n in melody]
        vlo, vhi = params.vel_range
        span = max(1, vhi - vlo)
        actual_span = max(vels) - min(vels)
        vc = min(1.0, actual_span / (span * 0.4))

        score = sc * 0.30 + rc * 0.25 + dc * 0.18 + ic * 0.17 + vc * 0.10
        return score


# ─────────────────────────────────────────────────────────────
# MIDI BUILDER
# ─────────────────────────────────────────────────────────────

class MIDIBuilder:
    TICKS = 480  # per beat

    def sec_to_ticks(self, sec: float, bpm: float) -> int:
        return int(sec / (60.0 / bpm) * self.TICKS)

    def build(self, sections: List[Tuple[List[NoteEvent], float]], comp: ComposerProfile) -> MidiFile:
        mid = MidiFile(type=1, ticks_per_beat=self.TICKS)

        # --- Conductor track — tempo zmienia się per-sekcja ---
        cond = MidiTrack()
        mid.tracks.append(cond)
        cond.append(MetaMessage("track_name", name=f"{comp.name} Fantasy", time=0))
        cursor_sec = 0.0
        prev_tempo_tick = 0
        # Tymczasowy bpm do przeliczenia ticków (użyjemy pierwszego bpm)
        first_bpm = sections[0][1] if sections else 120.0
        for sec_notes, bpm in sections:
            abs_tick = self.sec_to_ticks(cursor_sec, first_bpm)
            delta = max(0, abs_tick - prev_tempo_tick)
            cond.append(MetaMessage("set_tempo", tempo=int(60_000_000 / bpm), time=delta))
            prev_tempo_tick = abs_tick
            sec_dur = max((n.time + n.duration for n in sec_notes), default=0)
            cursor_sec += sec_dur

        # --- Two note tracks: ch0=melody, ch1=accompaniment ---
        for ch_idx, ch_name in [(0, f"{comp.name} Melody"), (1, f"{comp.name} Accomp")]:
            track = MidiTrack()
            mid.tracks.append(track)
            track.append(MetaMessage("track_name", name=ch_name, time=0))
            track.append(Message("program_change", program=comp.midi_program, channel=ch_idx, time=0))

            events: List[Tuple[int, str, int, int]] = []  # (abs_tick, on/off, pitch, vel)
            offset_sec = 0.0
            for notes, bpm in sections:
                for n in notes:
                    if n.channel != ch_idx:
                        continue
                    t0 = self.sec_to_ticks(n.time + offset_sec, bpm)
                    t1 = self.sec_to_ticks(n.time + n.duration + offset_sec, bpm)
                    events.append((t0, "on",  n.pitch, n.velocity))
                    events.append((t1, "off", n.pitch, 0))
                sec_dur = max((n.time + n.duration for n in notes), default=0)
                offset_sec += sec_dur

            events.sort(key=lambda x: x[0])
            prev_tick = 0
            for (tick, kind, pitch, vel) in events:
                delta = max(0, tick - prev_tick)
                p = max(0, min(127, pitch))
                v = max(0, min(127, vel))
                if kind == "on":
                    track.append(Message("note_on", note=p, velocity=v, channel=ch_idx, time=delta))
                else:
                    track.append(Message("note_off", note=p, velocity=0, channel=ch_idx, time=delta))
                prev_tick = tick

        return mid

    def save(self, mid: MidiFile, path: str) -> bool:
        try:
            mid.save(path)
            return True
        except Exception as e:
            print(f"MIDI save error: {e}")
            return False


# ─────────────────────────────────────────────────────────────
# GUI WIDGETS
# ─────────────────────────────────────────────────────────────

DARK_STYLE = """
QMainWindow, QWidget          { background: #14142b; color: #dde0f5; }
QGroupBox                     { border: 1px solid #334466; border-radius: 6px;
                                margin-top: 10px; font-weight: bold; padding: 4px; }
QGroupBox::title              { color: #7799ee; subcontrol-origin: margin; padding: 0 4px; }
QPushButton                   { background: #1e2a48; border: 1px solid #3355aa;
                                border-radius: 5px; padding: 5px 10px; color: #d0d8f8; }
QPushButton:hover             { background: #283a64; border-color: #88aaff; }
QPushButton:checked           { background: #2a4a80; border-color: #aaccff; color: #ffffff; }
QComboBox, QSpinBox           { background: #1a1a38; border: 1px solid #334466;
                                border-radius: 4px; padding: 4px; color: #d0d8f8; }
QScrollArea                   { border: none; }
QTextEdit                     { background: #0e0e1e; border: 1px solid #223;
                                color: #8899bb; font-family: monospace; font-size: 10px; }
QProgressBar                  { border: 1px solid #334; border-radius: 4px; background: #111; }
QProgressBar::chunk           { background: #2255aa; border-radius: 3px; }
QLabel                        { color: #c0c8e8; }
QCheckBox                     { color: #c0c8e8; }
"""


class EmojiBtn(QPushButton):
    def __init__(self, emoji: str, emotion: EmotionState):
        super().__init__(emoji)
        self.emotion = emotion
        self.setFont(QFont("Segoe UI Emoji", 18))
        self.setFixedSize(50, 50)
        self.setToolTip(
            f"{emotion.name}\n"
            f"Valence {emotion.valence:+.1f}  Arousal {emotion.arousal:.1f}  Tension {emotion.tension:.1f}\n"
            f"Tempo ×{emotion.tempo_factor:.2f}  Scale: {emotion.scale_pref}  Art: {emotion.articulation}"
        )
        self.setStyleSheet(
            "QPushButton{border:2px solid #334;border-radius:8px;background:#1a1a2e;}"
            "QPushButton:hover{border-color:#88aaff;background:#282850;}"
        )


class SectionCard(QFrame):
    def __init__(self, emotion: EmotionState, idx: int):
        super().__init__()
        self.emotion = emotion
        self.setFixedSize(62, 78)
        self.setStyleSheet("QFrame{background:#1e2a40;border:1px solid #446;border-radius:6px;}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lbl = QLabel(emotion.emoji)
        lbl.setFont(QFont("Segoe UI Emoji", 16))
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name = QLabel(emotion.name[:7])
        name.setFont(QFont("Arial", 7))
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name.setStyleSheet("color:#8899aa;")
        lay.addWidget(lbl)
        lay.addWidget(name)


# ─────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────

class IconicComposerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎼 Iconic Composer Fantasy Engine")
        self.resize(1150, 740)

        self.composer    = COMPOSERS["Chopin"]
        self.emotions:   List[EmotionState] = []
        self.sections:   List[Tuple[List[NoteEvent], float]] = []  # (notes, bpm)
        self.midi_file:  Optional[MidiFile] = None
        self.is_playing  = False
        self._comp_btns: Dict[str, QPushButton] = {}

        self.engine  = FantasyEngine()
        self.builder = MIDIBuilder()

        self.rtmidi_out = None
        if RTMIDI_OK:
            try:
                self._midi_sys = rtmidi.MidiOut()
            except Exception:
                pass

        self._build_ui()

    # ── BUILD UI ──────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        root_lay = QHBoxLayout(root)
        self.setCentralWidget(root)

        # ── LEFT: composer + MIDI + params ──
        left = QWidget(); left.setFixedWidth(290)
        ll = QVBoxLayout(left); ll.setSpacing(6)

        # Composer selector
        cg = QGroupBox("🎭 Composer / Virtuoso")
        cg_lay = QGridLayout(cg)
        for i, (name, comp) in enumerate(COMPOSERS.items()):
            btn = QPushButton(f"{comp.emoji} {name}")
            btn.setCheckable(True)
            btn.setChecked(name == "Chopin")
            btn.clicked.connect(lambda _, c=comp: self._select_composer(c))
            self._comp_btns[name] = btn
            cg_lay.addWidget(btn, i // 2, i % 2)
        self.comp_info = QLabel()
        self.comp_info.setWordWrap(True)
        self.comp_info.setStyleSheet("color:#7788aa;font-size:10px;")
        cg_lay.addWidget(self.comp_info, len(COMPOSERS) // 2 + 1, 0, 1, 2)
        ll.addWidget(cg)
        self._update_comp_info()

        # MIDI Out
        mg = QGroupBox("🔌 MIDI Output")
        ml = QVBoxLayout(mg)
        self.midi_combo = QComboBox()
        self._refresh_midi_ports()
        self.midi_con_btn = QPushButton("Connect")
        self.midi_con_btn.clicked.connect(self._connect_midi)
        if not RTMIDI_OK:
            self.midi_con_btn.setEnabled(False)
            self.midi_combo.addItem("rtmidi not installed")
        ml.addWidget(self.midi_combo)
        ml.addWidget(self.midi_con_btn)
        ll.addWidget(mg)

        # Params
        pg = QGroupBox("⚙️ Parameters")
        pl = QGridLayout(pg)
        pl.addWidget(QLabel("BPM (0=Auto):"), 0, 0)
        self.bpm_spin = QSpinBox(); self.bpm_spin.setRange(0, 300); self.bpm_spin.setValue(0)
        self.bpm_spin.setSpecialValueText("Auto")
        pl.addWidget(self.bpm_spin, 0, 1)

        pl.addWidget(QLabel("Section dur (s):"), 1, 0)
        self.dur_spin = QSpinBox(); self.dur_spin.setRange(4, 60); self.dur_spin.setValue(8)
        pl.addWidget(self.dur_spin, 1, 1)

        self.loop_chk = QCheckBox("Loop playback")
        pl.addWidget(self.loop_chk, 2, 0, 1, 2)

        # Min Quality suwak
        pl.addWidget(QLabel("Min Quality:"), 3, 0)
        q_row = QHBoxLayout()
        self.quality_spin = QSlider(Qt.Orientation.Horizontal)
        self.quality_spin.setRange(10, 95)
        self.quality_spin.setValue(62)
        self.quality_spin.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.quality_spin.setTickInterval(10)
        self.quality_lbl = QLabel("0.62")
        self.quality_lbl.setFixedWidth(32)
        self.quality_lbl.setStyleSheet("color:#aabbdd;font-size:10px;")
        self.quality_spin.valueChanged.connect(
            lambda v: self.quality_lbl.setText(f"{v/100:.2f}")
        )
        q_row.addWidget(self.quality_spin)
        q_row.addWidget(self.quality_lbl)
        pl.addLayout(q_row, 3, 1)

        ll.addWidget(pg)
        ll.addStretch()
        root_lay.addWidget(left)

        # ── CENTER: emotions + generator ──
        center = QWidget()
        cl = QVBoxLayout(center); cl.setSpacing(8)

        # Emotion palette
        eg = QGroupBox("💭 Emotional Fantasy  — kliknij emoji aby dodać sekcję do progresji")
        el = QVBoxLayout(eg)
        pal = QWidget()
        pal_lay = QHBoxLayout(pal); pal_lay.setSpacing(5)
        for emoji, em in EMOTIONS.items():
            btn = EmojiBtn(emoji, em)
            btn.clicked.connect(lambda _, e=em: self._add_emotion(e))
            pal_lay.addWidget(btn)
        pal_lay.addStretch()
        el.addWidget(pal)

        # Progression strip
        hdr = QLabel("  Progresja nastrojów:")
        hdr.setStyleSheet("color:#7799ee;font-weight:bold;margin-top:4px;")
        el.addWidget(hdr)

        prog_scroll = QScrollArea()
        prog_scroll.setFixedHeight(92)
        prog_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        prog_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.prog_widget = QWidget()
        self.prog_layout = QHBoxLayout(self.prog_widget)
        self.prog_layout.setSpacing(6)
        self.prog_layout.addStretch()
        prog_scroll.setWidget(self.prog_widget)
        prog_scroll.setWidgetResizable(True)
        el.addWidget(prog_scroll)

        prog_ctrl = QHBoxLayout()
        clr = QPushButton("🗑 Clear"); clr.clicked.connect(self._clear_prog)
        undo = QPushButton("↩ Undo"); undo.clicked.connect(self._undo)
        prog_ctrl.addWidget(clr); prog_ctrl.addWidget(undo); prog_ctrl.addStretch()
        el.addLayout(prog_ctrl)
        cl.addWidget(eg)

        # Generate + play row
        gbg = QGroupBox("🎵 Generate · Play · Save")
        gbl = QHBoxLayout(gbg)

        self.gen_btn = QPushButton("🎼  Generate Fantasy")
        self.gen_btn.setFixedHeight(44)
        self.gen_btn.setStyleSheet(
            "font-size:13px;font-weight:bold;background:#183060;border-color:#4488cc;")
        self.gen_btn.clicked.connect(self._generate)

        self.play_btn = QPushButton("▶  Play")
        self.play_btn.setFixedHeight(44); self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._toggle_play)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setFixedHeight(44); self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)

        self.save_btn = QPushButton("💾  Save MIDI")
        self.save_btn.setFixedHeight(44); self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)

        for b in [self.gen_btn, self.play_btn, self.stop_btn, self.save_btn]:
            gbl.addWidget(b)
        cl.addWidget(gbg)

        # Log + progress
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(185)
        cl.addWidget(self.log)

        self.pbar = QProgressBar(); self.pbar.setRange(0, 100); self.pbar.setValue(0)
        cl.addWidget(self.pbar)

        root_lay.addWidget(center, 1)
        self.setStyleSheet(DARK_STYLE)
        self._log("Witaj w Iconic Composer Fantasy Engine!")
        self._log(f"Wybrany kompozytor: {self.composer.name} ({self.composer.texture})")
        self._log("Dodaj emoji-nastroje do progresji, potem kliknij Generate Fantasy.")

    # ── COMPOSER SELECTION ────────────────────────────────────

    def _select_composer(self, comp: ComposerProfile):
        self.composer = comp
        for name, btn in self._comp_btns.items():
            btn.setChecked(name == comp.name)
        self._update_comp_info()
        self._log(f"→ Kompozytor: {comp.emoji} {comp.name} | {comp.instrument} | {comp.texture}")

    def _update_comp_info(self):
        c = self.composer
        dna = "  ".join(f"{k[:6]}:{v:.0%}" for k, v in c.emotional_dna.items())
        self.comp_info.setText(
            f"{c.instrument}  ·  {c.texture}  ·  {c.rhythm_feel}  ·  {c.base_tempo} BPM\n{dna}"
        )

    # ── EMOTION MANAGEMENT ────────────────────────────────────

    def _add_emotion(self, em: EmotionState):
        self.emotions.append(em)
        card = SectionCard(em, len(self.emotions) - 1)
        self.prog_layout.insertWidget(self.prog_layout.count() - 1, card)
        self._log(f"  + {em.emoji} {em.name}  tempo×{em.tempo_factor:.2f}  "
                  f"scale={em.scale_pref}  art={em.articulation}")

    def _clear_prog(self):
        self.emotions.clear()
        while self.prog_layout.count() > 1:
            item = self.prog_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._log("Progresja wyczyszczona.")

    def _undo(self):
        if self.emotions:
            self.emotions.pop()
            idx = self.prog_layout.count() - 2
            if idx >= 0:
                item = self.prog_layout.takeAt(idx)
                if item.widget():
                    item.widget().deleteLater()

    # ── GENERATION ───────────────────────────────────────────

    def _generate(self):
        if not self.emotions:
            QMessageBox.information(self, "Brak emocji",
                "Dodaj przynajmniej jedną emocję emoji do progresji!")
            return
        self._stop()
        self.sections.clear(); self.midi_file = None
        self.pbar.setValue(0)
        total = len(self.emotions)
        comp = self.composer
        self._log(f"\n▶▶ Generowanie {total} sekcji dla {comp.name} ...")

        prev_notes: Optional[List[NoteEvent]] = None
        all_notes_count = 0

        for i, em in enumerate(self.emotions):
            params = self.engine.compute_params(
                comp, [em],
                section_idx=i, total_sections=total,
                section_dur_override=float(self.dur_spin.value()),
                bpm_override=self.bpm_spin.value()
            )
            notes, score = self.engine.generate_section(
                comp, params, prev_notes,
                quality_threshold=self.quality_spin.value() / 100.0
            )
            self.sections.append((notes, params.tempo_bpm))
            prev_notes = notes
            all_notes_count += len(notes)

            mel = [n for n in notes if n.voice == "melody"]
            acc = [n for n in notes if n.voice != "melody"]
            self._log(
                f"  Sekcja {i+1}: {em.emoji} {em.name} | "
                f"BPM={params.tempo_bpm:.0f} | scale={params.scale_type} | "
                f"mel={len(mel)} acc={len(acc)} | quality={score:.2f}"
            )
            self.pbar.setValue(int((i + 1) / total * 80))
            QApplication.processEvents()

        # Build MIDI
        self._log("Budowanie pliku MIDI …")
        self.midi_file = self.builder.build(self.sections, comp)
        self.pbar.setValue(98)
        self._log(f"✓ Gotowe!  {all_notes_count} nut · {total} sekcji")

        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.pbar.setValue(100)

    # ── PLAYBACK ─────────────────────────────────────────────

    def _toggle_play(self):
        if self.is_playing:
            self._stop()
        else:
            self._play_all()

    def _play_all(self):
        if not self.sections:
            return
        self.is_playing = True
        self.play_btn.setText("⏸  Pause")
        self._schedule_sections()

    def _schedule_sections(self, loop: bool = False):
        offset_ms = 0
        comp = self.composer

        for sec_idx, (notes, bpm) in enumerate(self.sections):
            # Czas trwania sekcji w ms — z rzeczywistym BPM tej sekcji
            sec_dur_s = max((n.time + n.duration for n in notes), default=0)
            sec_dur_ms = int(sec_dur_s * 1000)

            # program_change dla obu kanałów na początku sekcji
            _off = offset_ms
            for ch in range(2):
                QTimer.singleShot(_off, lambda c=ch: self._send_program(comp.midi_program, c))

            # Ciągła ekspresja CC11 co 40 ms przez całą sekcję
            cc_interval_ms = 40
            steps = max(1, sec_dur_ms // cc_interval_ms)
            for step in range(steps + 1):
                t_ms = offset_ms + step * cc_interval_ms
                frac = step / steps
                fade_in     = min(1.0, frac * 6)
                phrase_shape = 0.55 + 0.45 * math.sin(math.pi * frac)
                cc_val = int(max(20, min(127, 127 * fade_in * phrase_shape)))
                for ch in range(2):
                    QTimer.singleShot(t_ms, lambda v=cc_val, c=ch: self._send_cc(11, v, c))

            # Nuty — czas w ms = note.time (sekundy w skali sekcji) × 1000
            # note.time jest już w sekundach rzeczywistych (bpm uwzględniony w beat_dur)
            for note in notes:
                d0 = offset_ms + int(note.time * 1000)
                d1 = d0 + max(50, int(note.duration * 1000 * 0.92))
                QTimer.singleShot(d0, lambda n=note: self._note_on(n))
                QTimer.singleShot(d1, lambda n=note: self._note_off(n))

            offset_ms += sec_dur_ms + 150

        if self.loop_chk.isChecked():
            QTimer.singleShot(offset_ms + 300,
                lambda: self._schedule_sections(loop=True) if self.is_playing else None)
        else:
            QTimer.singleShot(offset_ms + 200, self._on_playback_finished)

    def _on_playback_finished(self):
        if self.is_playing:
            self._stop()
            self._log("▶ Odtwarzanie zakończone.")

    def _stop(self):
        self.is_playing = False
        self.play_btn.setText("▶  Play")
        if self.rtmidi_out:
            for ch in range(2):
                for p in range(128):
                    try:
                        self.rtmidi_out.send_message([0x80 | ch, p, 0])
                    except Exception:
                        pass

    def _note_on(self, note: NoteEvent):
        if self.rtmidi_out:
            ch = min(1, note.channel)
            try:
                self.rtmidi_out.send_message([0x90 | ch,
                                              max(0, min(127, note.pitch)),
                                              max(0, min(127, note.velocity))])
            except Exception:
                pass

    def _note_off(self, note: NoteEvent):
        if self.rtmidi_out:
            ch = min(1, note.channel)
            try:
                self.rtmidi_out.send_message([0x80 | ch, max(0, min(127, note.pitch)), 0])
            except Exception:
                pass

    def _send_program(self, program: int, channel: int):
        """FIX #4 — wysyła program_change (instrument GM) do portu rtmidi."""
        if self.rtmidi_out:
            try:
                self.rtmidi_out.send_message([0xC0 | channel, max(0, min(127, program))])
            except Exception:
                pass

    def _send_cc(self, cc: int, value: int, channel: int):
        """FIX #1 — wysyła Control Change (np. CC11 Expression) do portu rtmidi."""
        if self.rtmidi_out:
            try:
                self.rtmidi_out.send_message([0xB0 | channel,
                                              max(0, min(127, cc)),
                                              max(0, min(127, value))])
            except Exception:
                pass

    # ── MIDI I/O ─────────────────────────────────────────────

    def _refresh_midi_ports(self):
        if not RTMIDI_OK:
            return
        try:
            mo = rtmidi.MidiOut()
            ports = mo.get_ports()
            self.midi_combo.clear()
            self.midi_combo.addItems(ports if ports else ["No MIDI ports found"])
        except Exception:
            pass

    def _connect_midi(self):
        if not RTMIDI_OK:
            return
        idx = self.midi_combo.currentIndex()
        try:
            self.rtmidi_out = rtmidi.MidiOut()
            self.rtmidi_out.open_port(idx)
            self._log(f"✓ MIDI połączony: {self.midi_combo.currentText()}")
            self.statusBar().showMessage(f"MIDI: {self.midi_combo.currentText()}")
        except Exception as e:
            self._log(f"✗ MIDI błąd: {e}")
            self.rtmidi_out = None

    def _save(self):
        if self.midi_file is None:
            QMessageBox.warning(self, "Brak MIDI", "Najpierw wygeneruj kompozycję.")
            return
        fname, _ = QFileDialog.getSaveFileName(
            self, "Zapisz plik MIDI",
            f"{self.composer.name}_fantasy.mid",
            "MIDI Files (*.mid)"
        )
        if fname:
            ok = self.builder.save(self.midi_file, fname)
            self._log(f"{'✓ Zapisano:' if ok else '✗ Błąd zapisu:'} {fname}")

    # ── HELPERS ───────────────────────────────────────────────

    def _log(self, msg: str):
        self.log.append(f"▸ {msg}")
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = IconicComposerApp()
    win.show()
    sys.exit(app.exec())
