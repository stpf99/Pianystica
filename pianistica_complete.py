# ================================
# rewire.py — Like Author Mode + wybór MIDI Out + BPM i ms/q + Simplify from Chaos
# ================================
import sys
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QPushButton, QComboBox, QGroupBox, QHBoxLayout, QSpinBox
)
from PyQt6.QtCore import QTimer

import rtmidi

# ================================
# SILNIK SIMPLIFY FROM CHAOS MODE
# ================================
@dataclass
class ChaosNote:
    pitch: int
    velocity: int
    time: float
    duration: float
    chaos_origin: str = "pure"  # pure, simplified, expanded
    coherence_score: float = 0.0
    musical_features: Dict = field(default_factory=dict)

@dataclass
class ChaosAnalysis:
    scale_coherence: float = 0.0
    rhythmic_pattern: float = 0.0
    pitch_clustering: float = 0.0
    velocity_consistency: float = 0.0
    temporal_flow: float = 0.0
    dominant_intervals: List[int] = field(default_factory=list)
    suggested_key: int = 0
    tempo_stability: float = 0.0

class ChaosSimplifier:
    def __init__(self):
        self.chaos_history: List[List[ChaosNote]] = []
        self.simplification_history: List[List[ChaosNote]] = []
        self.generation_count = 0
        self.current_chaos: List[ChaosNote] = []
        self.last_simplified: List[ChaosNote] = []
        
    def generate_pure_chaos(self, duration: float = 8.0) -> List[ChaosNote]:
        """Generuje czysty chaos nut"""
        notes = []
        chaos_density = random.uniform(0.1, 0.8)  # gęstość chaosu
        
        t = 0.0
        while t < duration:
            if random.random() < chaos_density:
                pitch = random.randint(21, 108)  # pełny zakres pianina
                velocity = random.randint(1, 127)
                note_duration = random.uniform(0.05, 2.0)
                
                note = ChaosNote(
                    pitch=pitch,
                    velocity=velocity,
                    time=t,
                    duration=note_duration,
                    chaos_origin="pure"
                )
                notes.append(note)
            
            t += random.uniform(0.05, 0.5)
        
        return sorted(notes, key=lambda x: x.time)
    
    def expand_from_simplified(self, simplified_notes: List[ChaosNote]) -> List[ChaosNote]:
        """Tworzy nowy chaos bazując na uproszczonych danych"""
        if not simplified_notes:
            return self.generate_pure_chaos()
            
        expanded = []
        
        # Analizuj cechy uproszczonych nut
        avg_pitch = sum(n.pitch for n in simplified_notes) / len(simplified_notes)
        pitch_range = max(n.pitch for n in simplified_notes) - min(n.pitch for n in simplified_notes)
        avg_velocity = sum(n.velocity for n in simplified_notes) / len(simplified_notes)
        
        for base_note in simplified_notes:
            # Dla każdej uproszczonej nuty, generuj chaos wokół niej
            chaos_variants = random.randint(2, 8)
            
            for _ in range(chaos_variants):
                # Chaos inspirowany bazową nutą
                pitch_deviation = random.randint(-pitch_range//2, pitch_range//2)
                new_pitch = max(21, min(108, base_note.pitch + pitch_deviation))
                
                velocity_chaos = random.randint(-30, 30)
                new_velocity = max(1, min(127, base_note.velocity + velocity_chaos))
                
                time_chaos = random.uniform(-0.5, 1.5)
                new_time = max(0, base_note.time + time_chaos)
                
                duration_chaos = random.uniform(0.1, 2.0)
                
                expanded_note = ChaosNote(
                    pitch=new_pitch,
                    velocity=new_velocity,
                    time=new_time,
                    duration=duration_chaos,
                    chaos_origin="expanded"
                )
                expanded.append(expanded_note)
        
        return sorted(expanded, key=lambda x: x.time)
    
    def analyze_chaos(self, chaos_notes: List[ChaosNote]) -> ChaosAnalysis:
        """Analizuje chaos pod kątem cech muzycznych"""
        if not chaos_notes:
            return ChaosAnalysis()
            
        analysis = ChaosAnalysis()
        
        # Analiza skali/tonacji
        pitch_classes = [n.pitch % 12 for n in chaos_notes]
        pc_counts = {pc: pitch_classes.count(pc) for pc in range(12)}
        dominant_pc = max(pc_counts.items(), key=lambda x: x[1])[0]
        analysis.suggested_key = dominant_pc
        
        # Spójność skali (jak bardzo nuty pasują do skali durowej od dominującego PC)
        major_scale_pattern = [0, 2, 4, 5, 7, 9, 11]
        scale_notes = [(dominant_pc + interval) % 12 for interval in major_scale_pattern]
        in_scale_count = sum(1 for pc in pitch_classes if pc in scale_notes)
        analysis.scale_coherence = in_scale_count / len(pitch_classes) if pitch_classes else 0
        
        # Analiza rytmiczności
        if len(chaos_notes) > 1:
            time_intervals = [chaos_notes[i+1].time - chaos_notes[i].time 
                            for i in range(len(chaos_notes)-1)]
            time_intervals = [t for t in time_intervals if t > 0]
            if time_intervals:
                avg_interval = sum(time_intervals) / len(time_intervals)
                variance = sum((t - avg_interval) ** 2 for t in time_intervals) / len(time_intervals)
                analysis.tempo_stability = 1.0 / (1.0 + variance)  # im mniejsza wariancja, tym stabilniej
        
        # Clustering wysokości
        pitches = [n.pitch for n in chaos_notes]
        if len(set(pitches)) < len(pitches) * 0.8:  # jeśli jest dużo powtórzeń
            analysis.pitch_clustering = 0.8
        else:
            analysis.pitch_clustering = 0.2
            
        # Consistency velocity
        velocities = [n.velocity for n in chaos_notes]
        if velocities:
            vel_range = max(velocities) - min(velocities)
            analysis.velocity_consistency = 1.0 - (vel_range / 126.0)  # im mniejszy zakres, tym większa spójność
            
        # Dominujące interwały
        intervals = []
        for i in range(len(chaos_notes)-1):
            interval = abs(chaos_notes[i+1].pitch - chaos_notes[i].pitch)
            if interval <= 12:  # tylko interwały w obrębie oktawy
                intervals.append(interval)
        
        if intervals:
            interval_counts = {}
            for interval in intervals:
                interval_counts[interval] = interval_counts.get(interval, 0) + 1
            analysis.dominant_intervals = sorted(interval_counts.items(), 
                                               key=lambda x: x[1], reverse=True)[:3]
            analysis.dominant_intervals = [x[0] for x in analysis.dominant_intervals]
        
        return analysis
    
    def simplify_chaos(self, chaos_notes: List[ChaosNote], analysis: ChaosAnalysis) -> List[ChaosNote]:
        """Upraszcza chaos na podstawie analizy muzycznej"""
        if not chaos_notes:
            return []
            
        simplified = []
        
        # Filtruj i koryguj nuty na podstawie analizy
        for note in chaos_notes:
            simplified_note = ChaosNote(
                pitch=note.pitch,
                velocity=note.velocity,
                time=note.time,
                duration=note.duration,
                chaos_origin="simplified",
                coherence_score=0.0
            )
            
            # Korekta wysokości do skali
            if analysis.scale_coherence < 0.5:  # jeśli chaos jest bardzo niespójny skalowo
                # Przesuń nutę do najbliższej nuty ze skali
                major_scale_pattern = [0, 2, 4, 5, 7, 9, 11]
                scale_notes = [(analysis.suggested_key + interval) % 12 for interval in major_scale_pattern]
                
                current_pc = note.pitch % 12
                closest_scale_pc = min(scale_notes, key=lambda x: min(abs(x - current_pc), 
                                                                    abs(x - current_pc + 12), 
                                                                    abs(x - current_pc - 12)))
                
                octave = note.pitch // 12
                simplified_note.pitch = octave * 12 + closest_scale_pc
            
            # Korekta velocity dla większej spójności
            if analysis.velocity_consistency < 0.5:
                # Normalizuj velocity do bardziej spójnego zakresu
                target_velocity = 60 + int((note.velocity - 64) * 0.5)  # zmniejsz ekstremalne wartości
                simplified_note.velocity = max(30, min(100, target_velocity))
            
            # Korekta rytmiczna
            if analysis.tempo_stability < 0.3:
                # Quantize czas do najbliższej "sensownej" wartości
                quantum = 0.25  # ćwierćnuta
                quantized_time = round(note.time / quantum) * quantum
                simplified_note.time = quantized_time
            
            # Oblicz coherence_score dla tej nuty
            coherence_factors = [
                analysis.scale_coherence,
                analysis.velocity_consistency,
                analysis.tempo_stability,
                analysis.pitch_clustering
            ]
            simplified_note.coherence_score = sum(coherence_factors) / len(coherence_factors)
            
            simplified.append(simplified_note)
        
        # Usuń nuty z bardzo niskim coherence_score
        simplified = [n for n in simplified if n.coherence_score > 0.3]
        
        return sorted(simplified, key=lambda x: x.time)
    
    def next_iteration(self) -> Dict:
        """Główna metoda - generuje następną iterację chaos->simplify"""
        self.generation_count += 1
        
        if self.generation_count == 1:
            # Pierwsze uruchomienie - czysty chaos
            self.current_chaos = self.generate_pure_chaos()
        else:
            # Kolejne uruchomienia - chaos z poprzednich uproszczonych danych
            self.current_chaos = self.expand_from_simplified(self.last_simplified)
        
        # Analizuj chaos
        analysis = self.analyze_chaos(self.current_chaos)
        
        # Uprość chaos
        self.last_simplified = self.simplify_chaos(self.current_chaos, analysis)
        
        # Zapisz historię
        self.chaos_history.append(self.current_chaos.copy())
        self.simplification_history.append(self.last_simplified.copy())
        
        # Przygotuj dane do odtworzenia
        midi_notes = []
        for note in self.last_simplified:
            midi_notes.append((note.pitch, note.velocity, note.time))
        
        chaos_info = f"Gen{self.generation_count} | Scale:{analysis.scale_coherence:.2f} | Rhythm:{analysis.tempo_stability:.2f} | Notes:{len(self.last_simplified)}"
        
        return {
            "type": "SimplifyFromChaos",
            "name": f"Chaos→Order: {chaos_info}",
            "notes": midi_notes,
            "analysis": analysis,
            "chaos_count": len(self.current_chaos),
            "simplified_count": len(self.last_simplified)
        }

# ================================
# ODTWARZACZ SIMPLIFY FROM CHAOS
# ================================
class ChaosSimplifyPlayer:
    def __init__(self, parent):
        self.parent = parent
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next)
        self.is_playing = False
        self.simplifier = ChaosSimplifier()
        
    def start(self):
        self.stop()
        self.is_playing = True
        self.simplifier = ChaosSimplifier()  # Reset
        self.play_next()
        # Bardzo powolny rytm - każda iteracja co 15-25 sekund
        self.timer.start(random.randint(15000, 25000))
    
    def stop(self):
        self.is_playing = False
        try:
            self.timer.stop()
        except Exception:
            pass
    
    def play_next(self):
        if not self.is_playing:
            return
            
        iteration_data = self.simplifier.next_iteration()
        self.parent.question_label.setText(iteration_data['name'])
        
        # Odtwarzaj uproszczone nuty bardzo powoli
        for (note, vel, t) in iteration_data['notes']:
            delay_ms = int(max(0, t) * 2000)  # 2x wolniej niż normalnie
            QTimer.singleShot(delay_ms, lambda n=note, v=vel: self._note_on(n, v))
            QTimer.singleShot(delay_ms + 1000, lambda n=note: self._note_off(n))  # dłuższe nuty
    
    def _note_on(self, n, v):
        if self.parent.rtmidi_output:
            self.parent.rtmidi_output.send_message([0x90, n, v])
    
    def _note_off(self, n):
        if self.parent.rtmidi_output:
            self.parent.rtmidi_output.send_message([0x80, n, 0])

# ================================
# SILNIK LIKE AUTHOR MODE
# ================================
@dataclass
class LANote:
    pitch: int
    velocity: int
    time: float
    duration: float
    accent: Optional[str] = None

@dataclass
class LAConfig:
    base_octave: int = 36
    n_octaves: int = 3
    start_anchor: str = "center"
    choose_single_or_pair: str = "auto"
    pair_mode: str = "consonant"
    pair_stride: int = 2
    shift_step: int = 3
    shift_every: int = 0
    chop_fraction: int = 4
    tempo_quarter: float = 0.5
    intro_len: int = 2
    verse_len: int = 4
    chorus_len: int = 4
    outro_len: int = 2
    motif_len: int = 16
    velocity_base: int = 80
    velocity_span: int = 20
    structure_order: List[str] = field(default_factory=lambda: ["intro", "verse", "chorus", "verse", "chorus", "outro"])

class LikeAuthorArranger:
    CONSONANT_INTERVALS = [0, 3, 4, 7, 8, 9, 12]
    MAJOR_STEPS = [2, 2, 1, 2, 2, 2, 1]

    def __init__(self, current_scale_id: int = 0, critic=None, config: Optional[LAConfig] = None):
        self.scale_id = current_scale_id
        self.critic = critic
        self.cfg = config or LAConfig()
        self._motif: List[LANote] = []
        self._section_index = 0
        self._repeat_counter = 0
        self._build_scale_cache()
        self._build_motif()

    def _build_scale_cache(self):
        pc = [self.scale_id % 12]
        for st in self.MAJOR_STEPS:
            pc.append((pc[-1] + st) % 12)
        self.scale_pcs = pc[:-1]

    def _in_scale(self, pitch: int) -> bool:
        return (pitch % 12) in self.scale_pcs

    def _nearest_scale_pitch(self, p: int) -> int:
        candidates = []
        for o in range(-2, 3):
            for pc in self.scale_pcs:
                cand = pc + 12 * o
                dist = abs((p % 12) - pc)
                candidates.append((dist, p + (cand - (p % 12))))
        return min(candidates, key=lambda x: (x[0], abs(x[1]-p)))[1]

    def _anchor_octave(self) -> int:
        o0 = self.cfg.base_octave
        span = 12 * max(1, self.cfg.n_octaves)
        if self.cfg.start_anchor == "central_low":
            return o0
        if self.cfg.start_anchor == "central_high":
            return o0 + span - 12
        return o0 + (span // 2) - 6

    def _rand_vel(self) -> int:
        base = self.cfg.velocity_base
        return max(20, min(120, base + random.randint(-self.cfg.velocity_span, self.cfg.velocity_span)))

    def _pair_for(self, pitch: int) -> Optional[int]:
        if self.cfg.choose_single_or_pair == "single":
            return None
        if self.cfg.choose_single_or_pair == "auto" and random.random() < 0.5:
            return None
        if self.cfg.pair_mode == "consonant":
            iv = random.choice(self.CONSONANT_INTERVALS)
        elif self.cfg.pair_mode == "minor":
            iv = random.choice([0, 3, 7, 10, 12])
        else:
            iv = random.choice([0, 4, 7, 11, 12])
        return max(0, min(127, pitch + iv))

    def _build_motif(self):
        self._motif.clear()
        t = 0.0
        anchor = self._anchor_octave()
        cur = anchor
        for i in range(self.cfg.motif_len):
            jump = random.choice([0, 1, 2, 2, 3, 4, 5]) * random.choice([-1, 1])
            cur = max(0, min(115, cur + jump))
            cur = self._nearest_scale_pitch(cur)
            n = LANote(pitch=cur, velocity=self._rand_vel(), time=t, duration=self.cfg.tempo_quarter)
            self._motif.append(n)
            if self.cfg.pair_stride and (i % self.cfg.pair_stride == 0):
                p2 = self._pair_for(cur)
                if p2 is not None:
                    self._motif.append(LANote(pitch=p2, velocity=max(1, n.velocity-10), time=t, duration=self.cfg.tempo_quarter, accent="pair"))
            t += self.cfg.tempo_quarter
        self._motif.sort(key=lambda x: x.time)

    def _transpose(self, notes: List[LANote], semitones: int) -> List[LANote]:
        out = []
        for n in notes:
            p = max(0, min(127, n.pitch + semitones))
            if not self._in_scale(p):
                p = self._nearest_scale_pitch(p)
            out.append(LANote(p, n.velocity, n.time, n.duration, n.accent))
        return out

    def _chop(self, notes: List[LANote], fraction: int) -> List[LANote]:
        out: List[LANote] = []
        for n in notes:
            if fraction <= 1 or n.duration <= 0.05:
                out.append(n)
                continue
            d = n.duration / fraction
            for i in range(fraction):
                out.append(LANote(n.pitch, int(n.velocity * (0.9 + 0.1 * (i==0))), n.time + i * d, d, n.accent))
        return out

    def _accentize(self, notes: List[LANote], style: str) -> List[LANote]:
        profiles = {"intro": 0.85, "verse": 1.0, "chorus": 1.15, "outro": 0.9, "kick": 1.25}
        mul = profiles.get(style, 1.0)
        out = []
        for i, n in enumerate(notes):
            vel = int(max(1, min(127, n.velocity * mul)))
            out.append(LANote(n.pitch, vel, n.time, n.duration, style))
        return out

    def _apply_variations(self, base: List[LANote], section: str, rep_index: int) -> List[LANote]:
        notes = [LANote(n.pitch, n.velocity, n.time, n.duration, n.accent) for n in base]
        if self.cfg.shift_every and (rep_index % self.cfg.shift_every == 0):
            notes = self._transpose(notes, self.cfg.shift_step)
        if section in ("chorus", "outro") or (rep_index % 2 == 1):
            notes = self._chop(notes, self.cfg.chop_fraction)
        notes = self._accentize(notes, section)
        return sorted(notes, key=lambda x: (x.time, x.pitch))

    def next_structure(self) -> Dict:
        if self._section_index >= len(self.cfg.structure_order):
            self._section_index = len(self.cfg.structure_order) - 1
        section = self.cfg.structure_order[self._section_index]
        base = [LANote(n.pitch, n.velocity, n.time, n.duration, section) for n in self._motif]
        notes = self._apply_variations(base, section, self._repeat_counter)
        name = f"LikeAuthor:{section.capitalize()} (rep {self._repeat_counter+1})"
        self._repeat_counter += 1
        max_rep = {"intro": 1, "verse": 2, "chorus": 2, "outro": 1}.get(section, 1)
        if self._repeat_counter >= max_rep:
            self._section_index += 1
            self._repeat_counter = 0
        midi_notes: List[Tuple[int, int, float]] = []
        for n in notes:
            midi_notes.append((n.pitch, n.velocity, n.time))
        return {"type": "LikeAuthor", "name": name, "notes": midi_notes}

# ================================
# ODTWARZACZ LIKE AUTHOR
# ================================
class EnhancedLikeAuthorModePlayer:
    def __init__(self, parent):
        self.parent = parent
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next)
        self.is_playing = False
        self.arranger: Optional[LikeAuthorArranger] = None
        self.base_octave = 36
        self.velocity = 80
        self.current_scale_id = 0

    def _build_config_from_ui(self) -> LAConfig:
        bpm = self.parent.la_bpm.value()
        q_ms = self.parent.la_tempo_quarter.value()
        # priorytet: BPM, ale jeśli ktoś zmieni ms/q — używamy tego
        tempo_quarter = 60.0 / bpm if bpm > 0 else max(0.05, q_ms/1000.0)
        return LAConfig(
            base_octave=self.base_octave,
            n_octaves=self.parent.la_n_oct.value(),
            start_anchor=self.parent.la_start_anchor.currentText(),
            choose_single_or_pair='auto',
            pair_mode=self.parent.la_pair_mode.currentText(),
            pair_stride=self.parent.la_pair_stride.value(),
            shift_step=self.parent.la_shift_step.value(),
            shift_every=self.parent.la_shift_every.value(),
            chop_fraction=self.parent.la_chop_fraction.value(),
            tempo_quarter=tempo_quarter,
            motif_len=16,
            velocity_base=self.velocity,
            velocity_span=18,
        )

    def start(self):
        self.stop()
        self.is_playing = True
        cfg = self._build_config_from_ui()
        self.arranger = LikeAuthorArranger(self.current_scale_id, config=cfg)
        self.play_next()
        self.timer.start(int(cfg.tempo_quarter * 1000 * 8))

    def stop(self):
        self.is_playing = False
        try:
            self.timer.stop()
        except Exception:
            pass

    def play_next(self):
        if not self.is_playing or self.arranger is None:
            return
        structure = self.arranger.next_structure()
        self.parent.question_label.setText(structure['name'])
        for (note, vel, t) in structure['notes']:
            QTimer.singleShot(int(max(0, t) * 1000), lambda n=note, v=vel: self._note_on(n, v))
            QTimer.singleShot(int(max(0, t + self.arranger.cfg.tempo_quarter) * 1000), lambda n=note: self._note_off(n))

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
        self.setWindowTitle("Piano Learning with Like Author Mode + Simplify from Chaos")
        self.resize(900, 500)
        central = QWidget(); layout = QVBoxLayout(central)
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
        self.mode_combo.addItems(["Song Mode", "Jazz Theory Mode", "Relaxation Mode", "Like Author Mode", "Simplify from Chaos"])
        self.mode_combo.currentTextChanged.connect(self.switch_mode)
        layout.addWidget(self.mode_combo)

        # --- Like Author controls ---
        self.like_author_controls = QGroupBox("Like Author")
        la_layout = QHBoxLayout(self.like_author_controls)
        self.la_start_anchor = QComboBox(); self.la_start_anchor.addItems(["center", "central_low", "central_high"])
        self.la_n_oct = QSpinBox(); self.la_n_oct.setRange(1, 6); self.la_n_oct.setValue(3)
        self.la_pair_mode = QComboBox(); self.la_pair_mode.addItems(["consonant", "minor", "major"])
        self.la_pair_stride = QSpinBox(); self.la_pair_stride.setRange(0, 8); self.la_pair_stride.setValue(2)
        self.la_shift_step = QSpinBox(); self.la_shift_step.setRange(-12, 12); self.la_shift_step.setValue(3)
        self.la_shift_every = QSpinBox(); self.la_shift_every.setRange(0, 8); self.la_shift_every.setValue(0)
        self.la_chop_fraction = QSpinBox(); self.la_chop_fraction.setRange(1, 8); self.la_chop_fraction.setValue(4)
        self.la_bpm = QSpinBox(); self.la_bpm.setRange(20, 300); self.la_bpm.setValue(120); self.la_bpm.setSuffix(" BPM")
        self.la_tempo_quarter = QSpinBox(); self.la_tempo_quarter.setRange(100, 2000); self.la_tempo_quarter.setValue(500); self.la_tempo_quarter.setSuffix(" ms/q")

        for w in [
            ("Anchor", self.la_start_anchor),
            ("Octaves", self.la_n_oct),
            ("Pair", self.la_pair_mode),
            ("Stride", self.la_pair_stride),
            ("Shift", self.la_shift_step),
            ("Every", self.la_shift_every),
            ("Chop", self.la_chop_fraction),
            ("BPM", self.la_bpm),
            ("Quarter", self.la_tempo_quarter)
        ]:
            la_layout.addWidget(QLabel(w[0])); la_layout.addWidget(w[1])
        layout.addWidget(self.like_author_controls)

        self.like_author_player = None
        self.chaos_simplify_player = None

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
        # Zatrzymaj wszystkie poprzednie tryby
        if self.like_author_player:
            self.like_author_player.stop()
        if self.chaos_simplify_player:
            self.chaos_simplify_player.stop()
            
        if mode == "Like Author Mode":
            self.question_label.setText("Like Author Mode Active")
            if not self.like_author_player:
                self.like_author_player = EnhancedLikeAuthorModePlayer(self)
            self.like_author_player.start()
        elif mode == "Simplify from Chaos":
            self.question_label.setText("Simplify from Chaos - Generating initial chaos...")
            if not self.chaos_simplify_player:
                self.chaos_simplify_player = ChaosSimplifyPlayer(self)
            self.chaos_simplify_player.start()
        else:
            self.question_label.setText(f"{mode} - Not implemented yet")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PianoLearningApp()
    win.show()
    sys.exit(app.exec())
        