"""Session programs: sequences of phases that ease in, train, and ease out.

Every program follows the same arc so sessions feel unhurried:

  arrive    soundscape fades in, the brain's resting levels are measured
  warm-up   an easy target so rewards come often
  training  blocks of focused training, separated by short breathing interludes
  integrate rewards stop, the soundscape thins and fades out

The condition programs follow protocols from published neurofeedback research.
They are training programs, not treatments: NOVA is not a medical device.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .protocols import PROTOCOLS, Protocol, resolve_protocol

# Phase kinds. "baseline" calibrates; only "training" gives rewards.
KINDS = ("baseline", "training", "rest", "cooldown")
SOUNDSCAPES = {
    "aurora": "Aurora: airy Lydian pads and glass shimmer",
    "ocean": "Deep Ocean: low swells, slow waves, distant bells",
    "celestial": "Celestial: bright bells over a floating major drone",
    "rain": "Forest Rain: soft rain, warm pads, wooden chimes",
    "nebula": "Nebula: dark, slow drones with a glowing upper layer",
    "dawn": "Dawn: warm, rising major chords",
}
CUSTOM_DIR = Path.home() / ".nova" / "programs"


@dataclass(frozen=True)
class Phase:
    name: str
    kind: str
    minutes: float
    protocol: str | dict | None = None  # protocol name or custom spec; required for training
    reward_rate: float = 0.6  # target fraction of time rewarded (training only)
    guidance: str = ""
    binaural: float | None = None  # binaural beat (Hz) mixed under the soundscape

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"phase kind must be one of {', '.join(KINDS)}")
        if self.minutes <= 0:
            raise ValueError(f"phase '{self.name}' needs a positive duration")
        if self.kind == "training" and self.protocol is None:
            raise ValueError(f"training phase '{self.name}' needs a protocol")
        if not 0.05 <= self.reward_rate <= 0.95:
            raise ValueError("reward_rate must be between 0.05 and 0.95")

    @property
    def seconds(self) -> float:
        return self.minutes * 60.0

    def resolved_protocol(self) -> Protocol | None:
        return resolve_protocol(self.protocol) if self.protocol is not None else None


@dataclass(frozen=True)
class Program:
    id: str
    name: str
    category: str  # "condition", "state" or "custom"
    target: str  # short plain-language goal
    summary: str
    phases: tuple[Phase, ...]
    soundscape: str = "aurora"
    eyes: str = "open"  # "open" or "closed"
    evidence: str = ""  # "Most studied", "Emerging", "Limited" or "" for custom
    evidence_note: str = ""
    sites: tuple[str, ...] = ()
    caution: str = ""

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("a program needs at least one phase")
        if self.soundscape not in SOUNDSCAPES:
            raise ValueError(f"unknown soundscape {self.soundscape!r}; choose from {', '.join(SOUNDSCAPES)}")
        if not any(p.kind == "training" for p in self.phases):
            raise ValueError("a program needs at least one training phase")

    @property
    def minutes(self) -> float:
        return sum(p.minutes for p in self.phases)

    def protocols(self) -> dict[str, Protocol]:
        """Distinct protocols used by the program, in order of first use."""
        out: dict[str, Protocol] = {}
        for phase in self.phases:
            proto = phase.resolved_protocol()
            if proto is not None and proto.name not in out:
                out[proto.name] = proto
        return out

    def check_channels(self, channel_names: list[str]) -> list[str]:
        return [msg for p in self.protocols().values() if (msg := p.check_channels(channel_names))]

    def scaled(self, factor: float) -> "Program":
        """Lengthen or shorten training blocks; arrival and integration keep their length."""
        phases = tuple(
            replace(p, minutes=round(p.minutes * factor, 2)) if p.kind == "training" else p for p in self.phases
        )
        return replace(self, phases=phases)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["minutes"] = self.minutes
        d["protocols"] = {name: p.to_spec() for name, p in self.protocols().items()}
        for phase, pd in zip(self.phases, d["phases"]):
            proto = phase.resolved_protocol()
            pd["protocol_name"] = proto.name if proto else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Program":
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        fields["phases"] = tuple(Phase(**{k: v for k, v in p.items() if k in Phase.__dataclass_fields__}) for p in d["phases"])
        fields["sites"] = tuple(fields.get("sites", ()))
        fields.setdefault("category", "custom")
        fields.setdefault("target", "")
        fields.setdefault("summary", "")
        return cls(**fields)


def simple_program(protocol: Protocol, baseline_sec: float, training_sec: float, reward_rate: float = 0.6) -> Program:
    """A single baseline + training session, as used by `nova run --protocol`."""
    phases = []
    if baseline_sec > 0:
        phases.append(Phase("Baseline", "baseline", baseline_sec / 60.0))
    ref = protocol.name if PROTOCOLS.get(protocol.name) == protocol else protocol.to_spec()
    phases.append(Phase("Training", "training", training_sec / 60.0, ref, reward_rate))
    return Program(
        id=f"single-{protocol.name}",
        name=f"{protocol.name} training",
        category="custom",
        target=protocol.description,
        summary=protocol.description,
        phases=tuple(phases),
    )


# ----------------------------------------------------------------------------
# Built-in library
# ----------------------------------------------------------------------------

ARRIVE_OPEN = "Sit comfortably with a soft gaze. Let your breath slow while NOVA learns your resting levels."
ARRIVE_CLOSED = "Close your eyes and settle. Let your breath slow while NOVA learns your resting levels."
BREATHE = "Pause. Follow the swell of the sound: in as it rises, out as it falls."
INTEGRATE = "Let the effort go. Notice how you feel now, and stay with it as the sound fades."


def _arc(
    blocks: list[tuple[str, str, float, float, str]],
    *,
    eyes: str = "open",
    warmup: tuple[str, float, str],
    arrive: float = 2.0,
    interlude: float = 1.0,
    integrate: float = 2.5,
    binaural: float | None = None,
) -> tuple[Phase, ...]:
    """Build the standard arc. Each block is (name, protocol, minutes, reward_rate, guidance)."""
    w_proto, w_min, w_guide = warmup
    phases = [
        Phase("Arrive", "baseline", arrive, guidance=ARRIVE_CLOSED if eyes == "closed" else ARRIVE_OPEN, binaural=binaural),
        Phase("Warm-up", "training", w_min, w_proto, reward_rate=0.75, guidance=w_guide, binaural=binaural),
    ]
    for i, (name, proto, minutes, rate, guide) in enumerate(blocks):
        if i:
            phases.append(Phase("Breathe", "rest", interlude, guidance=BREATHE))
        phases.append(Phase(name, "training", minutes, proto, reward_rate=rate, guidance=guide, binaural=binaural))
    phases.append(Phase("Integrate", "cooldown", integrate, guidance=INTEGRATE))
    return tuple(phases)


BUILTIN: tuple[Program, ...] = (
    Program(
        id="focus-adhd",
        name="Focus Builder",
        category="condition",
        target="ADHD and attention",
        summary="Alternates lowering the theta/beta ratio at the top of the head with raising "
        "SMR over the right motor strip: the two most studied attention protocols.",
        soundscape="aurora",
        eyes="open",
        evidence="Most studied",
        evidence_note="Theta/beta and SMR training are the most researched neurofeedback protocols. "
        "Meta-analyses report improvements in parent-rated ADHD symptoms; effects are smaller in "
        "blinded ratings. Courses usually run 30-40 sessions.",
        sites=("Cz", "C4"),
        phases=_arc(
            [
                ("Steady attention", "theta-beta", 5, 0.6, "Keep a calm, steady attention on the sound. Brighter means you've found it."),
                ("Still body, clear mind", "smr", 5, 0.6, "Stay still and relaxed, eyes soft. Let the sound open up around you."),
                ("Sustain", "theta-beta", 5, 0.55, "Hold that steady attention a little longer. No need to strain."),
            ],
            warmup=("smr", 3, "Relax your shoulders and jaw. The sound brightens when you're calm and alert."),
        ),
    ),
    Program(
        id="mood-depression",
        name="Mood Lift",
        category="condition",
        target="Low mood and depression",
        summary="Trains frontal alpha asymmetry toward relatively more left-frontal activity, "
        "a pattern associated with approach motivation and positive mood.",
        soundscape="dawn",
        eyes="open",
        evidence="Emerging",
        evidence_note="Frontal alpha asymmetry training has small studies with encouraging results, "
        "but evidence is preliminary. Use alongside care from a clinician, not instead of it.",
        sites=("F3", "F4"),
        caution="If you are in crisis or thinking about harming yourself, contact local emergency "
        "services or a crisis line (call or text 988 in the US).",
        phases=_arc(
            [
                ("Brighten", "alpha-asymmetry", 5, 0.6, "Think of something you're looking forward to. Let the sound warm up."),
                ("Reach toward", "alpha-asymmetry", 5, 0.6, "Picture a place or person that feels good to move toward."),
                ("Carry it", "alpha-asymmetry", 5, 0.55, "Keep that warmth gently in mind as the sound rises."),
            ],
            warmup=("alpha-asymmetry", 3, "Settle into a light, curious mood. Rewards come easily at first."),
        ),
    ),
    Program(
        id="calm-anxiety",
        name="Calm Waters",
        category="condition",
        target="Anxiety",
        summary="Raises posterior alpha, the rhythm of relaxed wakefulness, with eyes closed, "
        "then deepens into slower theta-alpha calm.",
        soundscape="ocean",
        eyes="closed",
        evidence="Limited",
        evidence_note="Alpha up-training has small studies reporting reduced anxiety. Evidence is "
        "limited but the training is gentle and low-risk.",
        sites=("Pz", "O1", "O2"),
        phases=_arc(
            [
                ("Open the breath", "alpha", 5, 0.6, "Eyes closed. Breathe out slowly; let the waves carry the sound."),
                ("Soften", "alpha", 5, 0.6, "Let your forehead, jaw and hands go soft."),
                ("Drift", "alpha-theta", 4, 0.55, "Let thoughts drift past like clouds. Nothing to do."),
            ],
            eyes="closed",
            warmup=("alpha", 3, "Eyes closed. The sound grows fuller as you relax."),
        ),
    ),
    Program(
        id="stress-recovery",
        name="Unwind",
        category="condition",
        target="Stress",
        summary="Builds calm, still alertness with SMR training, then moves into relaxed alpha "
        "while discouraging high-beta tension.",
        soundscape="rain",
        eyes="open",
        evidence="Limited",
        evidence_note="SMR and alpha training are widely used for stress regulation. Research is "
        "limited, and results vary between people.",
        sites=("C4", "Pz"),
        phases=_arc(
            [
                ("Ground", "smr", 5, 0.6, "Feel your feet and the chair. Stay still; let the rain brighten."),
                ("Release", "alpha", 5, 0.6, "Let your eyes close if you like. Unclench, exhale, let go."),
                ("Rest in it", "alpha", 4, 0.55, "Stay with the calm. The sound will hold you there."),
            ],
            warmup=("smr", 3, "Drop your shoulders. Rewards come often as you settle."),
        ),
    ),
    Program(
        id="energy-fatigue",
        name="Recharge",
        category="condition",
        target="Fatigue and low energy",
        summary="Raises beta at the top of the head for alert engagement while holding down slow theta.",
        soundscape="celestial",
        eyes="open",
        evidence="Limited",
        evidence_note="Beta up-training for alertness has limited evidence. Stop if you feel tense or "
        "get a headache, and avoid it late in the evening.",
        sites=("Cz", "Fz"),
        phases=_arc(
            [
                ("Wake up", "beta", 4, 0.6, "Sit up tall, eyes open. Bring bright, interested attention to the sound."),
                ("Clear the fog", "theta-beta", 4, 0.6, "Keep your mind gently busy: notice each new bell."),
                ("Bright focus", "beta", 4, 0.55, "Hold that alert, easy energy."),
            ],
            warmup=("beta", 2, "Take three energizing breaths, then watch the sound come alive."),
            integrate=2.0,
        ),
    ),
    Program(
        id="deep-rest",
        name="Deep Rest",
        category="state",
        target="Deep relaxation",
        summary="Classic eyes-closed alpha-theta training: drift toward the edge of sleep while staying aware.",
        soundscape="nebula",
        eyes="closed",
        evidence="Limited",
        evidence_note="Alpha-theta training is used for deep relaxation and in addiction and trauma "
        "programs. Best done lying back, eyes closed, when you don't need to be alert afterward.",
        sites=("Pz", "O1", "O2"),
        phases=_arc(
            [
                ("Descend", "alpha", 5, 0.6, "Eyes closed. Let your body grow heavy."),
                ("Twilight", "alpha-theta", 6, 0.6, "Let images come and go. Stay just awake enough to hear the bells."),
                ("Deep", "alpha-theta", 6, 0.55, "Float. There's nothing to hold on to."),
            ],
            eyes="closed",
            warmup=("alpha", 3, "Eyes closed. Breathe slowly; the sound deepens as you relax."),
            integrate=3.0,
        ),
    ),
    Program(
        id="peak-focus",
        name="Peak Focus",
        category="state",
        target="Calm, sharp concentration",
        summary="SMR for a quiet body and steady mind, then beta for engaged concentration.",
        soundscape="celestial",
        eyes="open",
        evidence="Limited",
        evidence_note="SMR and beta training for performance have mixed evidence in healthy adults.",
        sites=("C4", "Cz"),
        phases=_arc(
            [
                ("Quiet body", "smr", 5, 0.6, "Stay physically still. Let your attention rest on the sound."),
                ("Engage", "beta", 5, 0.6, "Bring curious, focused attention to each detail of the sound."),
                ("Flow", "smr", 5, 0.55, "Relaxed and alert at once. Let it feel easy."),
            ],
            warmup=("smr", 3, "Settle your body. The sound brightens as you become still."),
        ),
    ),
)

BUILTIN_BY_ID = {p.id: p for p in BUILTIN}


# ----------------------------------------------------------------------------
# Custom programs
# ----------------------------------------------------------------------------


def build_custom(
    name: str,
    protocol: str | dict,
    *,
    minutes: float = 20.0,
    blocks: int = 3,
    reward_rate: float = 0.6,
    soundscape: str = "aurora",
    eyes: str = "open",
    binaural: float | None = None,
    guidance: str = "",
) -> Program:
    """A custom program on the standard arc, targeting any band or ratio."""
    if blocks < 1:
        raise ValueError("at least one training block is needed")
    proto = resolve_protocol(protocol)
    arrive, integrate, warm = 2.0, 2.0, 2.0
    interludes = 1.0 * (blocks - 1)
    block_min = max(1.0, (minutes - arrive - integrate - warm - interludes) / blocks)
    guide = guidance or "Relax and let the sound guide you. It brightens when you're on target."
    phases = _arc(
        [(f"Block {i + 1}", protocol, round(block_min, 2), reward_rate, guide) for i in range(blocks)],
        eyes=eyes,
        warmup=(protocol, warm, "Rewards come easily at first while you find the target."),
        arrive=arrive,
        integrate=integrate,
        binaural=binaural,
    )
    pid = "custom-" + (re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "program")
    return Program(
        id=pid,
        name=name,
        category="custom",
        target=f"{proto.metric_label} {'up' if proto.direction > 0 else 'down'}",
        summary=proto.description,
        phases=phases,
        soundscape=soundscape,
        eyes=eyes,
        sites=proto.sites,
    )


def load_custom(directory: Path = CUSTOM_DIR) -> list[Program]:
    programs = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            try:
                programs.append(Program.from_dict(json.loads(path.read_text())))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue  # skip files that aren't valid programs
    return programs


def save_custom(program: Program, directory: Path = CUSTOM_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{program.id}.json"
    path.write_text(json.dumps(program.to_dict(), indent=2))
    return path


def all_programs(directory: Path = CUSTOM_DIR) -> list[Program]:
    return list(BUILTIN) + load_custom(directory)


def get_program(ref: str, directory: Path = CUSTOM_DIR) -> Program:
    """Look up a program by id, or load one from a .json path."""
    path = Path(ref)
    if path.suffix == ".json" and path.exists():
        return Program.from_dict(json.loads(path.read_text()))
    for program in all_programs(directory):
        if program.id == ref:
            return program
    raise ValueError(f"unknown program {ref!r}; run 'nova programs' to list them")


def export_json(programs: list[Program] | None = None) -> str:
    """Built-in programs and soundscapes as JSON, embedded in the console for offline preview."""
    data = {"programs": [p.to_dict() for p in (programs or BUILTIN)], "soundscapes": SOUNDSCAPES}
    return json.dumps(data, separators=(",", ":"))


_EMBED_START = '<script type="application/json" id="nova-programs">'
_EMBED_END = "</script>"


def sync_dashboard(path: Path | None = None) -> bool:
    """Rewrite the console's embedded program library. Returns True if the file changed."""
    path = path or Path(__file__).parent / "dashboard" / "index.html"
    html = path.read_text(encoding="utf-8")
    start = html.index(_EMBED_START) + len(_EMBED_START)
    end = html.index(_EMBED_END, start)
    updated = html[:start] + export_json() + html[end:]
    if updated != html:
        path.write_text(updated, encoding="utf-8")
    return updated != html


if __name__ == "__main__":  # python -m nova.programs: refresh the console's embedded library
    print("updated" if sync_dashboard() else "already up to date")
