from __future__ import annotations

import json
import os
import re
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import AgentConfig


DEFAULT_CHALLENGE = "Hey! Quien eres?"
TTS_STYLE = "Di esto con voz muy aguda, chillona y graciosa, como un duende pequeñito de dibujos, con tono burlon pero entendible"

QUESTION_EXAMPLES = [
    "Que es PWM y por que sirve para controlar la velocidad de un motor?",
    "Como funciona un puente H y que problema resuelve en un robot con motores DC?",
    "Para que usarias un sensor de ultrasonidos en un robot movil?",
    "Que diferencia hay entre un servo y un motor DC con encoder?",
    "Que hace un controlador PID y que significan P, I y D?",
    "En un minisumo de AESSBot, que sensores ayudan a no salirse del dohyo?",
    "Que es I2C y por que puede ir bien para conectar sensores a una Raspberry Pi?",
    "Por que conviene alimentar motores y logica con lineas separadas o bien reguladas?",
    "Que es un pull-up en una entrada digital?",
    "Como detectarias una linea negra con sensores infrarrojos?",
]


SYSTEM_PROMPT = """Eres un molesto duende de AESS Estudiants, la asociacion de estudiantes de la ETSETB-UPC dedicada a la robotica y las nuevas tecnologias.
AESS promueve la robotica y la tecnologia con cursos, talleres, conferencias, concursos y AESSBot, su competicion de robotica con minisumo y retos de hardware y software.
Tu trabajo es vigilar la entrada: gente que no sea de AESS no puede pasar sin resolver tus preguntas.
Hablas en castellano con un punto pesado, burlon y teatral, pero sin insultos graves, sin amenazas y sin pedir datos personales sensibles.
Primero pregunta quien es. Despues haz preguntas cortas de robotica, Arduino, Raspberry Pi, sensores, actuadores, control, electronica basica o competiciones tipo minisumo.
Da por buena una respuesta si demuestra conocimiento tecnico razonable aunque no sea perfecta. Si supera las preguntas requeridas, di que te callas y deja pasar.
Responde siempre en JSON valido con las claves: spoken_reply, passed, score.
spoken_reply es el texto exacto que se leera por TTS. passed es booleano. score es el numero de respuestas tecnicas correctas acumuladas."""


@dataclass(frozen=True)
class GatekeeperReply:
    spoken_reply: str
    passed: bool
    score: int


class VoiceAssistant(Protocol):
    def transcribe(self, wav_path: Path) -> str:
        raise NotImplementedError

    def next_reply(self, history: list[dict[str, str]], transcript: str) -> GatekeeperReply:
        raise NotImplementedError

    def synthesize(self, text: str, output_path: Path) -> None:
        raise NotImplementedError


class AudioIO(Protocol):
    def record(self, output_path: Path, seconds: float) -> None:
        raise NotImplementedError

    def play(self, wav_path: Path) -> None:
        raise NotImplementedError


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def api_key_from_environment(env_path: Path | None = None) -> str:
    if env_path is not None:
        load_env_file(env_path)
    for name in ("GOOGLE_AI_STUDIO_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError("Missing GOOGLE_AI_STUDIO_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY")


def parse_gatekeeper_reply(raw_text: str) -> GatekeeperReply:
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if match is None:
        return GatekeeperReply(spoken_reply=raw_text.strip(), passed=False, score=0)

    data = json.loads(match.group(0))
    return GatekeeperReply(
        spoken_reply=str(data.get("spoken_reply", "")).strip() or "No te he entendido. Repite, humano sospechoso.",
        passed=bool(data.get("passed", False)),
        score=int(data.get("score", 0)),
    )


def build_gatekeeper_prompt(
    history: list[dict[str, str]],
    transcript: str,
    questions_to_pass: int,
) -> str:
    question_examples = "\n".join(f"- {question}" for question in QUESTION_EXAMPLES)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Puedes generar preguntas parecidas a estas, variando dificultad y tema:\n"
        f"{question_examples}\n\n"
        f"Necesita {questions_to_pass} respuestas tecnicas correctas para que te calles.\n"
        "Historial de la conversacion, en orden:\n"
        f"{json.dumps(history, ensure_ascii=False)}\n\n"
        f"Ultima respuesta transcrita del visitante: {transcript}\n\n"
        "Evalua la ultima respuesta, actualiza la puntuacion y devuelve solo JSON."
    )


def build_tts_prompt(text: str) -> str:
    return f"{TTS_STYLE}: {text}"


def write_wave_file(filename: Path, pcm: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(filename), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm)


class AlsaAudioIO:
    def record(self, output_path: Path, seconds: float) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["arecord", "-q", "-D", "plughw:1,0", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(int(seconds)), str(output_path)],
            check=True,
        )

    def play(self, wav_path: Path) -> None:
        subprocess.run(["aplay", "-q", str(wav_path)], check=True)


class GeminiVoiceAssistant:
    def __init__(
        self,
        api_key: str,
        gemini_model: str,
        tts_model: str,
        voice_name: str,
        questions_to_pass: int,
    ):
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install google-genai to use the voice gatekeeper") from exc

        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.gemini_model = gemini_model
        self.tts_model = tts_model
        self.voice_name = voice_name
        self.questions_to_pass = questions_to_pass

    def transcribe(self, wav_path: Path) -> str:
        audio_bytes = wav_path.read_bytes()
        response = self.client.models.generate_content(
            model=self.gemini_model,
            contents=[
                "Transcribe exactamente el habla de este audio. Devuelve solo la transcripcion.",
                self.types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            ],
        )
        return response.text.strip()

    def next_reply(self, history: list[dict[str, str]], transcript: str) -> GatekeeperReply:
        prompt = build_gatekeeper_prompt(history, transcript, self.questions_to_pass)
        response = self.client.models.generate_content(model=self.gemini_model, contents=prompt)
        return parse_gatekeeper_reply(response.text)

    def synthesize(self, text: str, output_path: Path) -> None:
        response = self.client.models.generate_content(
            model=self.tts_model,
            contents=build_tts_prompt(text),
            config=self.types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=self.types.SpeechConfig(
                    voice_config=self.types.VoiceConfig(
                        prebuilt_voice_config=self.types.PrebuiltVoiceConfig(voice_name=self.voice_name)
                    )
                ),
            ),
        )
        data = response.candidates[0].content.parts[0].inline_data.data
        write_wave_file(output_path, data)


class GatekeeperSession:
    def __init__(
        self,
        assistant: VoiceAssistant,
        audio: AudioIO,
        audio_dir: Path,
        record_seconds: float,
        max_rounds: int,
    ):
        self.assistant = assistant
        self.audio = audio
        self.audio_dir = audio_dir
        self.record_seconds = record_seconds
        self.max_rounds = max_rounds

    def _speak(self, text: str, stem: str) -> None:
        output_path = self.audio_dir / f"{stem}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.assistant.synthesize(text, output_path)
        self.audio.play(output_path)

    def run(self) -> bool:
        timestamp = int(time.time())
        history = [{"role": "assistant", "text": DEFAULT_CHALLENGE}]
        self._speak(DEFAULT_CHALLENGE, f"{timestamp}_challenge")

        for round_index in range(self.max_rounds):
            answer_path = self.audio_dir / f"{timestamp}_answer_{round_index + 1}.wav"
            self.audio.record(answer_path, self.record_seconds)
            transcript = self.assistant.transcribe(answer_path)
            history.append({"role": "user", "text": transcript})

            reply = self.assistant.next_reply(history, transcript)
            history.append({"role": "assistant", "text": reply.spoken_reply})
            self._speak(reply.spoken_reply, f"{timestamp}_reply_{round_index + 1}")

            if reply.passed:
                return True

        return False


def build_gatekeeper_session(config: AgentConfig) -> GatekeeperSession:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    assistant = GeminiVoiceAssistant(
        api_key=api_key_from_environment(env_path),
        gemini_model=config.gatekeeper_gemini_model,
        tts_model=config.gatekeeper_tts_model,
        voice_name=config.gatekeeper_voice_name,
        questions_to_pass=config.gatekeeper_questions_to_pass,
    )
    return GatekeeperSession(
        assistant=assistant,
        audio=AlsaAudioIO(),
        audio_dir=config.gatekeeper_audio_dir,
        record_seconds=config.gatekeeper_record_seconds,
        max_rounds=config.gatekeeper_max_rounds,
    )
