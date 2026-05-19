import unittest
import tempfile
from pathlib import Path

from pi_agent.gatekeeper import (
    GatekeeperReply,
    GatekeeperSession,
    build_tts_prompt,
    build_gatekeeper_prompt,
    parse_gatekeeper_reply,
)


class FakeAudio:
    def __init__(self):
        self.recorded_paths = []
        self.played_paths = []

    def record(self, output_path, seconds):
        self.recorded_paths.append((Path(output_path), seconds))

    def play(self, wav_path):
        self.played_paths.append(Path(wav_path))


class FakeAssistant:
    def __init__(self, replies):
        self.replies = list(replies)
        self.transcribed_paths = []
        self.spoken_texts = []

    def transcribe(self, wav_path):
        self.transcribed_paths.append(Path(wav_path))
        return "un puente H invierte la polaridad del motor"

    def next_reply(self, history, transcript):
        return self.replies.pop(0)

    def synthesize(self, text, output_path):
        self.spoken_texts.append(text)
        Path(output_path).write_bytes(b"fake-wav")


class GatekeeperTests(unittest.TestCase):
    def test_parse_gatekeeper_reply_extracts_model_json(self):
        reply = parse_gatekeeper_reply(
            'Claro. {"spoken_reply": "Correcto, ya me callo.", "passed": true, "score": 2}'
        )

        self.assertTrue(reply.passed)
        self.assertEqual(reply.spoken_reply, "Correcto, ya me callo.")
        self.assertEqual(reply.score, 2)

    def test_build_gatekeeper_prompt_contains_aess_context_and_schema(self):
        prompt = build_gatekeeper_prompt(
            history=[{"role": "assistant", "text": "Hey! Quien eres?"}],
            transcript="Soy visitante y se que PWM controla potencia.",
            questions_to_pass=2,
        )

        self.assertIn("molesto duende de AESS Estudiants", prompt)
        self.assertIn("ETSETB-UPC", prompt)
        self.assertIn("AESSBot", prompt)
        self.assertIn("spoken_reply", prompt)
        self.assertIn("Soy visitante", prompt)

    def test_build_gatekeeper_prompt_includes_robotics_question_examples(self):
        prompt = build_gatekeeper_prompt(
            history=[{"role": "assistant", "text": "Hey! Quien eres?"}],
            transcript="Soy de AESS.",
            questions_to_pass=2,
        )

        self.assertIn("PWM", prompt)
        self.assertIn("puente H", prompt)
        self.assertIn("sensor de ultrasonidos", prompt)
        self.assertIn("minisumo", prompt)

    def test_build_tts_prompt_requests_high_pitch_funny_voice(self):
        prompt = build_tts_prompt("Correcto, ya me callo.")

        self.assertIn("voz muy aguda", prompt)
        self.assertIn("duende", prompt)
        self.assertIn("Correcto, ya me callo.", prompt)

    def test_session_stops_when_gatekeeper_passes(self):
        audio = FakeAudio()
        assistant = FakeAssistant(
            [
                GatekeeperReply(
                    spoken_reply="Correcto. Has sobrevivido a mis preguntas, ya me callo.",
                    passed=True,
                    score=2,
                )
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            session = GatekeeperSession(
                assistant=assistant,
                audio=audio,
                audio_dir=Path(temp_dir),
                record_seconds=2.0,
                max_rounds=3,
            )

            passed = session.run()

        self.assertTrue(passed)
        self.assertEqual(len(audio.recorded_paths), 1)
        self.assertEqual(len(audio.played_paths), 2)
        self.assertEqual(assistant.spoken_texts[0], "Hey! Quien eres?")
        self.assertIn("ya me callo", assistant.spoken_texts[-1])


if __name__ == "__main__":
    unittest.main()
