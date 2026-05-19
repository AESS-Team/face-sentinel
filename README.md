# Face Sentinel

Face Sentinel runs on a Raspberry Pi 3 B+ with a Raspberry Pi Camera Rev 1.3. It keeps the original face-recognition event flow, and also has a separate human-presence trigger for the voice gatekeeper.

## Current Network Setup

The Pi config currently posts events to:

```text
http://10.43.250.50:8765/event
```

If this PC gets a different IP, edit `pi_agent/config.json` and change `receiver_url`.

## PC Receiver

Run this on Windows from PowerShell:

```powershell
cd C:\antigravity-projects\face-sentinel
python -m pc_receiver.server
```

Keep that terminal open. Incoming unknown-face events are saved to:

```text
C:\antigravity-projects\face-sentinel\pc_receiver\received
```

If Windows Firewall asks, allow Python on the private/local network.

## Copy Project To The Raspberry Pi

From PowerShell, once SSH works:

```powershell
scp -r C:\antigravity-projects\face-sentinel aess@aess.local:/home/aess/
```

Then SSH into the Pi:

```powershell
ssh aess@aess.local
```

## Pi Setup

On the Raspberry Pi:

```bash
sudo apt update
sudo apt install -y python3-opencv python3-picamera2 alsa-utils python3-pip
cd /home/aess/face-sentinel
python3 -m pip install --user -r pi_agent/requirements-pi.txt
python3 -m pi_agent.download_models
```

Test the camera separately if needed:

```bash
rpicam-still -o test.jpg --timeout 2000
```

If `rpicam-still` is missing on your installed OS, try:

```bash
libcamera-still -o test.jpg --timeout 2000
```

## Enroll Allowed Faces

Create one folder per allowed person:

```bash
mkdir -p pi_agent/known_faces/alice
mkdir -p pi_agent/known_faces/bob
```

Put several clear JPG/PNG photos of each person in their folder. Use different angles and lighting if possible. Then build the embedding database:

```bash
python3 -m pi_agent.enroll
```

This creates:

```text
pi_agent/known_faces/embeddings.json
```

## Run Detection

First dry-run locally on the Pi, without sending HTTP:

```bash
python3 -m pi_agent.run --dry-run --once
```

To debug the complete camera/recognition/HTTP path, run monitor mode. This sends labelled frames to the PC at 2 FPS:

```bash
python3 -m pi_agent.run --monitor --monitor-fps 2
```

Monitor frames are labelled as:

```text
no_face
known
unknown
```

The PC receiver stores them under:

```text
pc_receiver/received/monitor
```

Then run normally:

```bash
python3 -m pi_agent.run
```

When an unknown face appears, the Pi saves a local event under:

```text
pi_agent/events
```

and sends the same event to the PC receiver.

## Voice Gatekeeper

Any visible person can trigger the AESS voice gatekeeper, even if the camera does not have enough face data to identify them. The presence detector lives under `pi_agent/presence/` and looks for frontal/profile faces, upper bodies, full bodies, and OpenCV HOG person detections. The old face-recognition flow is still present for labelled monitor frames and unknown-face events.

When presence is detected, the gatekeeper asks "Hey! Quien eres?", records the answer through ALSA, transcribes it with Gemini, asks short robotics questions, and stops after the visitor gives enough correct technical answers.

The API key lives in `.env` as `GOOGLE_AI_STUDIO_API_KEY`; `.env` is ignored by git. Copy that file to the Pi together with the project, or set the same environment variable before running.

Example questions it can generate:

- Que es PWM y por que sirve para controlar la velocidad de un motor?
- Como funciona un puente H en un robot con motores DC?
- Para que usarias un sensor de ultrasonidos en un robot movil?
- Que diferencia hay entre un servo y un motor DC con encoder?
- Que hace un controlador PID?
- En un minisumo de AESSBot, que sensores ayudan a no salirse del dohyo?
- Que es I2C y por que puede ir bien con sensores en Raspberry Pi?

The current default models in `pi_agent/config.json` are:

```text
gemini_model: gemini-3.1-flash-lite
tts_model:    gemini-3.1-flash-tts-preview
```

`gemini-3.1-flash-lite` is used for audio transcription and answer evaluation because it is available to this project, has free input/output tokens, and is optimized for high-volume, low-latency tasks. `gemini-3-flash-preview` is available too, but is preview-only and was slower in a minimal local API check.

## Tuning

Edit `pi_agent/config.json`:

- `match_threshold`: higher means stricter matching. If known people are rejected, lower it slightly. If strangers are accepted, raise it.
- `unknown_cooldown_seconds`: minimum seconds between unknown-face HTTP events.
- `gatekeeper.questions_to_pass`: number of robotics answers needed before the guard shuts up.
- `gatekeeper.silence_after_pass_seconds`: seconds to stay quiet after someone passes.
- `gatekeeper.record_seconds`: seconds recorded for each spoken answer.
- `presence.enabled`: enables the person-present trigger for the voice gatekeeper.
- `presence.cooldown_seconds`: minimum seconds between gatekeeper attempts while someone remains visible.
- `camera.width` / `camera.height`: keep `640x480` on Raspberry Pi 3 B+ unless performance is poor.

## Files

```text
pc_receiver/server.py       PC HTTP receiver
pi_agent/config.json        Pi configuration
pi_agent/download_models.py Downloads YuNet and SFace ONNX models
pi_agent/enroll.py          Builds known-face embeddings
pi_agent/gatekeeper.py      Gemini STT, robotics challenge, and TTS
pi_agent/presence/          Human-presence detector for the gatekeeper
pi_agent/run.py             Live camera loop
tests/                      Unit tests for non-camera logic
```

## Test On The PC

```powershell
cd C:\antigravity-projects\face-sentinel
python -m unittest discover -v
```
