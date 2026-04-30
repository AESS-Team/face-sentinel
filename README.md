# Face Sentinel

Face Sentinel runs face recognition on a Raspberry Pi 3 B+ with a Raspberry Pi Camera Rev 1.3. It treats enrolled people as allowed and sends an HTTP event to this PC when it sees a face that does not match the enrolled set.

## Current Network Setup

The Pi config currently posts events to:

```text
http://10.192.175.133:8765/event
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
sudo apt install -y python3-opencv python3-picamera2
cd /home/aess/face-sentinel
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

Then run normally:

```bash
python3 -m pi_agent.run
```

When an unknown face appears, the Pi saves a local event under:

```text
pi_agent/events
```

and sends the same event to the PC receiver.

## Tuning

Edit `pi_agent/config.json`:

- `match_threshold`: higher means stricter matching. If known people are rejected, lower it slightly. If strangers are accepted, raise it.
- `unknown_cooldown_seconds`: minimum seconds between unknown-face HTTP events.
- `camera.width` / `camera.height`: keep `640x480` on Raspberry Pi 3 B+ unless performance is poor.

## Files

```text
pc_receiver/server.py       PC HTTP receiver
pi_agent/config.json        Pi configuration
pi_agent/download_models.py Downloads YuNet and SFace ONNX models
pi_agent/enroll.py          Builds known-face embeddings
pi_agent/run.py             Live camera loop
tests/                      Unit tests for non-camera logic
```

## Test On The PC

```powershell
cd C:\antigravity-projects\face-sentinel
python -m unittest discover -v
```
