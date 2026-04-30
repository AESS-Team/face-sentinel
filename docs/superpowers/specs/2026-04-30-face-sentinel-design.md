# Face Sentinel Design

## Goal

Run face recognition on a Raspberry Pi 3 B+ with a Raspberry Pi Camera Rev 1.3. When a face is detected but does not match the enrolled allowed faces, the Pi stores a local event photo/log and sends an HTTP event to the user's PC.

## Architecture

The project has two sides:

- `pi_agent`: runs on the Raspberry Pi, captures camera frames, detects faces with OpenCV YuNet, compares identities with OpenCV SFace embeddings, writes local events, and posts unknown-face events to the PC.
- `pc_receiver`: runs on the Windows PC, receives HTTP JSON events from the Pi and stores metadata plus event photos.

The first version avoids GPIO activation. GPIO can be added later once detection behavior has been tuned.

## Data Flow

1. The Pi loads configuration from `pi_agent/config.json`.
2. The Pi loads OpenCV model files from `pi_agent/models`.
3. The user enrolls allowed identities by placing photos in `pi_agent/known_faces/<name>` and running `enroll.py`.
4. `enroll.py` creates an embedding database at `pi_agent/known_faces/embeddings.json`.
5. `run.py` captures frames, detects faces, extracts embeddings, and compares them against the known database.
6. For unknown faces, `run.py` saves an image and JSON log in `pi_agent/events`, then posts the same event to `pc_receiver`.
7. The PC receiver stores incoming events under `pc_receiver/received`.

## Recognition Strategy

Use OpenCV YuNet for face detection and OpenCV SFace for face embeddings/comparison. This fits the use case better than YOLO because the goal is not only "is there a face?" but "is this face one of the enrolled people?"

The matching threshold is configurable. Unknown detections are rate-limited with a cooldown to avoid flooding the PC while the same person stays in frame.

## Error Handling

The Pi keeps operating if HTTP delivery fails. It logs the error locally and retries naturally on future unknown detections. Missing model files produce a clear setup error that tells the user to run the model download script.

## Testing

Automated tests cover configuration loading, face-match decisions, event persistence, and PC receiver request handling. Camera/model integration remains manually verified on the Raspberry Pi because it depends on the CSI camera and installed OpenCV build.
