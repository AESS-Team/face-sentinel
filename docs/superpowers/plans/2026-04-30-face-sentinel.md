# Face Sentinel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Raspberry Pi face sentinel that recognizes enrolled people and sends unknown-face events to a PC over HTTP.

**Architecture:** A `pi_agent` package owns camera/model/event logic. A `pc_receiver` package owns the local HTTP receiver. Tests exercise pure logic and HTTP handling without camera hardware.

**Tech Stack:** Python 3, OpenCV, NumPy, stdlib `http.server`, stdlib `unittest`, optional `requests` on the Pi.

---

### Task 1: Project Skeleton

**Files:**
- Create: `README.md`
- Create: `pi_agent/__init__.py`
- Create: `pc_receiver/__init__.py`
- Create: `tests/__init__.py`

- [x] Create folders and package files.

### Task 2: Decision Logic

**Files:**
- Create: `tests/test_recognition.py`
- Create: `pi_agent/recognition.py`

- [x] Write failing tests for known, unknown, and empty database decisions.
- [x] Implement cosine similarity and threshold decisions.
- [x] Run tests.

### Task 3: Event Store and Sender

**Files:**
- Create: `tests/test_events.py`
- Create: `pi_agent/events.py`

- [x] Write failing tests for local JSON/JPEG storage and sender payload shape.
- [x] Implement event persistence and HTTP POST helper.
- [x] Run tests.

### Task 4: PC Receiver

**Files:**
- Create: `tests/test_pc_receiver.py`
- Create: `pc_receiver/server.py`

- [x] Write failing tests for receiving an event and saving metadata/photo.
- [x] Implement stdlib HTTP JSON receiver.
- [x] Run tests.

### Task 5: Pi CLI Scripts

**Files:**
- Create: `pi_agent/config.json`
- Create: `pi_agent/download_models.py`
- Create: `pi_agent/enroll.py`
- Create: `pi_agent/run.py`
- Create: `pi_agent/requirements-pi.txt`

- [x] Implement config, model download, enrollment, and runtime scripts.
- [x] Keep hardware-specific camera path isolated in `run.py`.

### Task 6: Documentation

**Files:**
- Modify: `README.md`

- [x] Document PC setup, Pi setup, enrollment, running, and troubleshooting.
