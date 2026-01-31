# RoastCoach - Exercise Agnostic Motion Feedback

RoastCoach is a computer vision coaching tool that evaluates **any** movement without needing exercise labels (no “squat”, “deadlift”, etc.).
A coach uploads a reference video, RoastCoach extracts joint angles over time, builds a **reference envelope** (mean ± tolerance band), and then compares a user’s attempt to detect **where, when, and how** they deviate.
Feedback is grounded in measured kinematics (angles + phases), with confidence/safety gating.

## Core idea
**Movement = joint-angle trajectories through time.**
Instead of recognizing an exercise, we compare motion “fingerprints”:
- Pose → joint angles
- Rep segmentation + temporal normalization
- Reference envelope from coach reps
- Deviation detection (magnitude, phase, persistence)
- Optional constrained language layer for short coaching cues
