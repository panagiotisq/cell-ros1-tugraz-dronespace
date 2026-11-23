"""Result status values, labels and colours shared by the CELL side and the ROS 1 bridge.
Stdlib only.

CELL is map-relative: every tick it refines a guess, it never produces an absolute pose
from nothing (see DEPLOY_CELL_DRONESPACE.md sec 6-7). So "success" here means "the
correction was accepted"; on anything else the estimate HOLDS at the guess, it is not
dropped like a GLACE result would be.
"""

VALID = "valid"            # PnP correction accepted (enough inliers, step within bound)
HELD = "held"               # PnP failed / too few inliers / too large a step -> estimate = guess
SKIPPED = "skipped"         # inference not run (too few events, or no guess yet)
SYNTHETIC = "synthetic"     # test pose, never a CELL prediction
GUESS = "guess"              # the live mocap-derived guess itself (GuessForwarder's own marker)

CELL_STATUSES = (VALID, HELD, SKIPPED)

STATUS_COLORS = {VALID: (0.1, 0.9, 0.2), HELD: (1.0, 0.55, 0.0), SKIPPED: (0.6, 0.6, 0.6),
                 SYNTHETIC: (1.0, 0.0, 1.0), GUESS: (0.2, 0.5, 1.0)}
STATUS_LABELS = {VALID: "VALID (correction accepted)", HELD: "HELD (guess only, correction rejected)",
                 SKIPPED: "INFERENCE SKIPPED", SYNTHETIC: "SYNTHETIC TEST POSE", GUESS: "MOCAP GUESS"}
DISCONNECTED_LABEL = "NO INPUT - CELL PROCESS NOT CONNECTED"
