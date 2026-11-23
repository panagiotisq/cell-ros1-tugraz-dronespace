import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

BAG = "/path/to/demo_walkaround.bag"
WALK_START = 1794434459.174358278
OUT_NPZ = "/path/to/output/corrected_poses.npz"

ROT_RATE_THRESH = 800.0    # deg/s -- catches only the confirmed-impossible range (1000-2168 deg/s seen)
POS_RATE_THRESH = 3.0      # m/s
MERGE_GAP = 30             # merge sub-glitches within ~0.3s of each other
ANCHOR_MARGIN = 8          # small safety buffer against boundary artifacts

bag = rosbag.Bag(BAG)
ts, pos, quat = [], [], []
for _, msg, _ in bag.read_messages(topics=["/mocap_node/evk4/pose"]):
    t = msg.header.stamp.to_sec()
    if t < WALK_START:
        continue
    p, q = msg.pose.position, msg.pose.orientation
    ts.append(t)
    pos.append([p.x, p.y, p.z])
    quat.append([q.x, q.y, q.z, q.w])
bag.close()
ts = np.array(ts); pos = np.array(pos); quat = np.array(quat)
n = len(ts)
print(f"loaded {n} poses")

rots = R.from_quat(quat)
dt = np.diff(ts)
dpos = np.linalg.norm(np.diff(pos, axis=0), axis=1)
drot = (rots[1:] * rots[:-1].inv()).magnitude() * 180 / np.pi

rot_rate = drot / np.maximum(dt, 1e-3)
pos_rate = dpos / np.maximum(dt, 1e-3)
bad_step = (rot_rate > ROT_RATE_THRESH) | (pos_rate > POS_RATE_THRESH)

# a sample is bad if either its incoming or outgoing step is bad
bad_sample = np.zeros(n, dtype=bool)
bad_sample[1:] |= bad_step
bad_sample[:-1] |= bad_step

bad_idx = np.where(bad_sample)[0]
print(f"flagged {len(bad_idx)} bad samples (raw)")

# group into contiguous runs, merging gaps <= MERGE_GAP
runs = []
if len(bad_idx) > 0:
    start = bad_idx[0]
    prev = bad_idx[0]
    for i in bad_idx[1:]:
        if i - prev > MERGE_GAP:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))

print(f"\n{len(runs)} glitch run(s):")
for s, e in runs:
    print(f"  samples [{s},{e}]  t_rel=[{ts[s]-ts[0]:.3f}, {ts[e]-ts[0]:.3f}]s  "
          f"({'EDGE -- no good sample on one side!' if s == 0 or e == n-1 else 'interior'})")

pos_fixed = pos.copy()
quat_fixed = quat.copy()

for s, e in runs:
    good_before = s - 1 - ANCHOR_MARGIN
    good_after = e + 1 + ANCHOR_MARGIN
    if good_before < 0 or good_after >= n:
        print(f"  WARNING: run [{s},{e}] touches an edge, cannot bridge -- leaving as-is")
        continue
    t_lo, t_hi = ts[good_before], ts[good_after]
    key_times = [t_lo, t_hi]
    key_rots = R.from_quat([quat[good_before], quat[good_after]])
    slerp = Slerp(key_times, key_rots)
    for i in range(good_before + 1, good_after):
        frac = (ts[i] - t_lo) / (t_hi - t_lo)
        pos_fixed[i] = (1 - frac) * pos[good_before] + frac * pos[good_after]
        quat_fixed[i] = slerp([ts[i]])[0].as_quat()

np.savez(OUT_NPZ, ts=ts, pos=pos_fixed, quat=quat_fixed)
print(f"\nsaved corrected trajectory: {OUT_NPZ}")

# sanity check: recompute rates on the corrected trajectory in the same window
rots2 = R.from_quat(quat_fixed)
drot2 = (rots2[1:] * rots2[:-1].inv()).magnitude() * 180 / np.pi
rot_rate2 = drot2 / np.maximum(dt, 1e-3)
print("post-fix max rot_rate in previously-bad runs:",
      max((rot_rate2[s:e].max() if e > s else 0) for s, e in runs))
