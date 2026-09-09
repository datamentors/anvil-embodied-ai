[← Back to README](../README.md)

# João — where the gripper work stands, and what I would do next

**From Eliano, 2026-09-09.** I went through the workstation this evening: your code, the
profiles, and every monitor CSV in `inference_Gr00t/monitor_output/archive/`. This is what I
think happened, what the data actually says, and what I would do in the morning.

Short version: **your implementation is right, one of the three changes was never actually
run, and your own `_shorthorizon` profile is the most interesting result any of us has
produced so far.** I also had one recommendation wrong, and your data is what shows it.

---

## 1. What you built

Your version of the smoothing changes is better than what I wrote in the guide, in two places
that matter:

* You build the latch **before** the `type: none` early return. In my version the latch was
  created inside the ensembler branch, so `smoother.type: none` plus a latch would have
  silently done nothing. That was a real bug and you removed it.
* You `.copy()` the array before writing the latched value. `value_at` returns `base` itself
  when fewer than two chunks cover a tick, so writing in place would have mutated the queue's
  own array. I missed that.

You also added `bypass_indices` alongside the derived `passthrough_gripper` and unioned them,
which is a better interface than mine: explicit when you want it, derived from
`model_joint_order` when you do not.

So: no complaints about the code. The problem is what reached the robot.

---

## 2. What actually ran, and what did not

| Change | Status | Evidence |
|---|---|---|
| 1. Gripper bypassed from the ensembler | **Ran, and it helped** | `_ensemble_armsonly_live`, 826 s |
| 2. Gripper latch | **Never ran on the robot** | see below |
| 3. Widen the finger joint limits | **Not applied** | all five profiles still `[0.0, 0.05]` |

On change 2 the test is decisive. A latched channel can only ever emit two values,
`closed_target` and `open_target`. Across every run today the gripper channel takes between
743 and 6,177 distinct values:

```
run                                                  uniq(raw L)  uniq(raw R)
..._ensemble_armsonly_live (12:06)                          6177         6030
..._ensemble_live          (14:01)                          2041         1775
..._live                   (15:18)                          2034         1128
..._shorthorizon_live      (16:53)                          4890         3709
```

`inference_groot17_envelope2916_latch_live.yaml` was written at 14:02 and there is no monitor
archive for it. `.env.groot` still points `CONFIG_FILE` at `_ensemble_armsonly_live`. So the
latch has been built, unit-reasoned and replayed offline, but never armed.

**Change 1 did work, and it is worth recording.** Against the 09-08 baseline:

| | 09-08 ensemble, no bypass | 09-09 `_armsonly` |
|---|---:|---:|
| left transitions per 10 min | 257 | 25 |
| left half-closed band | 45 % | 20 % |

That is a real improvement. It is just not enough on its own: the median closed phase is still
4 ticks, about 130 ms.

---

## 3. Where I was wrong, and what your data shows instead

I told you to run guidance-on with no smoother first, and called it the honest control. You
ran it. It is the **worst** configuration of the day: the gripper never closed once.

I checked whether the robot was simply idle in those runs. It was not. The arm swept 1.55 rad
and travelled 69 to 85 rad of joint motion over 5 to 6 minutes, so it was working the task
while the gripper stayed inside a 7 mm band near fully open.

Putting all four profiles side by side, with Pi0.5 for reference:

| profile | guidance | smoother | trigger / horizon | overlap | median closed run | gripper range |
|---|---|---|---|---|---:|---:|
| `_live` (my "arm A") | on | none | 40 / 35 | 87 % | **0, never closes** | 0.007 m |
| `_ensemble_live` | off | ens 0.1, no bypass | 40 / 35 | 87 % | 2 ticks | 0.051 m |
| `_ensemble_armsonly` | off | ens 0.1 + bypass | 40 / 35 | 87 % | 4 ticks | 0.053 m |
| **`_shorthorizon`** | **on** | none | **8 / 8** | **20 %** | **20 ticks (0.67 s)** | 0.053 m |
| Pi0.5 09-05, which works | on | none | chunk 50, horizon 35 | 70 % | 29 ticks (1 s) | — |

**The lever is the overlap, not guidance on or off.** At 35 of 40 steps, roughly 87 % of every
new chunk is pinned to the chunk before it, so a channel that has to flip between two states
never gets the chance. Pi0.5 sits at 70 % and closes fine. Your `_shorthorizon` sits at 20 %
and gives the longest, most decisive closes we have measured on GR00T.

That also explains the 09-08 result you recorded in the changelog, where guidance ON removed
the dither and simultaneously stopped the gripper closing at all. Same mechanism, and you
measured it before I did. I read it as over-damping from the ensemble; it was mostly the
overlap.

---

## 4. The image has to be rebuilt before anything runs

All Docker images on the box are gone. `docker images` is empty, `docker system df` reports
zero images, and the disk went from 27 GB free to 312 GB free. The bash history has
`docker builder prune -f`, so the build cache is gone too and this will be a cold build.

This matters because `anvil-groot-inference` has **no registry prefix**:

```yaml
    image: anvil-groot-inference:${IMAGE_TAG:-groot17-008580}
    build:
      context: .
      dockerfile: Dockerfile
```

There is nowhere to pull it from and no saved tar on the machine. It only exists if it is
built locally. `.env.groot` still says `IMAGE_TAG=groot17-011000-gripbypass`, so building with
the env file as it stands recreates exactly the tag the profiles expect:

```bash
cd ~/anvil-embodied-ai/inference_Gr00t
docker compose -f docker-compose.groot.yml --env-file .env.groot build
```

Two things to watch, both of which have bitten us before:

* **Check nobody else is building first**, `ps -eo cmd | grep "docker compose.*build"`.
  Concurrent builds share BuildKit and you can end up attached to someone else's build and
  running their image.
* **Verify by content, not by timestamp.** `md5sum` a file inside the image against the host
  copy. BuildKit reuses `Created` timestamps and it has misled us before.

```bash
docker run --rm --entrypoint md5sum anvil-groot-inference:groot17-011000-gripbypass \
  /workspace/src/lerobot_groot_control/lerobot_groot_control/gripper_latch.py
md5sum ros2/src/lerobot_groot_control/lerobot_groot_control/gripper_latch.py
```

**Before you rebuild, please commit.** Everything you wrote today lives only on that box:
`inference_Gr00t/` is 79 MB of untracked files, including `gripper_latch.py`, the node
changes, four profiles and three scripts. Somebody is actively pruning that machine to free
disk. If you want, I will push it to
`review/training-inference-gripper-2026-09-08` for you.

---

## 5. What I would run next

Nobody has yet run the two things that look most promising **together**: the short horizon and
the latch. Your own replay tool already predicts the latch is worth arming. Replayed over the
real recorded runs:

| run, replayed | half-closed band | median closed run | transitions / 10 min |
|---|---:|---:|---:|
| `_armsonly`, dwell 1.0 s | 0.201 → **0.000** | 4 → **56 ticks (1.9 s)** | 23 → 12 |
| `_shorthorizon`, dwell 1.0 s | 0.113 → **0.000** | 17 → **30 ticks** | 204 → 150 |

Suggested order, all config-only once the image exists:

1. **`_shorthorizon` + latch.** Take `_shorthorizon_live`, add the `smoother.gripper_latch`
   block from `_latch_live`, keep `smoother.type: none`. The latch works without the
   ensembler, which is exactly why you moved its construction earlier.
2. **Sweep the horizon** with guidance on and no smoother: 8, 12, 20, 35. Four short runs.
   This turns "20 % overlap is better than 87 %" into a curve, and it is the single most
   informative hour available right now. Keep `queue_trigger_threshold` equal to the horizon
   so the comparison stays clean.
3. **Apply change 3** while you are in the profiles. The model is asking for −0.0047 and being
   clamped to 0.0, so the last 4 to 5 mm of the squeeze it does request is being thrown away.
4. Only then revisit the ensemble coefficient. With the gripper bypassed the ensembler is
   an arm-joint smoother, and it should be judged on arm jerk, not on grasps.

Score each run with grasp attempts, lifts and completed sorts. The four gripper numbers are
diagnostics for *why* an arm failed; they are not the result.

---

## 6. On retraining with a different AFO

I understand you want to change the training data and retrain, with a different AFO
lookahead. I think the instinct is right that the data is implicated. I would push back on the
sequencing and on which part of the data you change.

**First, the sequencing.** Retraining is days; the horizon sweep above is minutes and costs
nothing but robot time. More importantly, **they confound each other**. If you retrain and
then run the new checkpoint at trigger 40 / horizon 35, the gripper will be pinned by the
overlap exactly as it is now, and the honest conclusion would be "the new data did not help"
when the data was never given a chance to speak. Settle the inference lever first, then change
one thing in the data.

**Second, which part of the data.** Changing N alone does not fix the gripper, and it is worth
being precise about why.

The AFO rule is `action[t] = observation.state[t + N]`. At **any** N, the gripper label is
still a position the finger actually reached. During the demonstrations the finger stops on
the envelope at about 4 mm and stays there: the dataset's `q10` is 0.0041 m on the left finger
and 0.0038 m on the right. So the model is being taught to command the contact position, with
no squeeze margin, and a position-controlled gripper told to go where it already is exerts no
force.

What a larger N *does* change is the closing **transition**: during the frames where the
finger is travelling shut, `state[t+N]` is further shut than `state[t]`, so the command leads
the motion. That is a genuine effect and it may well improve the approach. But once the finger
is on the envelope, `state[t+N]` equals the same 4 mm for every N, so the hold still carries no
force. **A different N buys timing, not grip.**

If you are going to reconvert anyway, the higher-value changes, in order:

1. **Take the gripper label from the command topic**, keeping AFO for the 14 arm joints.
   `/follower_*_forward_position_controller/commands` is already configured in
   `openarm_bimanual_quest_afo.yaml`; AFO mode simply ignores it. Worth checking on one raw
   MCAP first that the commanded finger really does go deeper than the observed one during a
   grasp. If it does, this is the fix.
2. **If the command is not usable, binarise the gripper label**: closed below a threshold,
   open above it, with closed set to the mechanical minimum rather than the contact position.
   Less faithful to the demonstration, and still better than teaching contact-without-force.
3. **Per-channel N**, if you want to keep exploring the lookahead: a different lookahead for
   the gripper than for the arms. Same splice point as option 1 in `_finalize_afo_frame`, just
   indexing the pending deque at two different offsets. This is the version of "a different
   AFO" I would find most defensible, because it separates the timing lead you want on the
   arms from the label semantics you need on the gripper.

Whatever you pick, please change **one** thing against the current dataset, keep
`stratified_split_info.json` byte-identical so the runs stay comparable, and do not `--resume`
onto an existing dataset. Mixing two label conventions in one directory is the trap
`docs/data-conversion.md` already warns about.

One more thing worth knowing before you spend a week of GPU on it: the current live checkpoint
is step 11 000 of 88 000, about half an epoch. Some of what we are attributing to the data may
just be an undertrained model. Running a later checkpoint of the *existing* run is cheaper
than a new dataset, and it is a variable we have not controlled either.

---

## 7. Two traps in the evidence

Not criticism, just things that will mislead whoever reads the archive next.

* **Two pairs of archived runs are byte-identical**: 12:06 with 13:07, and 14:01 with 15:13,
  same size and same numbers to the last digit. `archive_monitor.sh` gzips whatever is in
  `inference_data.csv`, and the monitor only truncates that file when a container actually
  starts. So when a container fails to start, the next archive is a copy of the previous run
  under a new name and a new timestamp. Given the images were pruned, that is very likely what
  happened.
* One file is called `..._live.assumed.csv.gz`, which I read as you not being certain which
  profile was loaded. That instinct is correct and worth making structural: the archive name
  records `CONFIG_FILE` from the environment, not what the container actually parsed. The node
  prints its resolved settings at startup, so the reliable record is the startup block in the
  container log, saved next to the CSV.

---

## 8. If you only do four things

1. Commit `inference_Gr00t/` before the next prune takes it.
2. Rebuild the image, verify by `md5sum` and not by timestamp.
3. Run `_shorthorizon` with the latch on, then sweep the horizon 8 / 12 / 20 / 35.
4. Hold the reconvert until step 3 has an answer, then change the gripper label **source**,
   not only N.

The `_shorthorizon` profile was your idea and it is the best lead we have. I would chase that
before spending a week on data.
