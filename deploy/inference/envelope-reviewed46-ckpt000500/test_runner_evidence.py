#!/usr/bin/env python3
"""Static fail-closed contract checks for the isolated deployment runner."""

import unittest
from pathlib import Path

RUNNER = Path(__file__).with_name("_run_mode.sh")


class RunnerEvidenceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = RUNNER.read_text(encoding="utf-8")

    def test_live_requires_two_part_confirmation_before_compose_start(self) -> None:
        env_confirmation = "RUN_CKPT000500_ON_REAL_ROBOT"
        typed_confirmation = "HOME AND RUN CKPT000500 LIVE"
        compose_up = 'up --no-build --detach "${services[@]}"'
        self.assertIn(env_confirmation, self.text)
        self.assertIn(typed_confirmation, self.text)
        self.assertLess(self.text.index(env_confirmation), self.text.index(compose_up))
        self.assertLess(self.text.index(typed_confirmation), self.text.index(compose_up))

    def test_live_homes_and_rechecks_authority_before_compose_start(self) -> None:
        confirmation = self.text.index("HOME AND RUN CKPT000500 LIVE")
        home_call = self.text.index("require_robot_home", confirmation)
        post_home_gate = self.text.index(
            'require_graph_contract 0 0 "after homing before live startup"',
            home_call,
        )
        compose_up = self.text.index('up --no-build --detach "${services[@]}"')
        self.assertLess(confirmation, home_call)
        self.assertLess(home_call, post_home_gate)
        self.assertLess(post_home_gate, compose_up)
        self.assertIn("/workspace/prepare_robot_home.py", self.text)
        self.assertIn("robot_home_contract.json", self.text)

    def test_live_tty_is_captured_before_tee_and_phrase_reads_from_tty(self) -> None:
        capture = self.text.index("interactive_terminal=false")
        redirect = self.text.index('exec > >(tee -a "${driver_log_file}") 2>&1')
        self.assertLess(capture, redirect)
        self.assertIn('read -r -p "Type exactly', self.text)
        self.assertIn("answer </dev/tty", self.text)

    def test_live_uses_joint_worker_and_waits_for_policy_ready(self) -> None:
        live_start = self.text.index("  live)")
        live_end = self.text.index("\n    ;;", live_start)
        live = self.text[live_start:live_end]
        self.assertIn("export JOINT_STATE_WORKER=true", live)
        self.assertIn("export DEBUG=false", live)
        self.assertIn("export MONITOR_ENABLE=true", live)
        self.assertIn('"[RTC] POLICY_READY"', live)

    def test_driver_and_container_evidence_are_separate(self) -> None:
        self.assertIn('.driver.log"', self.text)
        self.assertIn('exec > >(tee -a "${driver_log_file}") 2>&1', self.text)
        self.assertIn('mv -f "${refresh_tmp}" "${log_file}"', self.text)
        self.assertNotIn("logs --no-color --follow", self.text)
        self.assertNotIn('tee -a "${log_file}"', self.text)

    def test_supervision_starts_immediately_after_compose_up(self) -> None:
        compose_up = self.text.index('up --no-build --detach "${services[@]}"')
        supervisor_start = self.text.index("start_supervision", compose_up)
        startup_wait = self.text.index("deadline=$((SECONDS + 300))", compose_up)
        compose_attempt_marked = self.text.index("compose_started=true", compose_up - 250)
        self.assertLess(compose_attempt_marked, compose_up)
        self.assertLess(compose_up, supervisor_start)
        self.assertLess(supervisor_start, startup_wait)

    def test_cleanup_refreshes_before_down(self) -> None:
        cleanup_start = self.text.index("cleanup() {")
        cleanup_end = self.text.index("\n}\nhandle_signal()", cleanup_start)
        cleanup = self.text[cleanup_start:cleanup_end]
        self.assertLess(cleanup.index("refresh_logs"), cleanup.index("down --remove-orphans"))

    def test_long_compose_gates_are_interruptible_and_cleaned_up(self) -> None:
        helper_start = self.text.index("run_interruptible() {")
        helper_end = self.text.index("\n}\n", helper_start)
        helper = self.text[helper_start:helper_end]
        self.assertIn('gate_pid="$!"', helper)
        self.assertIn('wait "${gate_pid}"', helper)

        self.assertIn(
            'run_interruptible "${compose[@]}" run --rm --no-deps dds-check',
            self.text,
        )
        self.assertIn('run_interruptible "${compose[@]}" build inference', self.text)

        cleanup_start = self.text.index("cleanup() {")
        cleanup_end = self.text.index("\n}\nhandle_signal()", cleanup_start)
        cleanup = self.text[cleanup_start:cleanup_end]
        self.assertIn('kill -TERM "${gate_pid}"', cleanup)
        self.assertIn('wait "${gate_pid}"', cleanup)

    def test_shadow_telemetry_is_started_and_cleaned_up(self) -> None:
        compose_up = self.text.index('up --no-build --detach "${services[@]}"')
        telemetry_start = self.text.index("start_runtime_telemetry", compose_up)
        startup_wait = self.text.index("deadline=$((SECONDS + 300))", compose_up)
        self.assertLess(compose_up, telemetry_start)
        self.assertLess(telemetry_start, startup_wait)

        cleanup_start = self.text.index("cleanup() {")
        cleanup_end = self.text.index("\n}\nhandle_signal()", cleanup_start)
        cleanup = self.text[cleanup_start:cleanup_end]
        self.assertIn('kill "${telemetry_pid}"', cleanup)
        self.assertIn('wait "${telemetry_pid}"', cleanup)

    def test_quiet_shadow_preserves_shadow_gates_without_monitor(self) -> None:
        quiet_start = self.text.index("  shadow_quiet)")
        quiet_end = self.text.index("\n    ;;", quiet_start)
        quiet = self.text[quiet_start:quiet_end]
        self.assertIn("shadow_mode=true", quiet)
        self.assertIn("export DEBUG=false", quiet)
        self.assertIn("export MONITOR_ENABLE=false", quiet)
        self.assertIn("services=(inference)", quiet)
        self.assertIn('"[RTC] POLICY_READY"', quiet)

        self.assertGreaterEqual(self.text.count('if [[ "${shadow_mode}" == "true" ]]'), 3)

    def test_joint_worker_ab_is_quiet_shadow_only(self) -> None:
        worker_start = self.text.index("  shadow_joint_worker)")
        worker_end = self.text.index("\n    ;;", worker_start)
        worker = self.text[worker_start:worker_end]
        self.assertIn("shadow_mode=true", worker)
        self.assertIn("export DEBUG=false", worker)
        self.assertIn("export MONITOR_ENABLE=false", worker)
        self.assertIn("export JOINT_STATE_WORKER=true", worker)
        self.assertIn("services=(inference)", worker)
        self.assertIn("joint_state_worker=True", worker)

    def test_joint_worker_monitor_preserves_csv_and_shadow_gates(self) -> None:
        worker_start = self.text.index("  shadow_joint_worker_monitor)")
        worker_end = self.text.index("\n    ;;", worker_start)
        worker = self.text[worker_start:worker_end]
        self.assertIn("shadow_mode=true", worker)
        self.assertIn("export DEBUG=false", worker)
        self.assertIn("export MONITOR_ENABLE=true", worker)
        self.assertIn("export JOINT_STATE_WORKER=true", worker)
        self.assertIn("services=(inference inference-monitor)", worker)
        self.assertIn("joint_state_worker=True", worker)

    def test_fail_closed_conditions_are_supervised(self) -> None:
        for expected in (
            "[WATCHDOG] LATCHED",
            "Traceback (most recent call last)",
            "required service stopped:",
            "NO-GO:",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, self.text)

    def test_supervisor_stops_project_before_signalling_blocked_runner(self) -> None:
        supervisor_start = self.text.index("start_supervision() {")
        supervisor_end = self.text.index("\n}\n\nstart_runtime_telemetry", supervisor_start)
        supervisor = self.text[supervisor_start:supervisor_end]
        fatal_branch = supervisor.index('if [[ -n "${reason}" ]]')
        stop = supervisor.index("stop_project_containers_now", fatal_branch)
        signal = supervisor.index('kill -HUP "${runner_pid}"', fatal_branch)
        self.assertLess(stop, signal)

        stop_fn_start = self.text.index("stop_project_containers_now() {")
        stop_fn_end = self.text.index("\n}\n\nstart_supervision", stop_fn_start)
        stop_fn = self.text[stop_fn_start:stop_fn_end]
        self.assertIn("label=com.docker.compose.project=${PROJECT_NAME}", stop_fn)
        self.assertIn("stop_inference_container_now", stop_fn)
        self.assertIn('docker stop --time 2 "${container_id}"', stop_fn)

        inference_stop_start = self.text.index("stop_inference_container_now() {")
        inference_stop_end = self.text.index("\n}\n", inference_stop_start)
        inference_stop = self.text[inference_stop_start:inference_stop_end]
        self.assertIn("label=com.docker.compose.service=inference", inference_stop)


if __name__ == "__main__":
    unittest.main()
