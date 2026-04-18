"""
compiler.py — Natural language → skill program JSON compiler
Owner: Vera

Runs Phi-3-mini-4k-instruct (Microsoft's on-device SLM) on the AMD
ROCm GPU via llama-cpp-python or ONNX Runtime ROCm execution provider.

Usage:
    compiler = SkillCompiler()                     # auto-detects backend
    compiler = SkillCompiler(backend="mock")       # for testing without GPU
    skill = compiler.compile("Put the canned goods on the middle shelf")
    # skill is a dict matching SKILL_PROGRAM_SCHEMA

Backends (tried in order if backend="auto"):
    1. llama_cpp   — llama-cpp-python with Phi-3 GGUF model (ROCm via HIP)
    2. onnx_rocm   — ONNX Runtime ROCMExecutionProvider (AMD ROCm GPU)
    3. mock        — Returns deterministic test fixture (never in production)

Install notes:
    llama_cpp:  CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python
    onnx_rocm:  pip install onnxruntime-rocm --break-system-packages
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# The four skills robot_api.replay_skill() handles directly by name.
# These are the only valid action values in a compiled skill program.
VALID_ACTIONS = {
    "pick_object",        # pick an object from the scene into the gripper
    "place_in_box",       # place whatever is in the gripper into the target box
    "full_pick_and_place", # pick an object and place it in the box in one motion
    "box_in_shelf",       # pick up a box and place it upright in a shelf slot
}

SKILL_PROGRAM_SCHEMA = {
    "skill_name": "string (snake_case identifier for this high-level task)",
    "description": "string",
    "steps": [
        {
            "step_id": "int (0-indexed)",
            "action": f"string — one of: {' | '.join(sorted(VALID_ACTIONS))}",
            "verification_query": (
                "string — yes/no question the VLM checks after this skill completes"
            ),
            "expected_result": "bool",
        }
    ],
    "parameters": {
        "z_offset_mm": "float — vertical approach offset in mm, default 0",
        "speed_scale": "float — replay speed multiplier, default 1.0",
        "approach_angle_deg": "float — wrist approach rotation in degrees, default 0",
        "gripper_close_force": "float — grip strength 0.0–1.0, default 0.6",
        "retry_count": "int — max automatic retries per step, default 2",
    },
}

# ---------------------------------------------------------------------------
# Static skill registry — the four canonical robot skills
#
# These are returned directly by compile() without invoking the LLM.
# Use these exact skill_name strings when calling run_skill() or
# run_from_webcam(). Diya must record LeRobot trajectories under these names.
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS = {
    "z_offset_mm": 0.0,
    "speed_scale": 1.0,
    "approach_angle_deg": 0.0,
    "gripper_close_force": 0.6,
    "retry_count": 2,
}

SKILL_REGISTRY: dict[str, dict] = {
    "pick_object": {
        "skill_name": "pick_object",
        "description": "Locate and pick an object from the scene into the gripper.",
        "steps": [
            {
                "step_id": 0,
                "action": "scan_scene",
                "verification_query": "Is there an object visible in the pick zone?",
                "expected_result": True,
            },
            {
                "step_id": 1,
                "action": "pick_object",
                "verification_query": "Is an object held securely in the gripper?",
                "expected_result": True,
            },
        ],
        "parameters": dict(_DEFAULT_PARAMS),
    },
    "place_in_box": {
        "skill_name": "place_in_box",
        "description": "Place the object currently held in the gripper into the target box.",
        "steps": [
            {
                "step_id": 0,
                "action": "place_in_box",
                "verification_query": "Is the object now placed inside the box?",
                "expected_result": True,
            },
            {
                "step_id": 1,
                "action": "scan_scene",
                "verification_query": "Is the box now occupied with the placed object?",
                "expected_result": True,
            },
        ],
        "parameters": dict(_DEFAULT_PARAMS),
    },
    "full_pick_and_place": {
        "skill_name": "full_pick_and_place",
        "description": "Pick an object from the scene and place it into the target box.",
        "steps": [
            {
                "step_id": 0,
                "action": "scan_scene",
                "verification_query": "Is there an object visible in the pick zone?",
                "expected_result": True,
            },
            {
                "step_id": 1,
                "action": "pick_object",
                "verification_query": "Is an object held securely in the gripper?",
                "expected_result": True,
            },
            {
                "step_id": 2,
                "action": "place_in_box",
                "verification_query": "Is the object now placed inside the box?",
                "expected_result": True,
            },
        ],
        "parameters": dict(_DEFAULT_PARAMS),
    },
    "box_in_shelf": {
        "skill_name": "box_in_shelf",
        "description": "Pick up a box and place it upright in an empty shelf slot.",
        "steps": [
            {
                "step_id": 0,
                "action": "scan_scene",
                "verification_query": "Is there an empty slot visible on the shelf?",
                "expected_result": True,
            },
            {
                "step_id": 1,
                "action": "pick_object",
                "verification_query": "Is a box held securely in the gripper?",
                "expected_result": True,
            },
            {
                "step_id": 2,
                "action": "move_box_to_shelf",
                "verification_query": "Is a box standing upright in the shelf slot?",
                "expected_result": True,
            },
        ],
        "parameters": dict(_DEFAULT_PARAMS),
    },
}


# ---------------------------------------------------------------------------
# System prompt (load-bearing — do not shorten)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a robot skill compiler for a LeRobot arm. You translate natural language \
commands into high-level skill programs.

HOW IT WORKS
The robot has four reusable sub-skills. Each step in your output invokes one of \
them by name. The robot executes the sub-skill autonomously; then the VLM checks \
your verification_query against the live camera feed before moving to the next step.

AVAILABLE SUB-SKILLS — every step's "action" must be exactly one of these:
  pick_object        — locate an object in the scene and pick it up into the gripper
  place_in_box       — place whatever the gripper is holding into the target box
  full_pick_and_place — pick an object from the scene and place it directly into the box
  box_in_shelf       — pick up a box and place it upright in an empty shelf slot

PARAMETERS — shared across all steps; tune only when the command implies it:
  z_offset_mm         float   vertical approach offset in mm          default 0
  speed_scale         float   replay speed multiplier                  default 1.0
  approach_angle_deg  float   wrist rotation offset in degrees         default 0
  gripper_close_force float   grip strength 0.0–1.0                   default 0.6
  retry_count         int     max automatic retries per step           default 2

OUTPUT SCHEMA — respond with a single JSON object matching this exactly:
{
  "skill_name": "<snake_case name for this overall task>",
  "description": "<one sentence describing the full task>",
  "steps": [
    {
      "step_id": <int, 0-indexed>,
      "action": "<pick_object | place_in_box | full_pick_and_place | box_in_shelf>",
      "verification_query": "<yes/no question the VLM checks after this sub-skill completes>",
      "expected_result": <true | false>
    }
  ],
  "parameters": {
    "z_offset_mm": <float>,
    "speed_scale": <float>,
    "approach_angle_deg": <float>,
    "gripper_close_force": <float>,
    "retry_count": <int>
  }
}

CANONICAL EXAMPLES:

Example 1 — "Refill the inventory":
{
  "skill_name": "refill_inventory",
  "description": "Pick all loose objects from the scene, place them in the box, then shelve the box.",
  "steps": [
    {"step_id": 0, "action": "full_pick_and_place", "verification_query": "Is there at least one object now inside the box?",          "expected_result": true},
    {"step_id": 1, "action": "full_pick_and_place", "verification_query": "Are there any remaining loose objects visible in the scene?", "expected_result": false},
    {"step_id": 2, "action": "box_in_shelf",         "verification_query": "Is the box now standing upright on the shelf?",              "expected_result": true}
  ],
  "parameters": {"z_offset_mm": 0, "speed_scale": 1.0, "approach_angle_deg": 0, "gripper_close_force": 0.6, "retry_count": 2}
}

Example 2 — "Pick up the item and store it":
{
  "skill_name": "pick_and_store",
  "description": "Pick an object from the scene and deposit it into the storage box.",
  "steps": [
    {"step_id": 0, "action": "pick_object",  "verification_query": "Is an object held securely in the gripper?",  "expected_result": true},
    {"step_id": 1, "action": "place_in_box", "verification_query": "Is the object now inside the box?",           "expected_result": true}
  ],
  "parameters": {"z_offset_mm": 0, "speed_scale": 1.0, "approach_angle_deg": 0, "gripper_close_force": 0.6, "retry_count": 2}
}

Example 3 — "Stock the shelf":
{
  "skill_name": "stock_shelf",
  "description": "Pick and place objects into the box until none remain, then move the box onto the shelf.",
  "steps": [
    {"step_id": 0, "action": "full_pick_and_place", "verification_query": "Is there at least one object now inside the box?",           "expected_result": true},
    {"step_id": 1, "action": "box_in_shelf",         "verification_query": "Is the box now placed upright in a shelf slot?",             "expected_result": true}
  ],
  "parameters": {"z_offset_mm": 0, "speed_scale": 1.0, "approach_angle_deg": 0, "gripper_close_force": 0.6, "retry_count": 2}
}

RULES:
- Use only the four listed sub-skill names. Never invent new action names.
- Sequence steps so each sub-skill's precondition is satisfied by the previous step \
(e.g. place_in_box assumes the gripper is already holding something; use pick_object first).
- verification_query must be a yes/no question answerable from a single camera frame.
- Use expected_result: false when the correct state is an absence (e.g. "no loose objects remain").
- Repeat a sub-skill in consecutive steps when the command implies multiple objects.
- Adjust parameters only when the natural language implies it \
(e.g. "gently" → gripper_close_force: 0.3, "slowly" → speed_scale: 0.5).

Respond ONLY with valid JSON matching this schema. No other text. No markdown. No explanation.\
"""

# ---------------------------------------------------------------------------
# Default model paths — override via environment variables
# ---------------------------------------------------------------------------

_DEFAULT_GGUF_PATH = os.environ.get(
    "PHI3_GGUF_PATH",
    "models/Phi-3-mini-4k-instruct-q4.gguf",
)
_DEFAULT_ONNX_PATH = os.environ.get(
    "PHI3_ONNX_PATH",
    "models/phi-3-mini-4k-instruct-onnx",
)

BackendType = Literal["auto", "llama_cpp", "onnx_rocm", "mock"]


# ---------------------------------------------------------------------------
# Compiler class
# ---------------------------------------------------------------------------

class SkillCompiler:
    """
    Compiles natural language shelf commands into structured skill programs.

    Args:
        backend:    "auto" tries llama_cpp → onnx_rocm → mock.
                    Force a specific backend for testing or when you know
                    what's available.
        model_path: Path to GGUF model (llama_cpp backend) or ONNX model
                    directory (onnx_rocm backend).
        temperature: Sampling temperature. Keep at 0.0 for deterministic output.
    """

    def __init__(
        self,
        backend: BackendType = "auto",
        model_path: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.temperature = temperature
        self._backend, self._engine = self._init_backend(backend, model_path)
        logger.info("SkillCompiler initialized with backend: %s", self._backend)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compile(self, nl_command: str) -> dict[str, Any]:
        """
        Compile a natural language command into a skill program.

        If nl_command exactly matches a key in SKILL_REGISTRY (case-insensitive,
        underscores/spaces normalised), the static program is returned immediately
        without invoking the LLM. This is the fast path used by run_from_webcam().

        Args:
            nl_command: Free-form shelf stocking instruction, e.g.
                        "Put the soup cans on the middle shelf",
                        OR a canonical skill name, e.g. "pick_object".

        Returns:
            Skill program dict matching SKILL_PROGRAM_SCHEMA.

        Raises:
            ValueError: If the model output cannot be parsed as valid JSON
                        matching the schema after two attempts.
        """
        # --- Registry fast path ---
        normalised = nl_command.strip().lower().replace(" ", "_")
        if normalised in SKILL_REGISTRY:
            skill = SKILL_REGISTRY[normalised]
            logger.info(
                "Registry hit for %r → %s (%d steps)",
                nl_command, skill["skill_name"], len(skill["steps"]),
            )
            return skill

        logger.info("Compiling via LLM: %r", nl_command)
        raw = self._engine(nl_command)
        skill = self._parse_and_validate(raw)

        logger.info("Compiled skill: %s (%d steps)", skill["skill_name"], len(skill["steps"]))
        return skill

    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    # Backend initialization
    # ------------------------------------------------------------------

    def _init_backend(
        self, backend: BackendType, model_path: str | None
    ) -> tuple[str, Any]:
        # [MOCK] Explicit mock request — skip hardware detection entirely.
        if backend == "mock":
            return "mock", self._mock_engine

        # [REAL] Primary path: Phi-3-mini-4k GGUF via llama-cpp-python (ROCm/HIP).
        # Setup: CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python
        #        download Phi-3-mini-4k-instruct-q4.gguf, set PHI3_GGUF_PATH.
        if backend in ("llama_cpp", "auto"):
            result = self._try_llama_cpp(model_path)
            if result is not None:
                return "llama_cpp", result
            if backend == "llama_cpp":
                raise RuntimeError(
                    "llama_cpp backend requested but llama-cpp-python is not installed "
                    "or model not found. Run: pip install llama-cpp-python"
                )

        # [REAL] Secondary path: Phi-3-mini ONNX via onnxruntime-rocm.
        # Setup: pip install onnxruntime-rocm transformers tokenizers
        #        set PHI3_ONNX_PATH to the converted model directory.
        if backend in ("onnx_rocm", "auto"):
            result = self._try_onnx_rocm(model_path)
            if result is not None:
                return "onnx_rocm", result
            if backend == "onnx_rocm":
                raise RuntimeError(
                    "onnx_rocm backend requested but onnxruntime-rocm is not installed "
                    "or model not found. Run: pip install onnxruntime-rocm"
                )

        # [MOCK] auto fell through — neither real backend found.
        # Output is deterministic keyword routing, NOT Phi-3 inference.
        logger.warning(
            "No GPU backend available (llama_cpp or onnx_rocm). "
            "Falling back to [MOCK] compiler — keyword routing only, not real Phi-3. "
            "Set PHI3_GGUF_PATH or install llama-cpp-python for real inference."
        )
        return "mock", self._mock_engine

    def _try_llama_cpp(self, model_path: str | None):
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError:
            logger.debug("llama-cpp-python not installed")
            return None

        path = model_path or _DEFAULT_GGUF_PATH
        if not Path(path).exists():
            logger.debug("GGUF model not found at %s", path)
            return None

        logger.info("Loading Phi-3 via llama_cpp from %s", path)
        llm = Llama(
            model_path=path,
            n_ctx=4096,
            n_gpu_layers=-1,   # offload all layers to ROCm GPU (HIP backend)
            verbose=False,
        )

        def engine(nl_command: str) -> str:
            # Do NOT pass response_format={"type": "json_object"} — the generic
            # JSON grammar suppresses Phi-3's content generation and produces {}.
            # The system prompt + _parse_and_validate handle cleanup instead.
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": nl_command},
                ],
                temperature=self.temperature,
                max_tokens=1024,
            )
            return response["choices"][0]["message"]["content"]

        return engine

    def _try_onnx_rocm(self, model_path: str | None):
        try:
            import onnxruntime as ort  # type: ignore
            from transformers import AutoTokenizer  # type: ignore
        except ImportError:
            logger.debug("onnxruntime or transformers not installed")
            return None

        path = model_path or _DEFAULT_ONNX_PATH
        if not Path(path).exists():
            logger.debug("ONNX model directory not found at %s", path)
            return None

        providers = ort.get_available_providers()
        if "ROCMExecutionProvider" not in providers:
            logger.debug("ROCMExecutionProvider not available, providers: %s", providers)
            return None

        logger.info("Loading Phi-3 via ONNX ROCm from %s", path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        session = ort.InferenceSession(
            str(Path(path) / "model.onnx"),
            providers=["ROCMExecutionProvider"],
        )

        def engine(nl_command: str) -> str:
            # Build prompt in Phi-3 chat format
            prompt = (
                f"<|system|>\n{_SYSTEM_PROMPT}<|end|>\n"
                f"<|user|>\n{nl_command}<|end|>\n"
                f"<|assistant|>\n"
            )
            inputs = tokenizer(prompt, return_tensors="np")
            outputs = session.run(None, dict(inputs))
            # Decode generated tokens (implementation depends on ONNX model variant)
            # For the GenAI ONNX models use model_builder output format
            generated = tokenizer.decode(outputs[0][0], skip_special_tokens=True)
            # Strip the prompt prefix if echoed
            if "<|assistant|>" in generated:
                generated = generated.split("<|assistant|>")[-1].strip()
            return generated

        return engine

    # ------------------------------------------------------------------
    # Mock engine (testing without hardware)
    # ------------------------------------------------------------------

    def _mock_engine(self, nl_command: str) -> str:
        """
        [MOCK] Deterministic keyword-based skill compiler.

        Returns a valid compiled skill program without invoking Phi-3.
        Used when backend="mock" or when no GPU backend is available.
        Output is keyword routing — it will be wrong for ambiguous commands.

        [REAL] In production this method is never called. The llama_cpp or
        onnx_rocm engine (see _init_backend) handles all compilation via
        actual Phi-3-mini-4k-instruct inference on the AMD ROCm GPU.
        """
        cmd = nl_command.lower()

        # Keyword routing → assemble registry-skill steps that match the intent.
        # Priority tiers (checked in order, first match wins):
        #   1. shelf + restock commands  → full workflow ending with box_in_shelf
        #   2. "just shelve the box"     → box_in_shelf only
        #   3. place / deposit only      → place_in_box only (gripper already full)
        #   4. pick only (no destination mentioned)
        #   5. default                   → full_pick_and_place

        if any(kw in cmd for kw in ("refill", "restock", "inventory", "stock")):
            skill_name = "refill_inventory_mock"
            steps = [
                {"step_id": 0, "action": "full_pick_and_place",
                 "verification_query": "Is there at least one object now inside the box?",
                 "expected_result": True},
                {"step_id": 1, "action": "full_pick_and_place",
                 "verification_query": "Are there any remaining loose objects visible in the scene?",
                 "expected_result": False},
                {"step_id": 2, "action": "box_in_shelf",
                 "verification_query": "Is the box now standing upright on the shelf?",
                 "expected_result": True},
            ]
        elif any(kw in cmd for kw in ("shelf", "shelve", "shelving")):
            skill_name = "shelve_box_mock"
            steps = [
                {"step_id": 0, "action": "box_in_shelf",
                 "verification_query": "Is the box now standing upright on the shelf?",
                 "expected_result": True},
            ]
        elif any(kw in cmd for kw in ("place", "put in box", "drop", "deposit", "release")):
            skill_name = "place_in_box_mock"
            steps = [
                {"step_id": 0, "action": "place_in_box",
                 "verification_query": "Is the object now inside the box?",
                 "expected_result": True},
            ]
        elif any(kw in cmd for kw in ("pick up", "pick", "grab", "grasp", "get", "fetch")):
            skill_name = "pick_object_mock"
            steps = [
                {"step_id": 0, "action": "pick_object",
                 "verification_query": "Is an object held securely in the gripper?",
                 "expected_result": True},
            ]
        else:
            # Default: pick an object and place it in the box
            skill_name = "pick_and_place_mock"
            steps = [
                {"step_id": 0, "action": "full_pick_and_place",
                 "verification_query": "Is the object now inside the box?",
                 "expected_result": True},
            ]

        return json.dumps({
            "skill_name": skill_name,
            "description": f"[MOCK] {nl_command}",
            "steps": steps,
            "parameters": {
                "z_offset_mm": 0.0,
                "speed_scale": 1.0,
                "approach_angle_deg": 0.0,
                "gripper_close_force": 0.6,
                "retry_count": 2,
            },
        })

    # ------------------------------------------------------------------
    # JSON parsing and validation
    # ------------------------------------------------------------------

    def _parse_and_validate(self, raw: str) -> dict[str, Any]:
        """Extract and validate JSON from model output."""
        # Strip markdown code fences if the model added them
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # Extract the first complete {...} JSON object.
        # Phi-3 sometimes appends "Note: ..." text after the JSON — this strips it.
        cleaned = self._extract_first_json_object(cleaned)

        try:
            skill = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Compiler output is not valid JSON.\n"
                f"Raw output: {raw!r}\n"
                f"Error: {e}"
            ) from e

        # Auto-unwrap: if required fields are missing the model may have wrapped
        # the program in a parent object (e.g. {"skill_program": {...}}).
        # Search recursively up to 3 levels deep for a dict that contains at
        # least one required field rather than requiring an exact single-child match.
        required_top = {"skill_name", "description", "steps", "parameters"}
        if not (required_top & skill.keys()):
            candidate = self._find_skill_dict(skill, required_top, depth=3)
            if candidate is not None:
                logger.debug(
                    "Auto-unwrapping nested skill program (found at depth > 0)"
                )
                skill = candidate

        try:
            self._validate_schema(skill)
        except ValueError as exc:
            # Re-raise with the raw model output attached so debugging is possible.
            raise ValueError(
                f"{exc}\n"
                f"Parsed JSON: {json.dumps(skill, indent=2)}\n"
                f"Raw model output: {raw!r}"
            ) from exc
        return skill

    @staticmethod
    def _find_skill_dict(
        obj: dict, required_top: set, depth: int
    ) -> "dict | None":
        """
        Recursively search *obj* for a nested dict that contains at least one
        key from *required_top*.  Returns the best match (most required keys
        present) found within *depth* levels, or None if nothing is found.
        """
        if depth == 0:
            return None
        best: dict | None = None
        best_score = 0
        for v in obj.values():
            if not isinstance(v, dict):
                continue
            score = len(required_top & v.keys())
            if score > best_score:
                best_score = score
                best = v
            # Recurse
            deeper = SkillCompiler._find_skill_dict(v, required_top, depth - 1)
            if deeper is not None:
                deeper_score = len(required_top & deeper.keys())
                if deeper_score > best_score:
                    best_score = deeper_score
                    best = deeper
        return best if best_score > 0 else None

    @staticmethod
    def _extract_first_json_object(text: str) -> str:
        """
        Return the first complete ``{...}`` JSON object found in *text*.

        Phi-3 occasionally appends a "Note: ..." sentence after the closing
        brace.  This method finds the balanced closing brace and truncates
        anything after it, so ``json.loads`` receives clean input.

        Falls back to returning *text* unchanged if no complete object is
        found (the subsequent ``json.loads`` call will raise the right error).
        """
        depth = 0
        start: Optional[int] = None
        for i, ch in enumerate(text):
            if ch == "{":
                if start is None:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : i + 1]
        return text  # no complete object — return as-is

    def _validate_schema(self, skill: dict) -> None:
        """Raise ValueError if required fields are missing or malformed."""
        required_top = {"skill_name", "description", "steps", "parameters"}
        missing = required_top - skill.keys()
        if missing:
            raise ValueError(f"Skill program missing required fields: {missing}")

        if not isinstance(skill["steps"], list) or len(skill["steps"]) == 0:
            raise ValueError("'steps' must be a non-empty list")

        required_step = {"step_id", "action", "verification_query", "expected_result"}
        for i, step in enumerate(skill["steps"]):
            missing_step = required_step - step.keys()
            if missing_step:
                raise ValueError(f"Step {i} missing fields: {missing_step}")
            action = step.get("action", "")
            if action not in VALID_ACTIONS:
                raise ValueError(
                    f"Step {i} has unknown action {action!r}. "
                    f"Must be one of: {sorted(VALID_ACTIONS)}"
                )

        required_params = {
            "z_offset_mm", "speed_scale", "approach_angle_deg",
            "gripper_close_force", "retry_count",
        }
        params = skill.get("parameters", {})
        missing_params = required_params - params.keys()
        if missing_params:
            raise ValueError(f"Parameters missing fields: {missing_params}")


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="SkillPatch NL compiler")
    parser.add_argument("command", nargs="?", default="Put the canned goods on the middle shelf")
    parser.add_argument("--backend", default="auto", choices=["auto", "llama_cpp", "onnx_rocm", "mock"])
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    compiler = SkillCompiler(backend=args.backend, model_path=args.model_path)
    result = compiler.compile(args.command)
    print(json.dumps(result, indent=2))
