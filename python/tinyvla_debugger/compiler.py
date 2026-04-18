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
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SKILL_PROGRAM_SCHEMA = {
    "skill_name": "string (snake_case identifier)",
    "description": "string",
    "steps": [
        {
            "step_id": "int (0-indexed)",
            "action": "string — one of: scan_shelf, pick_from_box, place_slot_1, "
                      "place_slot_2, place_slot_3, check_box_empty",
            "verification_query": "string — yes/no question about the scene",
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
# System prompt (load-bearing — do not shorten)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a robot skill compiler. You convert natural language shelf-stocking \
commands into structured skill programs for a LeRobot arm.

OUTPUT SCHEMA — your response must be a single JSON object matching this exactly:
{
  "skill_name": "<snake_case identifier for this skill>",
  "description": "<one sentence describing what this skill does>",
  "steps": [
    {
      "step_id": <int, 0-indexed>,
      "action": "<one of: scan_shelf | pick_from_box | place_slot_1 | place_slot_2 | place_slot_3 | check_box_empty>",
      "verification_query": "<yes/no question the VLM will check after this step>",
      "expected_result": <true | false>
    }
  ],
  "parameters": {
    "z_offset_mm": <float, default 0>,
    "speed_scale": <float, default 1.0>,
    "approach_angle_deg": <float, default 0>,
    "gripper_close_force": <float, default 0.6>,
    "retry_count": <int, default 2>
  }
}

CANONICAL EXAMPLES:

Example 1 — "Put the canned goods on the middle shelf":
{
  "skill_name": "stock_middle_shelf",
  "description": "Pick items from source box, place on middle shelf slots",
  "steps": [
    {"step_id": 0, "action": "scan_shelf", "verification_query": "Are there empty slots visible on the middle shelf?", "expected_result": true},
    {"step_id": 1, "action": "pick_from_box", "verification_query": "Is an object held securely in the gripper?", "expected_result": true},
    {"step_id": 2, "action": "place_slot_1", "verification_query": "Is there an item standing upright in shelf slot 1?", "expected_result": true},
    {"step_id": 3, "action": "check_box_empty", "verification_query": "Is the source box empty?", "expected_result": true}
  ],
  "parameters": {"z_offset_mm": 0, "speed_scale": 1.0, "approach_angle_deg": 0, "gripper_close_force": 0.6, "retry_count": 2}
}

Example 2 — "Refill slot 2 only":
{
  "skill_name": "refill_slot_2",
  "description": "Pick one item from source box and place it in shelf slot 2",
  "steps": [
    {"step_id": 0, "action": "pick_from_box", "verification_query": "Is an object held securely in the gripper?", "expected_result": true},
    {"step_id": 1, "action": "place_slot_2", "verification_query": "Is there an item standing upright in shelf slot 2?", "expected_result": true}
  ],
  "parameters": {"z_offset_mm": 0, "speed_scale": 1.0, "approach_angle_deg": 0, "gripper_close_force": 0.6, "retry_count": 2}
}

Example 3 — "Stock slots 1 and 2":
{
  "skill_name": "stock_slots_1_and_2",
  "description": "Pick and place items into shelf slots 1 and 2 sequentially",
  "steps": [
    {"step_id": 0, "action": "scan_shelf", "verification_query": "Are slots 1 and 2 empty?", "expected_result": true},
    {"step_id": 1, "action": "pick_from_box", "verification_query": "Is an object held securely in the gripper?", "expected_result": true},
    {"step_id": 2, "action": "place_slot_1", "verification_query": "Is there an item standing upright in shelf slot 1?", "expected_result": true},
    {"step_id": 3, "action": "pick_from_box", "verification_query": "Is an object held securely in the gripper?", "expected_result": true},
    {"step_id": 4, "action": "place_slot_2", "verification_query": "Is there an item standing upright in shelf slot 2?", "expected_result": true},
    {"step_id": 5, "action": "check_box_empty", "verification_query": "Is the source box empty?", "expected_result": true}
  ],
  "parameters": {"z_offset_mm": 0, "speed_scale": 1.0, "approach_angle_deg": 0, "gripper_close_force": 0.6, "retry_count": 2}
}

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
        temperature: Sampling temperature. 0.1 recommended — 0.0 (greedy) causes
                     degenerate empty output with long prompts on some GGUF builds.
    """

    def __init__(
        self,
        backend: BackendType = "auto",
        model_path: str | None = None,
        temperature: float = 0.1,
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

        Args:
            nl_command: Free-form shelf stocking instruction, e.g.
                        "Put the soup cans on the middle shelf".

        Returns:
            Skill program dict matching SKILL_PROGRAM_SCHEMA.

        Raises:
            ValueError: If the model output cannot be parsed as valid JSON
                        matching the schema after two attempts.
        """
        logger.info("Compiling: %r", nl_command)

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
        if backend == "mock":
            return "mock", self._mock_engine

        if backend in ("llama_cpp", "auto"):
            result = self._try_llama_cpp(model_path)
            if result is not None:
                return "llama_cpp", result
            if backend == "llama_cpp":
                raise RuntimeError(
                    "llama_cpp backend requested but llama-cpp-python is not installed "
                    "or model not found. Run: pip install llama-cpp-python"
                )

        if backend in ("onnx_rocm", "auto"):
            result = self._try_onnx_rocm(model_path)
            if result is not None:
                return "onnx_rocm", result
            if backend == "onnx_rocm":
                raise RuntimeError(
                    "onnx_rocm backend requested but onnxruntime-rocm is not installed "
                    "or model not found. Run: pip install onnxruntime-rocm"
                )

        # auto fell through — use mock with a clear warning
        logger.warning(
            "No GPU backend available (llama_cpp or onnx_rocm). "
            "Falling back to MOCK compiler — output is deterministic test data, "
            "not real Phi-3 inference. Set PHI3_GGUF_PATH or install llama-cpp-python."
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
            # Raw completion with Phi-3 chat template built manually.
            # - No manual <s>: llama_cpp adds BOS automatically; including it
            #   in the string causes a duplicate BOS and degrades output quality.
            # - Assistant turn primed with "{" forces the model to start the JSON
            #   object immediately rather than generating preamble.
            # - temperature=0.0 (greedy) causes degenerate empty {} output with
            #   long prompts; 0.1 adds just enough variance to escape it.
            # - Text-level stops ("CANONICAL", "\nNote", "\nExample") catch
            #   Phi-3's tendency to repeat the system prompt after the JSON closes.
            prompt = (
                f"<|system|>\n{_SYSTEM_PROMPT}<|end|>\n"
                f"<|user|>\n{nl_command}<|end|>\n"
                f"<|assistant|>\n{{"
            )
            response = llm(
                prompt,
                temperature=max(self.temperature, 0.1),  # never go below 0.1
                max_tokens=1024,
                stop=[
                    "<|end|>", "<|user|>", "<|system|>", "<|endoftext|>",
                    "\nCANONICAL", "\nNote:", "\nExample",  # catch system-prompt bleed
                ],
                echo=False,
            )
            # Prepend the { we used to prime the assistant turn
            return "{" + response["choices"][0]["text"]

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

    # # ------------------------------------------------------------------
    # # Mock engine (testing without hardware)
    # # ------------------------------------------------------------------

    # def _mock_engine(self, nl_command: str) -> str:
    #     """
    #     Returns a deterministic skill program based on keywords in the command.
    #     NOT Phi-3 inference — for development and CI only.

    #     TODO: HARDWARE — this entire method is mock data. In production the
    #     llama_cpp or onnx_rocm engine is used instead (see _init_backend).
    #     This only runs when neither backend is available or backend="mock".
    #     """
    #     cmd = nl_command.lower()

    #     # Detect slot-specific commands
    #     slots: list[int] = []
    #     for i in (1, 2, 3):
    #         if str(i) in cmd or ("one" == cmd.split()[-1] and i == 1):
    #             slots.append(i)
    #     if not slots:
    #         slots = [1, 2]  # default: fill slots 1 and 2

    #     steps = [
    #         {
    #             "step_id": 0,
    #             "action": "scan_shelf",
    #             "verification_query": "Are there empty slots visible on the shelf?",
    #             "expected_result": True,
    #         }
    #     ]
    #     step_id = 1
    #     for slot in slots:
    #         steps.append({
    #             "step_id": step_id,
    #             "action": "pick_from_box",
    #             "verification_query": "Is an object held securely in the gripper?",
    #             "expected_result": True,
    #         })
    #         step_id += 1
    #         steps.append({
    #             "step_id": step_id,
    #             "action": f"place_slot_{slot}",
    #             "verification_query": f"Is there an item standing upright in shelf slot {slot}?",
    #             "expected_result": True,
    #         })
    #         step_id += 1
    #     steps.append({
    #         "step_id": step_id,
    #         "action": "check_box_empty",
    #         "verification_query": "Is the source box empty?",
    #         "expected_result": True,
    #     })

    #     skill_name = "stock_shelf_mock"
    #     if "slot" in cmd:
    #         slot_str = "_".join(f"slot_{s}" for s in slots)
    #         skill_name = f"stock_{slot_str}_mock"

    #     return json.dumps({
    #         "skill_name": skill_name,
    #         "description": f"[MOCK] {nl_command}",
    #         "steps": steps,
    #         "parameters": {
    #             "z_offset_mm": 0.0,
    #             "speed_scale": 1.0,
    #             "approach_angle_deg": 0.0,
    #             "gripper_close_force": 0.6,
    #             "retry_count": 2,
    #         },
    #     })

    # ------------------------------------------------------------------
    # JSON parsing and validation
    # ------------------------------------------------------------------

    def _parse_and_validate(self, raw: str) -> dict[str, Any]:
        """Extract and validate JSON from model output."""
        cleaned = raw.strip()

        # Strip markdown code fences if the model added them
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # Phi-3 often appends explanatory text after the JSON object
        # (e.g. "\n\nNote: ..."). Find the outermost {...} and discard the rest.
        cleaned = self._extract_first_json_object(cleaned)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Compiler output is not valid JSON.\n"
                f"Raw output: {raw!r}\n"
                f"Cleaned: {cleaned!r}\n"
                f"Error: {e}"
            ) from e

        # Phi-3 sometimes wraps the skill object in a single parent key,
        # e.g. {"result": {...}} or {"skill_program": {...}}.
        required_top = {"skill_name", "description", "steps", "parameters"}
        if not (required_top & parsed.keys()) and len(parsed) == 1:
            inner = next(iter(parsed.values()))
            if isinstance(inner, dict):
                logger.debug(
                    "Unwrapping model response: top-level key %r",
                    next(iter(parsed.keys())),
                )
                parsed = inner

        self._validate_schema(parsed)
        return parsed

    @staticmethod
    def _extract_first_json_object(text: str) -> str:
        """
        Return the substring covering the first complete {...} JSON object.
        Handles nested braces. If no object is found, returns the original text
        so the caller gets a clear JSONDecodeError rather than a silent miss.
        """
        start = text.find("{")
        if start == -1:
            return text
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start=start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return text  # malformed — let json.loads produce the error

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

        required_params = {
            "z_offset_mm", "speed_scale", "approach_angle_deg", "retry_count",
            # "gripper_close_force"
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
