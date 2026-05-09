"""
Code_Exec_Tools.py
------------------
A single LangChain-compatible tool that lets the LLM propose Python code
for execution but requires explicit human approval before running it.

Design rationale
----------------
- The gate is a text stdin prompt (not voice/TTS) because Y/N is the fastest
  binary UX pattern: one keypress, immediate response, no audio latency.
- The tool runs code in a subprocess so that bad code cannot corrupt the
  agent's in-process state or kill the interpreter.
- Only stdlib + langchain_core.tools — no LangGraph or project dependencies —
  so this module is importable anywhere in the project tree.
"""

import json
import sys
import subprocess

from langchain_core.tools import tool

# is_headless_mode is consulted at the top of code_exec so the tool refuses
# to run *any* user-supplied code on the public Render backend. The local CLI
# is unaffected — it sees is_headless_mode() == False and falls through to
# the existing y/n stdin gate.
from Project_Tools.Runtime_Options import is_headless_mode


@tool
def code_exec(code: str, reason: str) -> str:
    """
    Execute a Python code snippet after receiving explicit user approval.

    The tool prints the proposed code and the reason for running it to stdout,
    then blocks on a stdin prompt waiting for the user to type 'y' (allow) or
    anything else (deny).  Only an exact lowercase 'y' permits execution.

    On approval the code is run via subprocess using the same Python interpreter
    that hosts this process.  The subprocess is given a 20-second wall-clock
    timeout; long-running code is killed and a timeout error is returned.

    IMPORTANT — on denial: do NOT retry the same code.  Either ask the user
    what approach they prefer, or choose a completely different tool/strategy.

    Parameters
    ----------
    code : str
        Valid Python source code to execute.  The code runs in its own fresh
        interpreter, so no state from the agent process is inherited (other than
        installed packages and sys.executable).
    reason : str
        A short human-readable explanation of why this code needs to run.
        Shown to the user before they decide to allow or deny.

    Returns
    -------
    str
        JSON object with three keys:
          - "stdout"     : str  — captured standard output (truncated to 4000 chars)
          - "stderr"     : str  — captured standard error  (truncated to 4000 chars)
          - "returncode" : int  — process exit code (0 = success, non-zero = error,
                                  -1 = internal error or timeout)
        On denial returns the literal string:
          "User denied execution. Choose a different tool or ask the user what they prefer."
    """
    # -------------------------------------------------------------------------
    # Hosted backend safety gate.
    #
    # On the public Render deployment the y/n stdin prompt below is meaningless:
    # there is no human at stdin to approve, and even if there were, the caller
    # of the API is whoever hits /run, not the project owner. Allowing arbitrary
    # Python execution on a public web server with only a UI gate would let any
    # caller read environment variables (API keys), touch the filesystem, and
    # make outbound network calls. Until the tool runs inside a real sandbox
    # (Vercel Sandbox, e2b, gVisor microVM, etc.), the only safe behaviour in
    # hosted mode is a hard refusal. The local CLI is unaffected.
    # -------------------------------------------------------------------------
    if is_headless_mode():
        return (
            "code_exec is disabled in this hosted environment because arbitrary "
            "Python execution on a public server is not safe without a sandbox. "
            "Choose a different tool or describe the answer in prose."
        )

    # -------------------------------------------------------------------------
    # Step 1: Show the user what is about to run and why.
    # We print to stdout (not TTS) because Y/N is a synchronous binary decision;
    # a text prompt is faster and less disruptive than audio for a security gate.
    # -------------------------------------------------------------------------
    print(f"\n[code_exec] reason: {reason}")
    print("[code_exec] proposed code:")
    print(code)

    # -------------------------------------------------------------------------
    # Step 2: Block on stdin for explicit approval.
    # Any response other than exact lowercase "y" is treated as a denial —
    # this includes "n", "", "yes", typos, accidental newlines, etc.
    # Strict matching prevents an accidental Enter from approving execution.
    # -------------------------------------------------------------------------
    raw = input("\n[code_exec] Allow execution? (y/n): ")
    response = raw.strip().lower()

    if response != "y":
        # Return a plain string (not JSON) so the LLM's tool result parser
        # receives a clear, unambiguous refusal message rather than a JSON
        # error structure that might be misread as a code failure.
        return "User denied execution. Choose a different tool or ask the user what they prefer."

    # -------------------------------------------------------------------------
    # Step 3: Run the approved code in a subprocess.
    # Using sys.executable ensures the same venv / interpreter is used, so all
    # installed packages are available inside the subprocess.
    # capture_output=True prevents subprocess output from leaking to terminal
    # before we can include it in the structured JSON return.
    # -------------------------------------------------------------------------
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=20,  # 20-second wall-clock limit; kills runaway code
        )
        # Truncate at 4000 chars so the LLM's context window is not flooded
        # by large stdout/stderr from verbose scripts.
        return json.dumps({
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:4000],
            "returncode": result.returncode,
        })

    except subprocess.TimeoutExpired:
        # The subprocess was killed; return a structured error so the LLM
        # knows execution was attempted but did not complete.
        return json.dumps({
            "stdout": "",
            "stderr": "Execution timed out after 20s.",
            "returncode": -1,
        })

    except Exception as exc:
        # Catch-all for unexpected errors (e.g., OS-level fork failure).
        # returncode -1 signals an internal error distinct from a process failure.
        return json.dumps({
            "stdout": "",
            "stderr": f"code_exec error: {exc}",
            "returncode": -1,
        })
