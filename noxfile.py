import nox

nox.options.default_venv_backend = "uv"

nox.options.sessions = ["lint_or_check"]

@nox.session(python=["3.11", "3.12", "3.13"])
def lint_or_check(session):
    """Simple sanity check before publishing."""
    session.install("-e", ".")
    session.run("python", "-c", "import img2img_turbo; print('Import OK')")
