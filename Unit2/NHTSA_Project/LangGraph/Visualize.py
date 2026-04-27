from Graph import app

# Use mermaid.ink API with extended retries to handle transient slowness.
# The default timeout=10 / max_retries=1 is too tight for the render API —
# bumping retries to 5 with a 3-second delay gives mermaid.ink time to
# respond without switching to a local browser renderer (pyppeteer is broken
# on Apple Silicon ARM64).
png_bytes = app.get_graph().draw_mermaid_png(
    max_retries=5,
    retry_delay=3.0,
)
with open("workflow.png", "wb") as f:
    f.write(png_bytes)
