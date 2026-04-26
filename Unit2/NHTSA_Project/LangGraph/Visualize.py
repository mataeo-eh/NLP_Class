from Graph import app

png_bytes = app.get_graph().draw_mermaid_png()
with open("workflow.png", "wb") as f:
    f.write(png_bytes)