
import gradio as gr
from search import search


def gradio_search(query):
    if not query.strip():
        return "Please enter a search query."

    results = search(query, top_k=5)

    output = ""

    for result in results:
        output += (
            f"### {result['rank']}. {result['title']}\n"
            f"**Similarity:** {result['score']:.4f}\n\n"
            f"{result['abstract']}\n\n"
            f"---\n\n"
        )

    return output


demo = gr.Interface(
    fn=gradio_search,
    inputs=gr.Textbox(
        label="Search CS Papers",
        placeholder="e.g. deep learning for cyber attack detection"
    ),
    outputs=gr.Markdown(),
    title="CS Semantic Search Engine",
    description="Search 30,000 computer science papers using semantic similarity."
)


if __name__ == "__main__":
    demo.launch()
