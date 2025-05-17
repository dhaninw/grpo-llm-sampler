This project contains two Jupyter Notebooks for LLM inference experiments:
1.  `MoDE.ipynb`: Mixture-of-Dexperts (MoD) on a "Needle-in-a-Haystack" task.
2.  `proxy-tuning-aime-test.ipynb`: Proxy-Tuning for mathematical problems.

**GPU VRAM REQUIRED: At Least 16GB**
These notebooks use large models and are very demanding. A powerful GPU is essential. Expect long runtimes.

## File Placement

*   **For `MoDE.ipynb`:**
    1.  Place your **`my_book.txt`** (the haystack text) in the `/content/` directory of your Jupyter environment (e.g., if using Google Colab, upload it to the default `/content/` folder).
    2.  Place your preprocessed **`needle_set_hard.json`** (NOLIMA-Hard dataset) in the `/content/` directory.

*   **For `proxy-tuning-aime-test.ipynb`:**
    *   No specific input data files are needed beyond the prompts defined within the notebook.

## Running the Notebooks

1.  **Open the Notebooks**: Use Jupyter Lab, Jupyter Notebook, or Google Colab.
2.  **Dependencies**:
    *   Installed in the first few cells
3.  **Execute Cells**: Run all cells in order from top to bottom.
    *   Model downloading and inference steps will take a significant amount of time.

**Output:**
*   **`MoDE.ipynb`**: Will print progress and results to the console.
*   **`proxy-tuning-aime-test.ipynb`**: Will print progress and generated math solutions. Results are saved to `aime2025.txt` in the notebook's working directory. 

**Key Things to Note:**
*   **Long Runtimes:** Be patient, especially during model downloads and the main processing loops. For initial testing, consider reducing the number of test cases in `qwen3.ipynb` (e.g., by modifying `CONTEXT_LENGTHS_TO_TEST` or the slice of `nolima_hard_configs_loaded`).
*   **Batching:** It is easy to run out of VRAM, so you will have to reduce the batch size. This is not done automatically. 
