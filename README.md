# A Hybrid Methodology for Forex Forecasting

## Abstract
We introduce a novel approach by applying a hybrid machine learning framework to the area of Forex market 
forecasting. Focusing on four major currency pairs—EUR/USD, GBP/USD, USD/JPY, and USD/CHF—from 2013 to 2023, our approach 
combines supervised learning algorithms with unsupervised HDBSCAN clustering. The framework employs a unique method of 
deriving targets from clustering results, using moving averages. Additionally, it encompasses a diverse range of supervised
models, from statistical to deep learning, from memory-less to memory-based, widening its applicability. The clustering 
is specifically conducted on an expansive dataset of 48 currency pairs to generate new features for the supervised models. 
Our evaluation employs multi-faceted metrics such as classification accuracy, financial returns, and risk-adjusted 
performance. Notably, the results reveal significant improvements in predictive accuracy, particularly with the GRU model,
while providing nuanced insights into the complexities of financial performance metrics. These findings underscore the 
importance of a more advanced approach to Forex trading that takes into account both risk management and market volatility.

## References
Arrese Rodriguez, J. (2023). *A Hybrid Methodology for Forex Forecasting*. MSc in Artificial Intelligence dissertation, The University of Bath.


## Table of Contents
1. [Execution Environment](#execution-environment)
2. [Configuration](#configuration)
3. [Execution Methods](#execution-methods)
   - [Optional Step: Data Cleanup](#preliminary-step)
   - [Option 1: Automated Notebook Execution](#execute-notebooks)
   - [Option 2: Automated Pipeline Execution](#execute-pipelines)

---

## Execution Environment <a name="execution-environment"></a>

The experiments in this study were conducted on a MacBook Pro with an M1 chip. The technical specifications of the system are as follows:

- **Processor**: Apple M1 chip with 10-core CPU and 16-core GPU
- **Memory**: 16 GB Unified Memory
- **Storage**: 512 GB SSD
- **Operating System**: macOS Ventura 13.4

### Software and Libraries

- **Python Version**: 3.9
- **Package Manager**: Pip

#### Required Python Packages

```
papermill==2.4.0
pandas==2.0.2
matplotlib==3.7.1
scikit-learn==1.2.2
tensorflow==2.13.0
tensorflow-macos==2.13.0
tensorflow-metal==1.0.0
plotly==5.15.0
tsfresh==0.20.1
hdbscan==0.8.29
ipykernel==6.23.3
```

## Configuration <a name="configuration"></a>
To set up the project, follow these steps from the project's root directory (e.g. `path/to/forex_forecasting`):

1. **Create and Activate Virtual Environment**
    ```bash
    python -m venv venv-forex-forecasting
    source venv-forex-forecasting/bin/activate  # macOS and Linux
    .\venv-forex-forecasting\Scripts\Activate  # Windows
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt  # Windows and Linux
    pip install -r requirements-macos.txt  # macOS
    ```

3. **Update Configuration File**
    - Open the `src/config.py` file in a text editor.
    - Locate the `PROJECT_ROOT_DIR` variable.
    - Replace its value with the absolute path to the project root directory:
        ```python
        PROJECT_ROOT_DIR = 'path/to/forex_forecasting'
        ```

4. **Set Script Permissions**
    ```bash
    find scripts -type f -name "*.sh" -exec chmod +x {} \;
    ```
   
5. **Create a New Kernel**
    ```bash
    python -m ipykernel install --user --name=venv-forex-forecasting
    ```

## Execution Methods <a name="execution-methods"></a>

There are two primary ways to execute the project, each with its own set of advantages. Regardless of the chosen method,
the results will be saved in the `reports` folder.

Before initiating either method, it is advisable to clean the `reports` and `processed` folders if they contain data from previous runs.
```bash
scripts/clean.sh
```
### Option 1: Automated Notebook Execution <a name="execute-notebooks"></a>

**Recommended**

This method automatically runs Jupyter notebooks in a sequential manner via a script. It is the recommended choice because notebooks offer rich contextual information, such as training summaries and visualisations. 

**Available Commands:**

```bash
# Standalone
scripts/notebooks/run_notebooks_standalone.sh

# Hybrid MA5
scripts/notebooks/run_notebooks_hybrid_MA5.sh

# Hybrid MA10
scripts/notebooks/run_notebooks_hybrid_MA10.sh

# Run All
scripts/notebooks/run_all_notebooks.sh
```

### Option 2: Automated Pipeline Execution <a name="execute-pipelines"></a>

This alternative runs code directly through pipelines, also triggered by a script. While it is more streamlined, it lacks the informative context provided by notebooks.

**Available Commands:**

```bash
# Standalone
scripts/pipelines/run_pipelines_standalone.sh

# Hybrid MA5
scripts/pipelines/run_pipelines_hybrid_MA5.sh

# Hybrid MA10
scripts/pipelines/run_pipelines_hybrid_MA10.sh

# Run All
scripts/pipelines/run_all_pipelines.sh
```

