# FRP Data Processing and Classification

## Overview

This repository contains a comprehensive data processing and machine learning pipeline for classifying Fiber-Reinforced Polymer (FRP) research papers into 12 distinct categories. The project leverages semantic embeddings and unsupervised clustering to automatically organize and categorize academic research abstracts from multiple query datasets.

## Project Description

The FRP Data Processing project combines multiple research query results into a unified database, processes the data through advanced NLP techniques, and classifies papers into meaningful research categories using machine learning. The final output is an organized Excel database with papers grouped by their research focus.

### Key Features

- **Data Aggregation**: Combines results from 6 different research queries into a single dataset
- **Data Cleaning**: Removes duplicates and standardizes the dataset
- **Semantic Embeddings**: Uses Sentence Transformers to generate high-dimensional embeddings for paper abstracts
- **Unsupervised Clustering**: Applies K-means clustering (12 clusters) to identify research categories
- **Keyword Extraction**: Extracts TF-IDF based keywords to characterize each cluster
- **Visualization**: Creates PCA-based 2D visualization of clustered documents
- **Structured Output**: Exports results to an organized Excel file with category-specific sheets

## Repository Structure

```
FRP data processing/
├── README.md                          # This file
├── data_processing.ipynb              # Data aggregation, cleaning, and embedding generation
├── data_classification.ipynb          # Clustering, classification, and output generation
├── Data/
│   ├── Query_01.csv                   # Query result dataset 1
│   ├── Query_02.csv                   # Query result dataset 2
│   ├── Query_03.csv                   # Query result dataset 3
│   ├── Query_04.csv                   # Query result dataset 4
│   ├── Query_05.csv                   # Query result dataset 5
│   ├── Query_06.csv                   # Query result dataset 6
│   └── data_with_embeddings.pkl       # Processed data with embeddings (generated)
├── Media/
│   └── Cluster Plot.png               # Visualization of clustered papers
└── Final Database.xlsx                # Final classified output (generated)
```

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Required Dependencies

Install the required packages using:

```bash
pip install pandas numpy scikit-learn sentence-transformers matplotlib openpyxl xlsxwriter
```

**Package Details:**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **scikit-learn**: Machine learning algorithms (K-means, PCA, TF-IDF)
- **sentence-transformers**: Semantic embedding model (all-MiniLM-L6-v2)
- **matplotlib**: Data visualization
- **openpyxl**: Excel file reading/writing
- **xlsxwriter**: Excel workbook creation

## Usage

### Step 1: Run Data Processing (`data_processing.ipynb`)

This notebook handles the initial data preparation:

1. **Load Data**: Reads all CSV files from the `Data/` directory
2. **Combine Datasets**: Concatenates all query results into a single DataFrame
3. **Remove Duplicates**: Eliminates duplicate records
4. **Generate Embeddings**: Uses Sentence Transformers to create semantic embeddings for each abstract
   - Model: `all-MiniLM-L6-v2` (384-dimensional embeddings)
   - Input: Paper abstracts
   - Output: Stored in `Abstract_Embeddings` column
5. **Save Processed Data**: Exports data with embeddings to `data_with_embeddings.pkl`

### Step 2: Run Classification (`data_classification.ipynb`)

This notebook performs clustering and creates the final organized database:

1. **Load Embeddings**: Reads the processed data with embeddings
2. **Normalize Embeddings**: Applies L2 normalization for consistent clustering
3. **K-means Clustering**: Clusters papers into 12 categories
   - n_clusters: 12
   - random_state: 42 (for reproducibility)
   - max_iter: 300
4. **Extract Keywords**: Uses TF-IDF to identify top 10 keywords per cluster
5. **Assign Categories**: Maps clusters to human-readable category names
6. **Visualize Results**: Creates 2D PCA visualization showing cluster distribution
7. **Export to Excel**: Generates organized Excel file with category-specific sheets

## Research Categories

The project classifies papers into the following 12 research categories:

| ID | Category |
|----|----------|
| 0 | Machine Learning and Data-Driven FRP Modeling |
| 1 | Bond Behavior and Reinforcement with FRP Bars |
| 2 | FRP in Bridge Engineering and Deck Systems |
| 3 | Flexural and Shear Strengthening of RC Beams Using FRP |
| 4 | Confinement and Axial Strengthening of Concrete Columns |
| 5 | FRP–Steel Joints, Adhesive Bonding and Interface Behavior |
| 6 | Carbon Fiber Composites: Mechanical Behavior and Matrix Systems |
| 7 | Natural Fiber and Bio-Based Composite Materials |
| 8 | Manufacturing and Machining of FRP Composites |
| 9 | Sandwich Composite Structures and Panel Stability |
| 10 | Seismic Retrofitting of Masonry Walls Using FRP |
| 11 | Damage, Fatigue, and Thermal Effects in Composite Laminates |

## Technical Workflow

### Data Flow Diagram

```
Query CSVs (1-6)
    ↓
[data_processing.ipynb]
    ├─ Load & Combine
    ├─ Remove Duplicates
    ├─ Generate Embeddings (Sentence Transformers)
    └─ Save as PKL
    ↓
[data_with_embeddings.pkl]
    ↓
[data_classification.ipynb]
    ├─ Normalize Embeddings
    ├─ K-means Clustering
    ├─ Extract Keywords (TF-IDF)
    ├─ Generate Visualizations
    └─ Export Excel
    ↓
[Final Database.xlsx]
[Cluster Plot.png]
```

### Key Technologies

1. **Sentence Transformers**: State-of-the-art semantic embeddings capturing meaning of abstract text
2. **K-means Clustering**: Unsupervised algorithm for grouping similar documents
3. **PCA (Principal Component Analysis)**: Dimensionality reduction for 2D visualization
4. **TF-IDF (Term Frequency-Inverse Document Frequency)**: Statistical method for identifying important keywords
5. **Pandas & NumPy**: Data manipulation and numerical operations

## Output Files

### 1. `data_with_embeddings.pkl`
- **Format**: Python pickle file
- **Contents**: Original data plus 384-dimensional semantic embeddings
- **Size**: Depends on input data
- **Purpose**: Intermediate file for classification step

### 2. `Final Database.xlsx`
- **Format**: Microsoft Excel workbook
- **Structure**: 12 sheets, one per research category
- **Contents**: Complete paper information (title, abstract, authors, etc.) organized by category
- **Columns**: Maintains all original data columns plus Cluster_ID and Category
- **Purpose**: Final organized database for research analysis

### 3. `Cluster Plot.png`
- **Format**: PNG image
- **Content**: 2D scatter plot showing cluster distribution
- **Axes**: PCA-reduced dimensions (2 components)
- **Colors**: Different colors represent different clusters
- **Purpose**: Visual validation of clustering quality

## Data Format

### Input CSV Files
Expected columns in source query CSV files:
- **Title**: Paper title
- **Abstract**: Paper abstract (used for embeddings)
- **Authors**: Author names
- Additional metadata fields

### Output Excel Format
Each sheet contains:
- All original columns from source CSVs
- **Cluster_ID**: Numeric cluster assignment (0-11)
- **Category**: Human-readable category name
- Rows organized by research category

## Configuration

### Adjustable Parameters

#### In `data_classification.ipynb`:

```python
# K-means Configuration
kmeans = KMeans(n_clusters=12, random_state=42, n_init=10, max_iter=300)
```

- **n_clusters**: Number of research categories (default: 12)
- **random_state**: Seed for reproducibility (default: 42)
- **n_init**: Number of initializations (default: 10)
- **max_iter**: Maximum iterations (default: 300)

#### TF-IDF Configuration
```python
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
```

- **max_features**: Maximum vocabulary size (default: 5000)
- **stop_words**: Language for stop word removal (default: 'english')

## Performance Considerations

- **Data Size**: Tested with ~3,000+ papers across 6 query datasets
- **Embedding Generation**: Approximately 2-5 minutes for typical dataset size
- **Clustering**: K-means completes in seconds to < 1 minute
- **Memory**: Requires ~1-2 GB RAM for typical datasets
- **Storage**: Output pickle file ~3-4x larger than original due to embeddings

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Solution: Run `pip install -r requirements.txt` or manually install packages listed above

2. **File Not Found Errors**
   - Ensure all Query CSV files are in the `Data/` directory
   - Check file paths match the notebook (e.g., `'Data/data_with_embeddings.pkl'`)

3. **Memory Issues**
   - Reduce data size or run on machine with more RAM
   - Process data in batches if needed

4. **Slow Embedding Generation**
   - Normal for first run; model needs to download (~100MB)
   - Subsequent runs use cached model
   - Consider GPU acceleration with `device='cuda'` in SentenceTransformer

5. **Excel Export Errors**
   - Ensure no special characters in category names
   - Check available disk space
   - Verify xlsxwriter is installed: `pip install xlsxwriter`

## Future Enhancements

- Add support for additional languages
- Implement hierarchical clustering for topic subcategories
- Add interactive visualization dashboard (Plotly)
- Implement active learning for manual refinement
- Add domain-specific embedding models
- Implement confidence scores for category assignments
- Add automated report generation

## Results Interpretation

### Understanding Cluster Quality

1. **Intra-cluster Cohesion**: Papers within each category should share similar research focus
2. **Inter-cluster Separation**: Different categories should address distinct research areas
3. **Keyword Relevance**: Top keywords should accurately represent category themes
4. **Visualization**: Well-separated clusters in PCA plot indicate good category distinction

### Category Assignment Confidence

- Papers are deterministically assigned to the nearest cluster center
- Proximity to cluster center indicates confidence level
- Papers equidistant from multiple clusters may be interdisciplinary

## Contact & Support

For questions or issues related to this project, please refer to the comments within each notebook or check the project documentation.

## License

This project processes research data for academic and research purposes. Ensure compliance with applicable copyright laws and data usage agreements for the source papers.

## Citation

If you use this categorization system for research, please cite:
```
FRP Data Processing and Classification Pipeline
Academic Year: 2025
```

## Changelog

- **v1.0** (December 2025): Initial release with 12-category classification system

---

**Last Updated**: December 2025
**Status**: Active
