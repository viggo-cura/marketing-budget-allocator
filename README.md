# Real-Time Marketing Budget Allocator

A Streamlit application for optimizing marketing budget allocation across different channels based on ROAS (Return on Ad Spend) performance.

## Features

- Real-time budget allocation optimization
- Historical performance tracking
- Channel-specific ROAS targets
- Budget utilization monitoring
- Organic search revenue tracking
- Interactive visualizations
- 14-day performance forecasting

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

## Configuration

The application allows you to configure:
- Total weekly budget
- Channel-specific ROAS targets
- Initial budget allocation across channels

## Dependencies

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
