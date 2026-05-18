# PROJECT DESCRIPTION
This is a data science exploratory data analysis project. You are going to work with 2018 Yellow Taxi Trip Data (2018_Yellow_Taxi_Trip_Data_20260516.csv) which is 14.47 GB and has 112 million rows. Your task is to randomly sample 500K lines of the data first. Then, create plots and report as the output of your data exploratory analysis. 

# DATA EDA RULES
- Never load the full file. Pay attention to the data size.
- Use the lambda sampling approach in reading the csv file. Example: df = pd.read_csv(file, skiprows= lambda x: x != 0 and random.random() > 0.005). We are going to use 500K lines for the EDA.
- Use random.seed(42) for reproducibility and debugging purposes.
- Look for the missing values (null) or "N/A" values in the columns. Log the counts. If MORE THAN  40% of a COLUMN's values are null, drop that COLUMN entirely. If less than 40%, drop only the affected rows from your analysis. Always report how many rows are dropped  and/or column(s) are dropped as well (if applicable).
- If there are any long string columns, look for ways to simplify such as making those column binary if it is applicable or possible. 
- Look for the max, min, median values for the numeric columns, do the statistical analysis on them. 
- Look for the correlation, distributions, and outliers of data.  
- Generate a report with graphs.

# VISUALIZATION
- Use plotly for all standard EDA plots (histograms, scatter, box plots, correlation heatmaps, time series)
- For plotly plots: use width=1200, height=600, template="plotly_white"
- For any matplotlib/seaborn fallback plots: use figsize=(12,6), dpi=300, colormap="viridis"
- Use folium for any geographic coordinate visualizations (pickup/dropoff heatmaps, route patterns)
- For geographic visualizations, generate BOTH:
  1. Folium HeatMap of pickup AND dropoff density
  2. Folium scatter overlay for coordinate outliers (points outside NYC bounding box: lat 40.4774–40.9176, lon -74.2591–73.7004)
  - Flag and count coordinate outliers in the report
- All plotly plots must be saved as both PNG (/plots) and embedded as interactive HTML in the final eda_report.html
- All folium maps saved as standalone HTML in /plots folder

# CODE WRITING INSTRUCTIONS AND RULES
- Create virtual environment with Python version 3.12
- Following libraries are approved for the EDA and plotting: numpy, pandas, folium, plotly, random, matplotlib/seaborn and other necessary libraries. Include all the necessary packages into the requirement.txt file.
- You are going to use ruff as linting and formatting. Run: ruff check --fix . && ruff format .
- ruff rules to enforce: unused imports (F401), undefined names (F821), PEP8 line length max 88 chars (E501)
- When you create python scripts, always include what the code does with brief description and include input and output arguments are at the top. If applicable or necessary, add NOTES if the human user needs to pay attention to any important section of the script or the whole script itself. Always include version number of script file along with creation/update datetime. 
- Always include comments about what the code blocks are doing in the scripts you created.
- For global variables, use all caps definition.
- Do not include any API access keys and/or other secrets into the scripts. Include them into the .env file. 

# PROJECT BEHAVIOR RULES
- Follow this file naming convention:
  - Main EDA script: eda_main.py
  - Visualization script: eda_viz.py
  - Output report: eda_report.html
  - Plots folder: /plots
  - Error logs folder: /logs
  - Sampled data cache (if saved): data_sample.parquet
- After sampling, immediately save the 500K rows to data_sample.parquet as a cache. 
- On subsequent runs, check if data_sample.parquet exists first. If it exists, load it directly. Only re-sample from the CSV if data_sample.parquet is missing.
- Cache the dataset schema (column names, dtypes, shape) into data_schema.json after first load. Include column names, dtypes, shape, null counts, sample values to the schema. Reference this file in subsequent steps instead of re-reading the CSV.
- Create README.md file that includes:
  - Project overview and objective
  - Dataset description (source, size, row count)
  - How to run the EDA (step-by-step commands)
  - Output files and what each one contains
  - Python dependencies and how to install them (pip install -r requirements.txt)
- After sampling and statistical analysis are complete, spawn a subagent exclusively for plot generation. The subagent saves all plots to /plots folder as PNG files. Main agent waits for subagent to finish before assembling the final report.
- Configure a PostToolUse hook in .claude/settings.json to auto-run 'ruff check --fix . && ruff format .' after every Python file is written or edited.

# ERROR HANDLING
- All errors must be logged to /logs/error_log_{datetime}.txt with: timestamp, error type, affected script, and error message.
- Apply this 3-tier error response:
  - Auto-fixable errors (syntax, unused imports): fix automatically, log the fix, continue.
  - Data errors (missing column, bad dtype, empty result): log the error, skip that specific analysis step, continue with the rest.
  - Critical errors (file not found, MemoryError): log the error, STOP execution, report to user immediately.
- Repeating errors: if the same error occurs more than 2 times, mark that step as SKIPPED in the final report and move on. Do not retry more than 3 times.
