
# Patent Title Topic Modeling

This project focuses on topic modeling for patent titles using the BERTopic library. 
The goal is to extract meaningful topics from a collection of patent titles and analyze 
the underlying patterns and trends in the patent domain.

## Project Overview

The project started with the challenging task of collecting patent titles from the FIPS.ru website. 
The data collection process involved manual effort, including copying patent titles by hand and pasting them into a text file. 
Regular expressions were then used to extract the patent titles from the text file 
and format it as `pd.DataFrame` with the help of `get_df_from_txt` function from data_utils.py, which were stored as `.csv` in data/ directory.

The main analysis and modeling work were conducted in Jupyter Notebooks. 
Three notebooks were created for different patent domains: patents_edu.ipynb for education-related patents, 
patents_med.ipynb for medicine-related patents, and patents_tr_infr.ipynb 
for patents related to transportation and infrastructure.

The BERTopic library was utilized for topic modeling, which allows 
for the extraction of meaningful topics from the patent titles. 
The notebooks include data preprocessing, model training, 
topic extraction, and result visualization steps, but github does not render graphs created with plotly, 
so in order to see it you will need to clone the repo and run locally.

The results/ directory contains Excel files with the analysis results and extracted topic information. 
These files provide insights into the patterns and trends in patent titles within each domain.
Models are gitignored, since they are too big to host on github, 
but you can downlad it from [google drive](https://drive.google.com/drive/folders/19iAqw8tKcEZKHHhEfOwUI5pHLVK4gwKe?usp=sharing)
or using `download_models.sh` from results directory.

The utils/ directory includes utility modules for data handling, 
training BERTopic models, and visualizing the results. 
These modules provide reusable functions to streamline the analysis process and enhance code readability.

Please refer to the specific Jupyter Notebooks and utility modules for more detailed information on the implementation and analysis steps.


## Setup
To explore this project locally, follow these steps:

1. Clone the project repository.
2. Create a virtual environment with Python 3.9.17.
3. Run the command `pip install -r requirements.txt` to install the project's dependencies.
4. Install the ipykernel package by running `pip install ipykernel`.
5. Create a kernel linked to the virtual environment by running `python -m ipykernel install --user --name=myenv`.

These steps will ensure that you have the necessary dependencies installed and can run the Jupyter Notebooks 
within the project using the appropriate kernel.

Feel free to modify the instructions as needed based on your specific setup 
or any additional steps required to run the project successfully.


## Conclusion

This project demonstrates the application of topic modeling techniques using the 
BERTopic library for patent titles. By extracting meaningful topics, it enables the exploration 
of patterns and trends in different patent domains. The project's structure and utility modules 
contribute to efficient data handling, model training, and result visualization.

