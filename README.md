# REMISS Project: Fake News Analysis on Twitter

## Introduction

Welcome to the REMISS Project repository. This repository contains the metrics and data produced within the European project named REMISS, focused on analyzing fake news on Twitter. The goal of this project is to understand the dissemination and impact of fake news by examining various metrics and visualizing the data through different plots.

## Content Overview

The repository includes a variety of plots and analyses to provide insights into the behavior and propagation of fake news on Twitter. Below is an explanation of each type of plot available in the repository:

### Cascade CCDF

The Cascade Complementary Cumulative Distribution Function (CCDF) plot displays the probability of the size of information cascades (retweet chains) being greater than or equal to a certain value. This helps in understanding the spread and impact of fake news cascades.

### Cascade Count Over Time

This plot shows the number of cascades over a specific period, highlighting the temporal dynamics of fake news propagation on Twitter. It provides insights into peak periods of fake news dissemination.

### Average Emotion

The Average Emotion plot represents the overall emotional tone of tweets containing fake news. It uses sentiment analysis to classify emotions such as happiness, sadness, anger, and surprise, giving an overview of the emotional impact of fake news.

### Average Emotion Per Hour of the Day

This plot breaks down the average emotion of fake news tweets by the hour of the day. It helps in understanding how the emotional tone varies throughout the day, indicating possible patterns in user behavior.

### Tweets and Metrics Table

A comprehensive table listing tweets along with various metrics such as retweet count, likes, user engagement, and more. This table serves as a detailed dataset for further analysis and verification.

### Time Series Histograms

Time series histograms display the distribution of tweets over time, providing a clear visualization of tweet frequency and patterns. This can help in identifying trends and significant events related to fake news.

### User Profiling Plots

#### Tweeting Rate: Weekend vs Weekday

This plot compares the tweeting rate of users during weekends and weekdays. It highlights differences in user activity and engagement with fake news during different days of the week.

#### Tweeting Rate: Awake Hours vs Sleeping Hours

This plot compares the tweeting rate of users during awake hours (daytime) versus sleeping hours (nighttime). It provides insights into user behavior and the timing of fake news propagation.

### Automatic Multimodal Fact Checking

This section includes analyses and visualizations related to the automatic multimodal fact-checking system. It showcases how text, images, and other media are used to verify the authenticity of news and detect fake news.

### Cascade Propagation Analysis

#### Size

The Size plot examines the size of retweet cascades, indicating how far and wide fake news spreads across the Twitter network. It helps in understanding the reach and influence of fake news.

#### Depth

The Depth plot measures the maximum depth of retweet cascades, showing how many layers of retweets a single piece of fake news generates. This helps in understanding the hierarchical structure of the propagation.

#### Max-Breadth

The Max-Breadth plot shows the maximum breadth (the largest number of retweets at any level) of the cascades. It provides insight into the most active layers within the cascade.

#### Structural Virality

The Structural Virality plot quantifies how widely and evenly the information is shared within the network, rather than just how far it spreads. It considers the shape of the cascade, combining depth and breadth to measure the overall virality of fake news.

### User Reputation Analysis

This section focuses on analyzing user reputation based on various metrics such as legitimacy score, fakeness probability, reputation and status. It shows the network of users and their credibility in spreading fake news.

## Installation

To set up the project on your local machine, follow these steps:

### Step 1: Install MongoDB

1. **Download and Install MongoDB:**
   - Follow the instructions on the [MongoDB Download Center](https://www.mongodb.com/try/download/community) to download and install MongoDB for your operating system.

2. **Start MongoDB:**
   - Once installed, you can start the MongoDB service using the following command:
     ```bash
     mongod
     ```

### Step 2: Install Dependencies

1. **Clone the Repository:**
   - Clone this repository to your local machine using the following command:
     ```bash
     git clone https://github.com/your-username/REMISS-Project.git
     cd REMISS-Project
     ```

2. **Create a Virtual Environment (Optional but Recommended):**
   - It is recommended to create a virtual environment to manage dependencies. You can create a virtual environment using `venv` or `virtualenv`:
     ```bash
     python3 -m venv venv
     source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
     ```

3. **Install Dependencies:**
   - Install the necessary Python packages using `pip`:
     ```bash
     pip install -r requirements.txt
     ```

### Optional: Install Using Conda and environment.yaml

1. **Create a Conda Environment:**
   - If you prefer using Conda, you can create a new environment using the provided `environment.yaml` file:
     ```bash
     conda env create -f environment.yaml
     ```

2. **Activate the Conda Environment:**
   - Activate the newly created Conda environment:
     ```bash
     conda activate remiss-env
     ```

Now you have set up the project environment with all necessary dependencies and MongoDB installed. You are ready to start working with the REMISS Project.

## Extra Data

This project requires additional data for profiling and multimodal fact-checking. Follow the instructions below to properly configure and use this data.

### Profiling Data

The profiling data includes various configuration files that are essential for user profiling and analysis.

1. **Download the Profiling Data:**
   - Download the profiling data from the provided URL and unzip it into the root directory of the project.

2. **Configuration Files:**
   - Ensure that the configuration files are properly placed in the appropriate directories. These files contain necessary settings and parameters for running the profiling modules.

### Multimodal Data

The multimodal data includes images required for the multimodal fact-checking modules.

1. **Download the Multimodal Data:**
   - Download the multimodal data from the provided URL and unzip it into the root directory of the project.

2. **Images for Fact Checking:**
   - The images should be stored in the designated folder as specified in the configuration files. Make sure the paths in the configuration files point to the correct locations of these images.

### Configuration

1. **Unzip the Data:**
   - After downloading the data, unzip it into the root directory of the project:
     ```bash
     unzip profiling_data.zip -d .
     unzip multimodal_data.zip -d .
     ```

2. **Configure Paths:**
   - If necessary, configure the paths in the configuration files to match the locations where the data has been unzipped. Ensure that all modules can access the required data without any issues.

## Configuration

The REMISS Project uses a YAML configuration file to manage various settings and paths. Below is an example of a configuration file and an explanation of its sections:

### Example Configuration File (`config.yaml`)

```yaml
mongodb:
  host: localhost
  port: 27017

graph_layout: fruchterman_reingold

graph_simplification:
  threshold: 0.95

available_datasets:
  - test_dataset_2

theme: PULSE
debug: True
frequency: 1D

textual:
  api_url: 'http://srvinv02.esade.es:5005/api'

profiling:
  data_dir: './profiling_data/'

multimodal:
  data_dir: './multimodal_data/'

wordcloud:
  max_words: 50
  width: 400
  height: 400
  match_width: True
```

### Configuration Sections

- **mongodb:** 
  - `host`: The hostname for the MongoDB server.
  - `port`: The port number for the MongoDB server.

- **cache_dir:**
  - Directory for caching data.

- **graph_layout:**
  - Specifies the layout algorithm for graph visualization. In this case, it's set to `fruchterman_reingold`.

- **graph_simplification:**
  - `threshold`: The threshold value for simplifying the graph.

- **available_datasets:**
  - Lists the datasets available for analysis. Uncomment datasets as needed.

- **theme:**
  - The theme for the visualizations. Here, it is set to `PULSE`.

- **debug:**
  - A boolean flag to enable or disable debug mode.

- **frequency:**
  - The frequency for data analysis, set to `1D` for daily analysis.

- **textual:**
  - `api_url`: The API URL for the textual analysis service.

- **profiling:**
  - `data_dir`: Directory path for the profiling data.

- **multimodal:**
  - `data_dir`: Directory path for the multimodal data.

- **wordcloud:**
  - `max_words`: Maximum number of words in the word cloud.
  - `width`: Width of the word cloud image.
  - `height`: Height of the word cloud image.
  - `match_width`: A boolean flag to match the width.

### Setting Up the Configuration

1. **Create the Configuration File:**
   - Create a file named `config.yaml` in the root directory of the project and copy the example configuration into it.

2. **Edit the Configuration File:**
   - Modify the configuration settings as per your requirements, ensuring that the paths to the profiling and multimodal data are correctly specified.

3. **Use the Configuration File:**
   - Ensure that your project scripts and modules load and use the `config.yaml` file for their settings. You can use a YAML parser in your code to read and apply these settings.

## Usage

To use the REMISS Project application, you need to call the main script `app.py` with the path to your configuration file as an argument. This will start the server, which will be accessible at port 8050.

### Running the Application

1. **Start the Server:**
   - Run the application by executing the following command in your terminal:
     ```bash
     python app.py <config_filepath>
     ```
   - Replace `<config_filepath>` with the path to your `config.yaml` file.

2. **Access the Application:**
   - Open your web browser and go to `http://localhost:8050` to access the application.

