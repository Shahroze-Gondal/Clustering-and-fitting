from sklearn.metrics import silhouette_score
import sklearn.cluster as cluster
import cluster_tools as ct
import errors as err
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '1'
pd.options.mode.chained_assignment = None


def file_reader(location):
    """
    Reads a CSV file and returns a pandas DataFrame.

    Parameters:
    ------------
    location (str): The filename of the CSV file to be read.

    Returns:
    ---------
    df (pandas.DatFrame): The DataFrame containing the data
    read from the CSV file.
    """
    address = location
    df = pd.read_csv(address, skiprows=4)
    df = df.drop(
        columns=[
            'Country Code',
            'Indicator Name',
            'Indicator Code',
            'Unnamed: 67'])
    return df


def fiting_and_forecast(df, Country_name, Ind, title, tit_fore):
    """
    Fiting the values and making forcast

    Parameters
    ----------
    df : Pandas Dataframe
        DESCRIPTION.
    Country_name : String
        Country Name.
    Ind : String
        Indicator name.
    title : String 
        Title of Graph.
    tit_fore : String
        Title of forcast.

    Returns
    -------
    None.

    """
    # fit exponential growth
    popt, pcorr = opt.curve_fit(exp_growth, df.index, df[Country_name],
                                p0=[4e3, 0.001])
    # much better
    df["pop_exp"] = exp_growth(df.index, *popt)
    plt.figure()
    plt.plot(df.index, df[Country_name], label="data")
    plt.plot(df.index, df["pop_exp"], label="fit")
    plt.legend()
    plt.xlabel('Years')
    plt.ylabel(Ind)
    plt.title(title)
    plt.savefig('Data.png', dpi=300)
    years = np.linspace(1995, 2030)
    pop_exp = exp_growth(years, *popt)
    sigma = err.error_prop(years, exp_growth, popt, pcorr)
    low = pop_exp - sigma
    up = pop_exp + sigma
    plt.figure()
    plt.title(tit_fore)
    plt.plot(df.index, df[Country_name], label="data")
    plt.plot(years, pop_exp, label="Forecast")
    # plot error ranges with transparency
    plt.fill_between(years, low, up, alpha=0.5, color="y")
    plt.legend(loc="upper left")
    plt.xlabel('Years')
    plt.ylabel(Ind)
    plt.savefig('forecast.png', dpi=300)
    plt.show()


def Clean_and_filter_data(first_ind_name, Second_ind_name, df1, df2, Year):
    """
    Filtering out and cleaning the Data for Clustering

    Parameters
    ----------
    first_ind_name : string
        DESCRIPTION.
    Second_ind_name : string
        DESCRIPTION.
    df1 : Pandas DataFrame
        DESCRIPTION.
    df2 : Pandas Data Frame
        DESCRIPTION.
    Year : string
        DESCRIPTION.

    Returns
    -------
    cluster_dataframe : TYPE
        DESCRIPTION.

    """
    df1 = df1[['Country Name', Year]]
    df2 = df2[['Country Name', Year]]
    df = pd.merge(df1, df2,
                  on="Country Name", how="outer")
    df = df.dropna()
    df = df.rename(
        columns={
            Year +
            "_x": first_ind_name,
            Year +
            "_y": Second_ind_name})
    cluster_dataframe = df[[first_ind_name, Second_ind_name]].copy()
    return cluster_dataframe


def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * (t - 1990))
    return f


def Specific_country_data(df, country_name, start_year, end_year):
    """
    Get Data For the Specific Country.

    Parameters
    ----------
    df : Pandas DataFrame 
        DESCRIPTION.
    country_name : String 
        DESCRIPTION.
    start_year : int
        DESCRIPTION.
    end_year : int
        DESCRIPTION.

    Returns
    -------
    df : Pandas DataFrame
        DESCRIPTION.

    """
    # Taking the Transpose
    df = df.T
    df.columns = df.iloc[0]
    df = df.drop(['Country Name'])
    df = df[[country_name]]
    df.index = df.index.astype(int)
    df = df[(df.index > start_year) & (df.index <= end_year)]
    df[country_name] = df[country_name].astype(float)
    return df


def create_clusters(
        df,
        indicator1,
        indicator2,
        xlabel,
        ylabel,
        title,
        no_of_clusters,
        df_fit,
        df_min,
        df_max):
    """


    Parameters
    ----------
    df : Pandas DataFrame
        Data Frame.
    indicator1 : String
        Indicator 1.
    indicator2 : String
        Indicator 2.
    xlabel : String
        Xlabel Value.
    ylabel : String
        Ylabel Value.
    title : String
        Title of Graph.
    no_of_clusters : Int
        no of clusters.
    df_fit : Pandas DataFrame
        Normalize DataFrame.
    df_min : int 
        minimum Value.
    df_max : int 
        Maximum Value.

    Returns
    -------
    None.

    """
    nc = no_of_clusters  # number of cluster centres
    kmeans = cluster.KMeans(n_clusters=nc, n_init=10, random_state=0)
    kmeans.fit(df_fit)
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    plt.figure(figsize=(8, 8))
    # scatter plot with colours selected using the cluster numbers
    # now using the original dataframe
    scatter = plt.scatter(df[indicator1], df[indicator2], c=labels, cmap="tab10")
    # colour map Accent selected to increase contrast between colours
    # rescale and show cluster centres
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:, 0]
    yc = scen[:, 1]
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig('Clustering_plot.png', dpi=300)
    plt.show()


def silhouette_score_plot(data, max_clusters=10):
    """
    Evaluate and plot silhouette scores for different numbers of clusters.

    Parameters:
    - data: The input data for clustering.
    - max_clusters: The maximum number of clusters to evaluate.

    Returns:
    """

    silhouette_scores = []

    for n_clusters in range(2, max_clusters + 1):
        # Perform clustering using KMeans
        kmeans = cluster.KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10)
        cluster_labels = kmeans.fit_predict(data)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Plot the silhouette scores
    plt.figure()
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='s')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()


CO2_emissions_metric_tons_per_capita = file_reader(
    'CO2_emissions_metric_tons_per_capita.csv')
Renewable_energy_consumption_percentage = file_reader(
    'Renewable_energy_consumption_percentage.csv')
cluster_dataframe = Clean_and_filter_data(
    'Renewable_energy_consumption_percentage',
    'CO2_emissions_metric_tons_per_capita',
    Renewable_energy_consumption_percentage,
    CO2_emissions_metric_tons_per_capita,
    '2020')

df_fit, df_min, df_max = ct.scaler(cluster_dataframe)
silhouette_score_plot(df_fit, 12)
create_clusters(
    cluster_dataframe,
    'Renewable_energy_consumption_percentage',
    'CO2_emissions_metric_tons_per_capita',
    'Renewable energy consumption percentage',
    'CO2 emissions metric tons per capita',
    'CO2 Emissions vs Renewable Energy Consumption in 2020',
    2,
    df_fit,
    df_min,
    df_max)
df = Specific_country_data(
    CO2_emissions_metric_tons_per_capita,
    'China',
    1990,
    2020)
fiting_and_forecast(
    df,
    'China',
    'CO2_emissions_metric_tons_per_capita',
    "CO2 Emissions Metric Tons Per Capita In China 1990-2020",
    "CO2 Emissions Metric Tons Per Capita In China Forecast Untill 2030")
df = Specific_country_data(
    CO2_emissions_metric_tons_per_capita,
    'Germany',
    1990,
    2020)
fiting_and_forecast(
    df,
    'Germany',
    'CO2_emissions_metric_tons_per_capita',
    "CO2 Emissions Metric Tons Per Capita In Germany 1990-2020",
    "CO2 Emissions Metric Tons Per Capita In Germany Forecast Untill 2030")
df = Specific_country_data(
    Renewable_energy_consumption_percentage,
    'China',
    1990,
    2020)
fiting_and_forecast(
    df,
    'China',
    'Renewable_energy_consumption_percentage',
    "Renewable Energy Consumption Percentage In China 1990-2020",
    "Renewable Energy Consumption Percentage In China Forecast Untill 2030")
df = Specific_country_data(
    Renewable_energy_consumption_percentage,
    'Germany',
    1990,
    2020)
fiting_and_forecast(
    df,
    'Germany',
    'Renewable_energy_consumption_percentage',
    "Renewable Energy Consumption Percentage In Germany 1990-2020",
    "Renewable Energy Consumption Percentage In Germany Forecast Untill 2030")
