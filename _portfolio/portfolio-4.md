---
title: "Spotify Data Analysis"
excerpt: "Spotify is a service incredibly popular and widely used that provides digital music from record labels such as Sony, EMI, Warner Music Group and Universal. The purpose of this project is to apply Data Science techniques with Machine Learning algorithms to study this application. We will show below the results of our research on the data-set and analyse it by generate a predicting function given others variables."
header:
    teaser: "https://wayiok.github.io/academicpages.github.io/images/portfolio/p4.png"
repo: "https://github.com/wayiok/Spotify-Data-Analysis"
collection: portfolio
---

# Introduction

Spotify is a service incredibly popular and widely used that provides digital music from record labels such as Sony, EMI, Warner Music Group and Universal. The purpose of this project is to apply Data Science techniques with Machine Learning algorithms to study this application. We will show below the results of our research on the data-set and analyse it by generate a predicting function given others variables.

# Overall Description

## Columns

**Table 1: General features in data set**

| Column | Type | Description |
|---|---|---|
| Acousticness | Percentage | Acoustic measure |
| Danceability | Percentage | Suitable for dancing |
| Duration | Miliseconds | Track Duration |
| Energy | Percentage | Intensity and activity |
| Explicit | Binary | Lack of ambiguity |
| Instrumentalness | Percentage | Vocals in track |
| Key | Number | Overall key |
| Mode | Binary | Mode modality |
| Liveness | Percentage | Audience presence |
| Loudness | Percentage | Loudness in decibels |
| Popularity | Number | Rank in 100 |
| Speechiness | Percentage | Spoken words degree |
| Tempo | Number | Beats per minute |
| Valence | Percentage | Musical positiveness |
| Year | Time | Year of the registry |

The original dataset gotten from spotify, a music streaming service, has the information of 100 years of songs in a .csv format, specifically from the year 1921 until 2020. Using the python programming language as hour main tool of data analysis, the description of the dataset can be resumed as showed in Table 1.

## Binary Features

The first analysis, as showed in Figure 1, was made by plotting the total percentage of explicitness in the songs classified as fashionable. Considering the explicitly percentage in both pie charts, we can infer that explicitness is not a mayor factor in most of the tracks inside this data.

![Result](https://wayiok.github.io/academicpages.github.io/images/portfolio/p4-1.png)

## Percentage Features

The second analysis executed was by getting the mean values of the features that go from a range between 0 and 1. This values can be seen on Figure 2.

![Result](https://wayiok.github.io/academicpages.github.io/images/portfolio/p4-2.png)

This figure presents an interesting behaviour that all tracks share, mean values between the years 1921 until 1960 have several fluctuation points. After the year 1961, we can notice that the fall or rise of the percentage features have steadily changes. This seems to suggest that data after 1960 is more reliable than the data before, simply because the variance between the means can be seen graphically and are better plotted in this overall analysis.

# Analysis

Several plots were generated for finding the changes of the features over the years and we succeeded on finding some remarkable impacts.

## Interesting changes of the features over the years

![Result](https://wayiok.github.io/academicpages.github.io/images/portfolio/p4-3.png)

This figure point out that the popularity of the tracks have a huge increased number throughout the years, starting from 1960s and climax around 2020s.

![Result](https://wayiok.github.io/academicpages.github.io/images/portfolio/p4-4.png)

Others feature such as Acousticness, Energy, Instrumentalness, etc. that either increase or decrease over the years. This means we can locate the release year of a track based on its others features.

## Predicting algorithm

As a result, we have create an algorithm to predict the released year of a track. (Details of the code is in the Jupyter Notebook)

![Result](https://wayiok.github.io/academicpages.github.io/images/portfolio/p4-5.png)

This figure shows that the accuracy of the algorithm is quite high and can be trusted. It may not determine the exact year of the track but it can narrow down the period of time of the release. We believe with some more time and information, we can improve the algorithm.

# References

- https://dorazaria.github.io/machinelearning/spotify-popularity-prediction/
- https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify
- https://www.music-tomorrow.com/blog/how-spotify-recommendation-system-works-a-complete-guide-2022
- https://artists.spotify.com/help/article/loudness-normalization
- https://medium.com/geekculture/analyzing-punk-sub-genres-using-the-spotify-api-cacaa95de4f4
- https://rpubs.com/puneet1193/832604



