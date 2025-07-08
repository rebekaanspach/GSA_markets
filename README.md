# Market GSA model

This repository is linked to the project ''**Uncertain markets: screening sources of uncertainty in life cycle assessment**''and provides a tool to screen market mixes in the Ecoinvent database. 


## Step 1: Filtering market mixes

The first step is to identify the market mixes to screen within the product system under analysis as explained in Section 2.4. 

First: 
 - enter the metadata of the product system under analysis in the excel file '`to_read.xlsx`. Indicate the name of the product system, its Ecoinvent identifier and the quantitiy required.

Then, filtering can be executed by running the juypter notebook `filtering_markets.ipynb` which runs `filtering_markets.py`. 
For this:
 - define a cut-off value as explained in Section 2.3.1 of the paper.

The output of this notebook is a list of markets (identified within the product system under analysis) saved in `markets/{market_name}_activities.csv` and `markets/{market_name}_indices.csv`.
## Step 2: Screening by ISIC categories

The second step does high-level screening. μ∗ scores are calculated for the highest level of aggregation of markets within the product system. Markets are grouped by their international standard industrial classification two-digit numbers (ISIC) as written in equation 11 of the paper. 
Screening by ISIC categories is done by running the jupyter notebook `GSA-morris-by-category.ipynb` which calls the script `morris_category_analysis.py`. `morris_category_analysis.py` uses the list of markets identified in Step 1 and saved in the `market` folder. The list of markets serve as the inputs, the x in equation 4, of the sensitivity analysis problem.

To run the notebook:
 - define the GSA Morris method's settings (number of levels and number of trajectories).

The output of this notebook is two excel files containing the sensitivity scores by ISIC category:
 -  `results/Global_{market_name}_category_scores.csv` can be used to plot a figure like Figure 4 of the paper. 
 -  `results/Global_{market_name}_category_scores_to_detailed_analysis.csv` is used in Step 3 to identify individual market in the most influencial ISIC categories.

## Step 3: Screening by markets
The third step calculates μ∗ scores for individual markets within the product system. The inputs of the sensitivity analysis problem are the individual markets in ISIC categories that had μ∗ scores above 6% as defined in equation 12. 
Screening by markets is done by running the jupyter notebook `GSA-morris-by-markets.ipynb` which calls the script `morris_markets_analysis.py`. `GSA-morris-by-markets.ipynb.ipynb` uses the excel file `results/Global_{market_name}_category_scores_to_detailed_analysis.csv` to identify the individual markets within the most influencial categories (μ∗ scores above 6%). Markets in other categories are not modelled uncertain to reduce the number of variables of the problem and keep computational time manageable.  

To run the notebook:
 - define the GSA Morris method's settings (number of level p and number of trajectories).

The output of this notebook is an excel file containing the sensitivity scores by markets.:
 - `results/Global_{market_name}_markets_scores.csv` can be used to plot a figure like Figure 5 of the paper. 

Figures are plotted in the `Figures-....ipynb` files.
