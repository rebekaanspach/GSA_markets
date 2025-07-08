import brightway2 as bw
import bw2calc as bc
import bw2data as bd
import numpy as np
import pandas as pd
import presamples as ps
import scipy
import scipy.stats as stats
from code.bw_helpers import sample_results, MyMonteCarloLCA, collect_results
from code.bw_helpers import print_recursive_calculation_to_list
from time import time


def filter_technosphere_exchanges(fu, method, cutoff, max_calc):
    """Use brightway's GraphTraversal to identify the relevant
    technosphere exchanges in a non-stochastic LCA."""
    start = time()
    res = bw.GraphTraversal().calculate(fu, method, cutoff, max_calc)
    print(f"cutoff: {cutoff}, max_calc: {max_calc}")

    # get all edges
    technosphere_exchange_indices = []
    for e in res['edges']:
        if e['to'] != -1:  # filter out head introduced in graph traversal
            technosphere_exchange_indices.append((e['from'], e['to']))
    print('TECHNOSPHERE {} filtering resulted in {} of {} exchanges and took {} iterations in {} seconds.'.format(
        res['lca'].technosphere_matrix.shape,
        len(technosphere_exchange_indices),
        res['lca'].technosphere_matrix.getnnz(),
        res['counter'],
        np.round(time() - start, 2),
    ))
    return technosphere_exchange_indices


def get_lca(fu, method):
    """Calculates a non-stochastic LCA and returns a the LCA object."""
    lca = bw.LCA(fu, method=method)
    lca.lci()
    lca.lcia()
    print('Non-stochastic LCA score:', lca.score)

    # add reverse dictionaries
    lca.activity_dict_rev, lca.product_dict_rev, lca.biosphere_dict_rev = lca.reverse_dict()

    return lca


def get_activities(lca, indices, biosphere=False):
    """Get actual activity objects from indices.

    Returns
    -------
    activities : list of tuples
        List of (from_act, to_act) pairs
    """
    activities = []
    for i in indices:
        from_act = bw.get_activity(lca.biosphere_dict_rev[i[0]]) if biosphere else bw.get_activity(lca.activity_dict_rev[i[0]])
        to_act = bw.get_activity(lca.activity_dict_rev[i[1]])
        activities.append( to_act)

    return activities


def recursive_calc(fu, methods, cutoff,  max_calc, output_dir="presamples\glass"):
    """
    Run a Consequential Life Cycle Assessment (CLCA) analysis and export results.
    
    Parameters:
    project_name (str): Brightway2 project name.
    db_name (str): Name of the database containing the functional unit.
    fu_code (str): Code of the functional unit.
    methods (list): List of impact assessment methods.
    output_dir (str): Directory for saving output CSV files.
    """
    

    all_demand_dict = {fu: 1}

    lca= get_lca(all_demand_dict, methods[0])

    t_indices =filter_technosphere_exchanges(all_demand_dict, methods[0], cutoff= cutoff, max_calc= max_calc)

    activities = get_activities(lca, t_indices)
    
#    # Define databases
#    db_list = ['Anesthesia', 'Raw materials', 'Packaging', 'Instrumentation', 'AM HTO', 'AM HTO- jig steel', 
#               'UKR', 'CM HTO', 'Transport', 'Sterilisation']
    
#    # Run Monte Carlo LCA
#    clca = MyMonteCarloLCA(
#        all_demand_dict, 
#        method=methods[0], 
#        final_activities=[],
#        database_name=db_list, 
#        tech_indices=None,
#        bio_indices=None,
#        include_only_specific_bio_uncertainty=True, 
#        include_only_specific_exc_uncertainty=True,
#        seed=False
#    )
#    next(clca)
    
#    # Collect results
#    results = []
#    for method in methods:
#        result = print_recursive_calculation_to_list(
#            fu,
#            method,
#            amount=1,
#            max_level=max_level,
#            cutoff=cutoff,
#        )
#        results.append(result)
#        print(f"Finished processing method: {method}")
    
#    flattened_results = [item for sublist in results for item in sublist]
    
    indices=[]
    for result in activities:
        if 'market' in str(result):
            indice = (result[0],result[1])
            indices.append(indice)


    duplicate_activities_removed =list(set(indices))
    
    
    activities =[]
    for i in range(len(duplicate_activities_removed)):
        db = bw.Database(duplicate_activities_removed[i][0])
        activity = db.get(duplicate_activities_removed[i][1])
        activities.append(activity)
    
    categories = list(
    sublist[1] 
    for activity in activities 
    for sublist in activity['classifications'] 
    if sublist[0] == 'ISIC rev.4 ecoinvent'
    )

    categories_df = pd.DataFrame({
        'Category': categories,
        'Activity': [str(bw.Database(act[0]).get(act[1])) for act in activities]
    })
    
    # Convert to DataFrames
    activities_df = pd.DataFrame(activities)
    categories_df = pd.DataFrame({'Category': categories})
    indices_df =pd.DataFrame(indices)
    
    # Save to CSV
    activities_df.to_csv(f"markets_to_screen/{output_dir}_activities.csv", index=False)
    categories_df.to_csv(f"markets_to_screen/{output_dir}_categories.csv", index=False)
    indices_df.to_csv(f"markets_to_screen/{output_dir}_indices.csv", index=False)
    
    print("CLCA analysis complete. Results saved to", output_dir)
    #return activities
    
# Example usage:
# run_clca_analysis("UK-wood-clca", "cutoff-3.9.1", "3040de6dd4125bb9ee919cfbb750a6cd", methods)
