import brightway2 as bw
import bw2calc as bc
import bw2data as bd
import numpy as np
import pandas as pd
import presamples as ps
import scipy
import scipy.stats as stats
from code.functions import make_dirichlet_distribution, get_elec_input, create_presamples, query_for_activities
from code.bw_helpers import sample_results, MyMonteCarloLCA, collect_results
from collections import defaultdict
from SALib.analyze.morris import analyze
import math
import re
import copy
import random
import time


def calculate_logratios(data):
        grouped_exchanges = defaultdict(list)
        for entry in data:
            grouped_exchanges[entry['group_market']].append(entry)

        log_ratios = {}
        for group, exchanges in grouped_exchanges.items():
            # Calculate the number of exchanges for each 'activity'
            activity_counts = {}
            # Update the values to be 1 divided by the number of exchanges for the same 'activity'
            for entry in exchanges:
                activity_counts[entry['activity']] = activity_counts.get(entry['activity'], 0) + 1
            

            values = np.array([1/activity_counts[entry['activity']] for entry in exchanges])
            log_ratios[group] = np.log(values[:-1] / values[-1])

        return log_ratios


def remove_parentheses(text):
        return re.sub(r'\(.*?\)', '', text)


def ilr_inverse(ilr_coords):
    """Inverse ALR transformation."""
    D = len(ilr_coords) + 1  # Since ALR uses D-1 coordinates
    x = np.zeros(D)  # Array to store the original composition
    
    # Exponentiate the ALR coordinates and set the last component as 1
    x[:-1] = np.exp(ilr_coords)
    x[-1] = 1 # The last component is the reference, so we don't need to change it
    
    # Normalize the components to sum to 1
    return x / np.sum(x)


def define_problem(input_data):
    # Initialize the output structure
    result = {
        'groups_normalisation': [],
        'groups': [],
        'names': [],
        'num_vars': 0,
        'bounds': [],
        'sample_scaled': True
    }

    # Step 1: Group items by 'group_market'
    grouped_items = {}
    for item in input_data:
        group = item['group_market']
        if group not in grouped_items:
            grouped_items[group] = []
        grouped_items[group].append(item)

    # Step 2: Remove one item per group
    filtered_data = []
    for group, items in grouped_items.items():
        if len(items) > 1:
            items.pop()  # Remove last item from each group
        filtered_data.extend(items)  # Keep the remaining items
    
        
    for item in filtered_data:
        name = f"{item['activity']}" 
        result['groups'].append(item['category'])
        result['groups_normalisation'].append(item['group_market'])
        result['names'].append(name)
        result['num_vars'] = len(filtered_data)
    

    log_ratios = calculate_logratios(input_data)
    
    category_bounds = []
    counter = 0
    for index, item in enumerate(filtered_data):
        group = item['group_market']
        

        
        if group in log_ratios:   
            bounds = [log_ratios[group][counter], log_ratios[group][counter]+0.00000000000001]

        
        if counter == len(log_ratios[group])-1:
            counter = 0
        else:
            counter = counter + 1

        category_bounds.append(bounds)

    result['bounds'] = category_bounds
    

    return result





def perform_local_morris_analysis(fu, 
                                  amount,
                                  reference_product,
                                 d_label,  
                                 methods, 
                                 db, 
                                 trajectories, 
                                 num_levels,
                                 category_names,
                                 category_numbers,
                                 market_dir= "example"):
    # Initialize the demand dictionary

    all_demands= [{fu: amount}]
    all_demand_dict = all_demands[0].copy()
    for other_demand in all_demands[1:]:
        all_demand_dict.update(other_demand)

    clca = MyMonteCarloLCA(all_demand_dict, 
                           method=methods[0], 
                           final_activities=[],
                           database_name=db, 
                           tech_indices=None,
                           bio_indices=None,
                           include_only_specific_bio_uncertainty=True, 
                           include_only_specific_exc_uncertainty=True,
                           seed=False)
    next(clca)

    indices = pd.read_csv(f"markets_to_screen/{market_dir}_indices.csv")
    categories = pd.read_csv(f"markets_to_screen/{market_dir}_categories.csv")

    indices_list = list(indices.itertuples(index=False, name=None))
    duplicate_activities_removed = list(set(indices_list))

    activities = []
    for i in range(len(duplicate_activities_removed)):
        db = bw.Database(duplicate_activities_removed[i][0])
        activity = db.get(duplicate_activities_removed[i][1])
        activities.append(activity)

    categories = categories.values.tolist()
    categories = [category[0] for category in categories]

    exchanges = MyMonteCarloLCA.get_exchanges_of_activity(clca, activities)
    exchanges_units_sorted = [exc for exc in exchanges if exc['exchange']['unit'] == exc['activity']['unit']]
    exchanges = [exc for exc in exchanges_units_sorted if exc['tech_indices'][0] != exc['tech_indices'][1]]




    exchanges_by_index = defaultdict(list)
    for exc in exchanges:
        second_index = exc["tech_indices"][1]
        exchanges_by_index[second_index].append(exc)


    to_delete = []
    to_delete_exc = []
    for second_index, group in exchanges_by_index.items():
        for exc in group:
            exchange_text = remove_parentheses(str(exc["exchange"]))
            activity_text = remove_parentheses(str(exc["activity"]))

            if (
                # (
                # ("market" in exchange_text and "market group" not in activity_text) or
                # ("generic market" not in exchange_text and "market group" not in activity_text)
                # ) or
                # ("treatment" in exchange_text and "waste" not in activity_text 
                #    and "electricity" not in activity_text
                #    and "heat" not in activity_text
                #    and "refinery" not in activity_text
                #    and "copper" not in activity_text
                #    and "biogas" not in activity_text
                #    ) or
                 ("tap water" in exchange_text  and "tap water" not in activity_text) or
                 ("tap water" in exchange_text  and "tap water" not in activity_text) or
                 ("municipal solid waste" in exchange_text  and "municipal solid waste" not in activity_text) or
                (exc["value"] > 0) != (group[0]["value"] > 0)):  
                to_delete.append(exc["tech_indices"])
                to_delete_exc.append([exc["exchange"], exc["activity"]])
            
            #if exc['value'] == 1:
            #    to_delete.extend([other["tech_indices"] for other in group if other["tech_indices"][1] == exc["tech_indices"][1]])

    #print(to_delete_exc)
    exchanges_sorted = [
        exc for exc in exchanges 
        if exc['tech_indices'] not in to_delete
    ]


    exchanges = []
    second_indices = [exc["tech_indices"][1] for exc in exchanges_sorted]

    for exc in exchanges_sorted:
        if second_indices.count(exc["tech_indices"][1]) != 1:
            exchanges.append(exc)
 #       else:
             #print(exc["activity"])


    




    acti = [
        str(exc['activity'])  for exc in exchanges 
    ]
    #print(len(set(acti)))



    categories = []
    for act in list(set(activities)):
        for sublist in dict(act)['classifications']: 
            if sublist[0] == 'ISIC rev.4 ecoinvent':
                categories.append(sublist[1])

    activity_to_group = {}
    group_counter = 0
    for entry in exchanges:
        activity = entry['activity']
        if activity not in activity_to_group:
            activity_to_group[activity] = group_counter
            group_counter += 1
        entry['group_market'] = activity_to_group[activity]

    
    problem = define_problem(exchanges)


    category_mapping = {
        "01": "Crop and animal production, hunting and related service activities",
        "02": "Forestry and logging",
        "05": "Mining of coal and lignite",
        "06": "Extraction of crude petroleum and natural gas",
        "07": "Mining of metal ores",
        "08": "Other mining and quarrying",
        "09": "Mining support service activities",
        "10": "Manufacture of food products",
        "16": "Manufacture of wood and products of wood and cork",
        "17": "Manufacture of paper and paper products",
        "19": "Manufacture of coke and refined petroleum products",
        "20": "Manufacture of chemicals and chemical products",
        "22": "Manufacture of rubber and plastic products",
        "23": "Manufacture of other non-metallic mineral products",
        "24": "Manufacture of basic metals",
        "25": "Manufacture of fabricated metal products",
        "26": "Manufacture of computer, electronic, and optical products",
        "27": "Manufacture of electrical equipment",
        "28": "Manufacture of machinery and equipment",
        "29": "Manufacture of motor vehicles, trailers, and semi-trailers",
        "30": "Manufacture of other transport equipment",
        "35": "Electricity, gas, steam and air conditioning supply",
        "37": "Sewerage",
        "38": "Waste collection, treatment, and disposal activities",
        "41": "Construction of buildings",
        "42": "Civil engineering",
        "43": "Specialized construction activities",
        "45": "Wholesale and retail trade and repair of motor vehicles and motorcycles",
        "46": "Wholesale trade, except of motor vehicles and motorcycles",
        "49": "Land transport and transport via pipelines",
        "50": "Water transport",
        "82": "Office administrative, office support, and other business support activities"
    }


    problem['groups'] = [category_mapping.get(group[:2], 'Unknown') for group in problem['groups']]

    market_to_evaluate = [name for group, name in zip(problem['groups'], problem['names']) if group in category_names]

    #exchanges = [exc for exc in exchanges if str(exc['category'])[:2] in category_numbers]
    

    df = pd.DataFrame(problem)

    #df = df[df['groups'].isin(category_names)]

    df['groups'] = df['names']
    df['category'] = problem['groups']
    problem = df.to_dict(orient="list")

    problem['num_vars'] = len(problem['names'])



    


    updated_problem = {}

    
    unique_groups = list(set(market_to_evaluate))
   # print(len(unique_groups))
    updated_problem = {}

    for group in unique_groups:
        # Create a new bounds list with [-3, 3] for the current group
        problem_with_new_bounds = [[-3, 3] if grp == group else bound for grp, bound in zip(problem['groups'], problem['bounds']) ]
        problem_with_new_group_names = [group if bound == [-3, 3] else "other" for grp, bound in zip(problem['groups'], problem_with_new_bounds)]
    


        # Identify indices where bounds are not [-3, 3]
        indices = [i for i, bound in enumerate(problem_with_new_group_names)]

        # Find one group different from the current group
        #different_group = next(grp for i, grp in enumerate(problem['groups']) if i in filtered_indices and grp != group)

        # Get all indices of that different group
        #keep_indices = [i for i in filtered_indices if problem['groups'][i] == different_group]
        #print(f"All indices of group {different_group} different from {group}: {keep_indices}")

        # Finalize bounds: Keep selected indices and [-3, 3], set others to None
        #final_bounds = [bound if i in keep_indices or bound == [-3, 3] else None for i, bound in enumerate(new_bounds)]

        # Get valid indices (non-None bounds)
        #valid_indices = [i for i, bound in enumerate(final_bounds)]

        # Create a new problem dictionary with only necessary keys
        new_problem = {
            key: [problem[key][i] for i in indices]
            for key in {'names', 'bounds', 'groups', 'groups_normalisation'} if key in problem
        }

        new_problem['num_vars'] = len(problem_with_new_group_names)
        new_problem['bounds'] = [i for i in problem_with_new_bounds] 
        new_problem['groups'] = [i for i in problem_with_new_group_names] # Assign filtered bounds

        updated_problem[group] = new_problem
    #print(set(updated_problem[unique_groups[0]]))

    groupings = []
    for category in unique_groups:
        groupings.append(list(set(updated_problem[category]['groups'])))
    #print(groupings)

    #print(updated_problem['Manufacture of basic metals'])

    print('finish update problem')
 

    from SALib.sample.morris import sample

    morris_samples = {}
    for category, sub_problem in updated_problem.items():
        morris_samples[category] = sample(sub_problem, trajectories, num_levels)

    X_dict = {}
    for category, samples in morris_samples.items():
        X = []
        for alr_values in samples:
            groups = updated_problem[category]['groups_normalisation']
            unique_groups = np.unique(groups)
            grouped_data = {group: alr_values[groups == group] for group in unique_groups}

            compositions = []
            for group in unique_groups:
                compositions.append(ilr_inverse(grouped_data[group]))  
            all_compositions = np.hstack(compositions)
            X.append(all_compositions)

        X = np.array(X)
        X = X.T  

        X_with_zeros = np.insert(X, 0, 0, axis=1)  
        X_with_zeros = np.insert(X_with_zeros, 0, 0, axis=1) 
        X_with_zeros = np.insert(X_with_zeros, 0, 0, axis=1) 

        X_dict[category] = X_with_zeros



    for exchange in exchanges:
        #print(exchange["category"])
        exchange["category"] = category_mapping.get(exchange["category"][:2], "Unknown")



    print('finish lookup')
    
    results = {}

    for index, (category, sample) in enumerate(X_dict.items()):
        # Filter exchanges based on category matching selected dual groups
        # Check if the length of filtered exchanges matches the length of exchanges
        samples_with_negatives = [
            {
                'indices': exc['tech_indices'],
                'activity': exc['activity'],
                'exchange': exc['exchange'],
                'exc_database': exc['database'],
                'act_database': exc['activity'][0],
                'sample': -sample[j] if exc['value'] < 0 else sample[j],
                'old_value': exc['value'],
            }
            for j, exc in enumerate(exchanges) 
        ]
        results[category] = samples_with_negatives




    #print(results['Electricity, gas, steam and air conditioning supply'])

    presamples_data_dict = {}
    presamples_indices_dict = {} 

    for category, result in results.items():
        presamples_data_dict[category] = [] 
        presamples_indices_dict[category] = []  

        for exc in result:
            process = bw.Database(exc['exc_database']).get(exc['exchange'][1])
            activity = bw.Database(exc['act_database']).get(exc['activity'][1])
            value = exc['sample']

            data, indices = create_presamples(
                activity=activity,
                process=process,
                exchanges={
                    "new_exchange": process,
                    "new_value": lambda process: value
                }
            )

            presamples_data_dict[category].append(data)
            presamples_indices_dict[category].append(indices)

        presamples_data_dict[category] = np.vstack(presamples_data_dict[category])

    index_matrix_data = {}
    for market, samples in results.items():
        index_matrix_data = [
            (presamples_data_dict[market], presamples_indices_dict[market], 'technosphere')
        ]

        index_id, index_path = ps.create_presamples_package(matrix_data=index_matrix_data)

        globals()[f'index_id_{market}'] = index_id
        globals()[f'index_path_{market}'] = index_path

    presamples = [{"runs": len(morris_samples),
                   "name": market,
                   "path": [globals()[f'index_path_{market}']], 
                   "number": "all"} for market, samples in presamples_data_dict.items()]

    indices = []
    results = []
    final_activities = []
    component_order = []

    for presample in presamples: 
        start_time = time.time()
        clca = MyMonteCarloLCA(all_demand_dict, 
                               method=methods[0], 
                               presamples=presample["path"],
                               final_activities=final_activities,
                               database_name=db, 
                               bio_indices=None,
                               tech_indices=None,
                               include_only_specific_bio_uncertainty=True, 
                               include_only_specific_exc_uncertainty=True,
                               seed=False)
        next(clca)
        temp_results = collect_results(clca, 
                                       all_demands, 
                                       final_activities,
                                       len(morris_samples[presample['name']]),
                                       methods[0],
                                       d_label, 
                                       component_order, 
                                       database_name=db, 
                                       bio_indices=None,
                                       tech_indices=None,
                                       include_only_specific_bio_uncertainty=True, 
                                       include_only_specific_exc_uncertainty=True)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        temp_results["runtime"] = runtime  
        temp_results["runs"] = presample["runs"]
        temp_results["category"] = presample["name"]
        results.append(temp_results)

    morris_results = pd.concat(results)

    morris_results.to_csv(f"results//Local_{market_dir}_market_results.csv", index=False)

    SA_results = {}
    for market, samples in presamples_data_dict.items():
        LCIA = morris_results[morris_results['category'] == market]

        
        SA_normalized = analyze(updated_problem[market], morris_samples[market], np.array(LCIA['score']), 
                                conf_level=0.95,  
                                print_to_console=False, 
                                num_levels=num_levels)

        SA_results[market] = pd.DataFrame(SA_normalized)

    data = []
    for market in SA_results.keys():
        names_list = SA_results[market]['names'].tolist()

        if market in names_list:
            index = names_list.index(market)
            mu_star_value = SA_results[market]['mu_star'][index]
            mu_star_value_conf = SA_results[market]['mu_star_conf'][index]
            data.append({'names': market, 'mu_star': mu_star_value, 'mu_star_conf': mu_star_value_conf})

    
    lca = bw.LCA(all_demands[0], method=methods[0])
    lca.lci()
    lca.lcia()

    mean_score= lca.score
    df_filtered = pd.DataFrame(data)
    for market in df_filtered['names']:
        #mean_score = morris_results[morris_results['category'] == market]['score'].max()
        df_filtered['reference_product'] = reference_product
        df_filtered.loc[df_filtered['names'] == market, 'score'] = mean_score
        mu_star_value = df_filtered.loc[df_filtered['names'] == market, 'mu_star'].values[0]
        mu_star_value_conf = df_filtered.loc[df_filtered['names'] == market, 'mu_star_conf'].values[0]
        df_filtered.loc[df_filtered['names'] == market, 'mu_star_local_relative'] = mu_star_value * 100 / mean_score
        df_filtered.loc[df_filtered['names'] == market, 'mu_star_local_relative_conf'] = mu_star_value_conf * 100 / mean_score



    df_filtered.to_csv(f"results/Local_{market_dir}_market_scores.csv", index=False)
    
    return df_filtered




