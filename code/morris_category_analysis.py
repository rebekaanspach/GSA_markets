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
import re
from collections import defaultdict
from SALib.sample.morris import sample
from SALib.analyze.morris import analyze
import time


def remove_parentheses(text):
        return re.sub(r'\(.*?\)', '', text)

def calculate_logratios(data):
        grouped_exchanges = defaultdict(list)
        for entry in data:
            grouped_exchanges[entry['group_market']].append(entry)

        log_ratios = {}
        for group, exchanges in grouped_exchanges.items():
            values = np.array([entry['value'] for entry in exchanges])
            log_ratios[group] = np.log(values[:-1] / values[-1])

        return log_ratios


def define_problem(input_data):
        result = {
            'groups_normalisation': [],
            'groups': [],
            'names': [],
            'num_vars': 0,
            'bounds': [],
            'sample_scaled': True
        }

        grouped_items = {}
        for item in input_data:
            group = item['group_market']
            if group not in grouped_items:
                grouped_items[group] = []
            grouped_items[group].append(item)

        filtered_data = []
        for group, items in grouped_items.items():
            if len(items) > 1:
                items.pop()  # Remove one item from each group
            filtered_data.extend(items)

        for item in filtered_data:
            name = f"{item['activity']}"
            result['groups'].append(item['category'])
            result['groups_normalisation'].append(item['group_market'])
            result['names'].append(name)
        
        result['num_vars'] = len(filtered_data)
        result['bounds'] = [[-3, 3]] * result['num_vars']
        return result


def ilr_inverse(ilr_coords):
        D = len(ilr_coords) + 1
        x = np.zeros(D)
        x[:-1] = np.exp(ilr_coords)
        x[-1] = 1
        return x / np.sum(x)

def perform_morris_category_analysis(fu, 
                                     amount,
                                     reference_product,
                                     reference_product_index,
                                     tier,
                                 d_label,  
                                 methods, 
                                 db, 
                                 trajectories, 
                                 num_levels,
                                 mean,
                                 market_dir="example"):
    

    all_demands= [{fu: amount}]
    all_demand_dict = all_demands[0].copy()
    for other_demand in all_demands[1:]:
        all_demand_dict.update(other_demand)

    indices = [1,1]

    clca = MyMonteCarloLCA(all_demand_dict, 
                           method=methods[0], 
                           final_activities=[],
                           database_name = db, 
                           tech_indices=None,
                           bio_indices=None,
                           include_only_specific_bio_uncertainty= True, 
                           include_only_specific_exc_uncertainty = True,
                           seed =False)
    next(clca)

    indices = pd.read_csv(f"markets_to_screen/{market_dir}_indices.csv")
    categories = pd.read_csv(f"markets_to_screen/{market_dir}_categories.csv")

    indices_list = list(indices.itertuples(index=False, name=None))
    duplicate_activities_removed = list(set(indices_list))

    activities =[]
    for i in range(len(duplicate_activities_removed)):
        db = bw.Database(duplicate_activities_removed[i][0])
        activity = db.get(duplicate_activities_removed[i][1])
        activities.append(activity)

    acti = [
        str(activity)  for activity in activities
    ]
    print(len(set(acti)))

    categories = categories.values.tolist()
    categories = [category[0] for category in categories ]

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
    print(len(set(acti)))

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

    
    print(exchanges)
    problem = define_problem(exchanges)




    category_mapping = {
    "01": "Crop and animal production, hunting and related service activities",
    "02": "Forestry and logging",
    "03": "Fishing and aquaculture",
    "05": "Mining of coal and lignite",
    "06": "Extraction of crude petroleum and natural gas",
    "07": "Mining of metal ores",
    "08": "Other mining and quarrying",
    "09": "Mining support service activities",
    "10": "Manufacture of food products",
    "11": "Manufacture of beverages",
    "12": "Manufacture of tobacco products",
    "13": "Manufacture of textiles",
    "14": "Manufacture of wearing apparel",
    "15": "Manufacture of leather and related products",
    "16": "Manufacture of wood and products of wood and cork",
    "17": "Manufacture of paper and paper products",
    "18": "Printing and reproduction of recorded media",
    "19": "Manufacture of coke and refined petroleum products",
    "20": "Manufacture of chemicals and chemical products",
    "21": "Manufacture of pharmaceuticals, medicinal chemical and botanical products",
    "22": "Manufacture of rubber and plastic products",
    "23": "Manufacture of other non-metallic mineral products",
    "24": "Manufacture of basic metals",
    "25": "Manufacture of fabricated metal products",
    "26": "Manufacture of computer, electronic, and optical products",
    "27": "Manufacture of electrical equipment",
    "28": "Manufacture of machinery and equipment",
    "29": "Manufacture of motor vehicles, trailers, and semi-trailers",
    "30": "Manufacture of other transport equipment",
    "31": "Manufacture of furniture",
    "32": "Other manufacturing",
    "33": "Repair and installation of machinery and equipment",
    "35": "Electricity, gas, steam and air conditioning supply",
    "36": "Water collection, treatment, and supply",
    "37": "Sewerage",
    "38": "Waste collection, treatment, and disposal activities",
    "39": "Remediation activities and other waste management services",
    "41": "Construction of buildings",
    "42": "Civil engineering",
    "43": "Specialized construction activities",
    "45": "Wholesale and retail trade and repair of motor vehicles and motorcycles",
    "46": "Wholesale trade, except of motor vehicles and motorcycles",
    "47": "Retail trade, except of motor vehicles and motorcycles",
    "49": "Land transport and transport via pipelines",
    "50": "Water transport",
    "51": "Air transport",
    "52": "Warehousing and support activities for transportation",
    "53": "Postal and courier activities",
    "55": "Accommodation",
    "56": "Food and beverage service activities",
    "58": "Publishing activities",
    "59": "Motion picture, video and television programme production, sound recording, and music publishing activities",
    "60": "Programming and broadcasting activities",
    "61": "Telecommunications",
    "62": "Computer programming, consultancy, and related activities",
    "63": "Information service activities",
    "68": "Real estate activities",
    "69": "Legal and accounting activities",
    "70": "Activities of head offices, management consultancy activities",
    "71": "Architectural and engineering activities, technical testing and analysis",
    "72": "Scientific research and development",
    "73": "Advertising and market research",
    "74": "Other professional, scientific, and technical activities",
    "75": "Veterinary activities",
    "77": "Rental and leasing activities",
    "78": "Employment activities",
    "79": "Travel agency, tour operator, and other reservation service and related activities",
    "80": "Security and investigation activities",
    "81": "Services to buildings and landscape activities",
    "82": "Office administrative, office support, and other business support activities",
    "84": "Public administration and defence, compulsory social security",
    "85": "Education",
    "86": "Human health activities",
    "87": "Residential care activities",
    "88": "Social work activities without accommodation",
    "90": "Creative, arts and entertainment activities",
    "91": "Libraries, archives, museums, and other cultural activities",
    "92": "Gambling and betting activities",
    "93": "Sports activities and amusement and recreation activities",
    "94": "Activities of membership organizations",
    "95": "Repair of computers and personal and household goods",
    "96": "Other personal service activities",
    "97": "Activities of households as employers of domestic personnel",
    "98": "Undifferentiated goods- and services-producing activities of households for own use",
    "99": "Activities of extraterritorial organizations and bodies"
}


    problem['groups'] = [
       category_mapping.get(group[:2], 'Unknown') for group in problem['groups']
    ]

    df = pd.DataFrame(problem)
    groups = df.groupby("groups_normalisation").agg({'groups': 'first'}).reset_index()
    count = groups['groups'].value_counts()
    

    morris_samples = sample(problem, trajectories, num_levels)

    

    X = []
    for alr_values in morris_samples:
        groups = problem['groups_normalisation']
        unique_groups = np.unique(groups)
        grouped_data = {group: alr_values[groups == group] for group in unique_groups}
        compositions = [ilr_inverse(grouped_data[group]) for group in unique_groups]
        all_compositions = np.hstack(compositions)
        X.append(all_compositions)

    X = np.array(X)
    X = X.T
    X_with_zeros = np.insert(X, 0, 0, axis=1)
    X_with_zeros = np.insert(X_with_zeros, 0, 0, axis=1)
    X_with_zeros = np.insert(X_with_zeros, 0, 0, axis=1)

    results = []
    was_negative = [exc['value'] < 0 for exc in exchanges]
    samples_with_negatives = [-s if was_negative[i] else s for i, s in enumerate(X_with_zeros)]

    new_samples = [
        {
            'indices': exc['tech_indices'],
            'activity': exc['activity'],
            'exchange': exc['exchange'],
            'exc_database': exc['database'],
            'act_database': exc['activity'][0],
            'sample': samples_with_negatives[j],
            'old_value': exc['value'],
            'index': i,
            'group_normalisation': exc['group_market'],
            'category' : exc['category']
        }
        for j, exc in enumerate(exchanges)
    ]

    results.extend(new_samples)
    all_results = pd.DataFrame(new_samples)
    

    presamples_data_dict = {}
    presamples_indices_dict = {}

    presamples_data = [] 
    presamples_indices = []  
    for exc in results:
        process = bw.Database(exc['exc_database']).get(exc['exchange'][1])
        activity = bw.Database(exc['act_database']).get(exc['activity'][1])
        value = exc['sample']

        data, indices = create_presamples(
            activity=activity,
            process=process,
            exchanges={"new_exchange": process, "new_value": lambda process: value}
        )

        presamples_data.append(data)
        presamples_indices.append(indices)

    presamples_data = np.vstack(presamples_data)
    presamples_data_dict = presamples_data
    presamples_indices_dict = presamples_indices

    index_matrix_data = [
        (presamples_data_dict, presamples_indices_dict, 'technosphere')
    ]

    index_id, index_path = ps.create_presamples_package(matrix_data=index_matrix_data)

    presamples = [
        {"runs": len(morris_samples), "name": "market mixes", "path": [index_path], "number": "all"},
    ]

    final_results = []
    for method in methods: 
        for presample in presamples: 
            start_time = time.time()
            clca = MyMonteCarloLCA(all_demand_dict, 
                                   method=method, 
                                   presamples=presample["path"],
                                   final_activities=[],
                                   database_name = db, 
                                   bio_indices=None,
                                   tech_indices=None,
                                   include_only_specific_bio_uncertainty=True, 
                                   include_only_specific_exc_uncertainty=True,
                                   seed=False)
            next(clca)
            temp_results = collect_results(clca, 
                                           all_demands, 
                                           [], 
                                           len(morris_samples),
                                           method,
                                           d_label, 
                                           [],
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
            final_results.append(temp_results)


  

    morris_results = pd.concat(final_results)

    #morris_results.to_csv(f"results//Global_{market_dir}_category_results.csv", index=False)

    SA = analyze( problem, morris_samples, np.array(morris_results['score']), conf_level=0.95,  print_to_console=False, num_levels=12
            )
    SA = pd.DataFrame(SA)

    runtime = morris_results['runtime'][0]


    lca = bw.LCA({db.get(f"{reference_product_index}"): amount}, method=methods[0])
    lca.lci()
    lca.lcia()

    mean_score= mean
    SA['trajectories'] = trajectories
    SA['reference_product'] = reference_product
    SA['tier'] = tier 
    SA['number of variables'] = len(set(acti))
    SA['runtime'] = runtime
    SA['number of runs'] = len(morris_samples)
    SA['score'] = mean_score
    SA['mu_star_relative'] = SA['mu_star']*100/mean_score
    SA['mu_star_conf_relative'] = SA['mu_star_conf']*100/mean_score
    count = count.reset_index(drop=True)
    SA = pd.concat([SA, count], axis=1)
    SA.to_csv(f"results/Global_{market_dir}_category_scores_{num_levels}_levels_{trajectories}_trajectory.csv", index=False)

    return SA
