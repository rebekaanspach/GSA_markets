
import brightway2 as bw
import bw2calc as bc
import bw2data as bd
import numpy as np
import pandas as pd
import presamples as ps
import scipy
import scipy.stats as stats
ps.__version__



def make_dirichlet_distribution(tuples, concentration, mc):
    alpha = np.array([old_value for old_value in tuples])
    rv = stats.dirichlet(alpha * concentration)
    samples = []
    
    def _sample_dirichlet():
        for _ in range(mc):
            sample = rv.rvs(size=1)[0]
            samples.append(sample)

    _sample_dirichlet()  # Call the sampling function to fill the samples list

    return samples


def get_elec_input(process, elec):
    for exchange in process.technosphere():
        if exchange.input == elec:
            return exchange 

    raise RuntimeError("not found")


def create_presamples(activity, process, exchanges ):
    data = []
    indices = []

    new_exchange = exchanges["new_exchange"]
    new_value = exchanges["new_value"](process)
    indices_to_include = (process, activity, "technosphere")

    #indices.append(indices_to_include)
    data.append(new_value)
    return np.array(data), indices_to_include


def query_for_activities(
    database_name, 
    activities_to_query, 
    locations_to_query=None
):
    all_activities = []
    
    for db in database_name:
        activities = [
            act for act in bw.Database(db)
            if any(keyword in act['name'] for keyword in activities_to_query)
            and (locations_to_query is None or act['location'] in locations_to_query)
        ]
        all_activities.extend(activities)
    
    return all_activities