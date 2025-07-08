import numpy as np
import bw2calc as bc
import logging
import pandas as pd
import brightway2 as bw
from bw2calc.monte_carlo import IterativeMonteCarlo
import re
_log = logging.getLogger(__name__)

# From bw2calc
try:
    from pypardiso import spsolve
except ImportError:
    from scipy.sparse.linalg import spsolve

from bw2calc.matrices import MatrixBuilder
from bw2calc.matrices import TechnosphereBiosphereMatrixBuilder as TBMBuilder


class MyMonteCarloLCA(bc.MonteCarloLCA):
    """Smarter iterative solution when doing contribution analysis.

    The original MonteCarloLCA stores only one `self.guess`, so if switching
    between calculating for multiple activities (like in contribution analysis)
    it does not solve efficiiently.

    Here we store different guesses for different demand vectors.
    """

    def __init__(self, 
                 *args, 
                 final_activities, 
                 database_name, 
                 bio_indices,
                 tech_indices,
                 include_only_specific_bio_uncertainty=True, 
                 include_only_specific_exc_uncertainty=True,
                 **kwargs):
    
        self.final_activities = final_activities
        self.database_name = database_name
        self.bio_indices = bio_indices
        self.tech_indices = tech_indices
        self.include_only_specific_bio_uncertainty=include_only_specific_bio_uncertainty
        self.include_only_specific_exc_uncertainty=include_only_specific_exc_uncertainty
    
        super().__init__(*args, **kwargs)

        # Cache guesses by current demand vector
        self.guesses = {}

        # Set up self.lca
        self.lca = bw.LCA(demand=self.demand, method=self.method)
        self.lca.load_lci_data()

    def load_mean_data(self):
        self.load_lci_data()
        self.tech =  self.tech_params['loc']
        self.bio = self.bio_params['loc']
        if self.lcia:
            self.load_lcia_data()
            self.cf_rng = self.cf_params['loc']
        if self.weighting:
            self.load_weighting_data()
            self.weighting_rng = self.weighting_params['loc']
        #print('start loading presamples')
        if self.presamples:
            self.presamples.reset_sequential_indices()
            #print('stop loading presamples')
    

    def new_sample(self, 
                   database_name, 
                   bio_indices,
                   tech_indices,
                   include_only_specific_bio_uncertainty, 
                   include_only_specific_exc_uncertainty, ):
        """Get new samples like __next__ but don't calculate anything."""
        #exclude_tech_background

        if not hasattr(self, "tech"):
            self.load_mean_data()

        if include_only_specific_exc_uncertainty:
            #print("my_rebuild_technosphere_matrix_specific_exc")
            self.my_rebuild_technosphere_matrix_specific_exc(self.tech, tech_indices)
        else:
            #print("Calling rebuild_technosphere_matrix")
            self.rebuild_technosphere_matrix(self.tech_rng.next())


        if include_only_specific_bio_uncertainty:
            #print("Calling my_rebuild_biosphere_matrix")
            self.my_rebuild_biosphere_matrix_specific_exc(self.bio, bio_indices)
        else:
            #print("Calling rebuild_biosphere_matrix")
            self.rebuild_biosphere_matrix(self.bio_rng)

        
        if self.lcia:
            self.rebuild_characterization_matrix(self.cf_rng)
        if self.weighting:
            self.weighting_value = self.weighting_rng
        #print('start updating matrices with presample')
        if self.presamples:
            self.presamples.update_matrices()
            #print('update matrices with presample')


        
    def solve_linear_system(self):
        demand_sig = tuple(self.demand.keys())
        _log.debug("    Solve linear system: %s", demand_sig)
        guess = self.guesses.get(demand_sig)
        if not self.iter_solver or guess is None:
            _log.debug("      solving from scratch...")
            self.guesses[demand_sig] = guess = spsolve(
                self.technosphere_matrix, self.demand_array
            )
            _log.debug("      done")
            return guess
        else:
            _log.debug("      solving iteratively...")
            solution, status = self.iter_solver(
                self.technosphere_matrix,
                self.demand_array,
                x0=guess,
                atol="legacy",
                maxiter=1000,
            )
            _log.debug("      done (status %s)", status)
            if status != 0:
                _log.debug("      solving again from scratch...")
                self.guesses[demand_sig] = guess = spsolve(
                    self.technosphere_matrix, self.demand_array
                )
                _log.debug("      done")
                return guess
            else:
                return solution
            

    def my_rebuild_biosphere_matrix_specific_exc(self, vector, bio_indices):
        """
        same as for my_rebuild_technosphere_matrix_specific_exc
        """
        # Backup the original matrix for comparison
        #original_matrix = self.technosphere_matrix.copy()

        # Get group tech_indices from activity labels
      
        # Create a new temporary biosphere matrix
        new_matrix = MatrixBuilder.build_matrix(
            self.bio_params, self.biosphere_dict, self.activity_dict,
            "row", "col",
            new_data=TBMBuilder.fix_supply_use(self.bio_params, vector.copy())
        )

        # Iterate over the relevant row and column bio_indices and update only those in the original matrix
        if bio_indices is not None: 
            #print('building biosphere matrix')
            self.biosphere_matrix[bio_indices[:, 0], bio_indices[:, 1]] = new_matrix[bio_indices[:, 0], bio_indices[:, 1]]
    
        
   

    def get_db_bio_tech_indices(self, database_name):
        """Return a dictionary with the tech_indices of all products that come from the specified database."""
        tech_indices = {}
        label = 'background'
        
        # Find activity keys that come from the specified database
        for activity_key, activity_value in self.activity_dict.items():
            if any(activity_key[0] == db_name for db_name in database_name):
                activity_index = self.activity_dict.get(activity_key)
                if activity_index is not None:
                    # Find non-zero row tech_indices in the corresponding column of the technosphere matrix
                    product_tech_indices = self.biosphere_matrix[:, activity_index].nonzero()[0]
                    if label not in tech_indices:
                        tech_indices[label] = []
                    tech_indices[label].extend(product_tech_indices)
    
        # Return a flattened list of tech_indices
        tech_indices[label] = list(set(tech_indices[label]))
    
        return tech_indices
    


        # Convert sparse matrices to arrays for comparison
        original_matrix = original_matrix.toarray()
        technosphere_matrix = self.technosphere_matrix.toarray()
        #print(technosphere_matrix)
        # Find the differences between the arrays
        row_tech_indices, col_tech_indices = np.where(original_matrix != technosphere_matrix)

        # Convert row and column tech_indices to a tuple of coordinate pairs
        differences = list(zip(row_tech_indices, col_tech_indices))

    
    def query_for_activities(
            self, 
            database_name_markets, 
            activities_to_query,
    ):
        for db in database_name_markets:
            activities = [
                act for act in bw.Database(db)
                if any(keyword in act['name'] for keyword in activities_to_query)
            ]
            return activities
        


    def get_exchanges_of_activity_without_class(self, activities):
        row_col_info = []
        col_tech_indices = [self.get_activity_index(activity['code']) for activity in activities]
        for column_index in col_tech_indices:
            nonzero_rows = self.lca.technosphere_matrix[:, column_index].nonzero()[0]
            for row_index in nonzero_rows:
                exc_code = self.get_activity_by_index(row_index)
                exc_name = bw.Database(exc_code[0]).get(exc_code[1])
            
                col_activity_code = self.get_activity_by_index(column_index)
                col_activity_name = bw.Database(col_activity_code[0]).get(col_activity_code[1])
                col_activity_value = - self.get_value(row_index, column_index)
                col_activity_db= exc_code[0]

                dict = {
                'tech_indices': [row_index, column_index],
                'activity': col_activity_name,
                'exchange': exc_name,
                'database': col_activity_db,
                'value': col_activity_value,
            }
                row_col_info.append(dict)
        return row_col_info
        
            
    def get_exchanges_of_activity(self, activities):
        row_col_info = []
        col_tech_indices = [self.get_activity_index(activity['code']) for activity in activities]
        for column_index in col_tech_indices:
            nonzero_rows = self.lca.technosphere_matrix[:, column_index].nonzero()[0]
            for row_index in nonzero_rows:
                exc_code = self.get_activity_by_index(row_index)
                exc_name = bw.Database(exc_code[0]).get(exc_code[1])
            
                col_activity_code = self.get_activity_by_index(column_index)
                col_activity_name = bw.Database(col_activity_code[0]).get(col_activity_code[1])
                col_activity_value = - self.get_value(row_index, column_index)
                col_activity_db= exc_code[0]

                category = list(
                sublist[1] 
                for sublist in col_activity_name.get('classifications', [])
                if sublist[0] == 'ISIC rev.4 ecoinvent'
                )
               
                
                dict = {
                'tech_indices': [row_index, column_index],
                'activity': col_activity_name,
                'exchange': exc_name,
                'database': col_activity_db,
                'value': col_activity_value,
                'category': category[0]
            }
                row_col_info.append(dict)
        return row_col_info

    def get_exchanges_of_activity_wo_category(self, activities):
        row_col_info = []
        col_tech_indices = [self.get_activity_index(activity['code']) for activity in activities]
        for column_index in col_tech_indices:
            nonzero_rows = self.lca.technosphere_matrix[:, column_index].nonzero()[0]
            for row_index in nonzero_rows:
                exc_code = self.get_activity_by_index(row_index)
                exc_name = bw.Database(exc_code[0]).get(exc_code[1])
            
                col_activity_code = self.get_activity_by_index(column_index)
                col_activity_name = bw.Database(col_activity_code[0]).get(col_activity_code[1])
                col_activity_value = - self.get_value(row_index, column_index)
                col_activity_db= exc_code[0]

                category = list(
                sublist[1] 
                for sublist in col_activity_name.get('classifications', [])
                if sublist[0] == 'ISIC rev.4 ecoinvent'
                )
            
                
                dict = {
                'tech_indices': [row_index, column_index],
                'activity': col_activity_name,
                'exchange': exc_name,
                'database': col_activity_db,
                'value': col_activity_value,
                'category': category
            }
                row_col_info.append(dict)
        return row_col_info
    

    def my_rebuild_technosphere_matrix_specific_exc(
            self, 
            vector, 
            tech_indices ):
        """
        Rebuild the technosphere matrix to update the matrix of foreground activities.

        Args:
            * vector (array): 1-dimensional NumPy array with length (# of technosphere parameters), in the same order as `self.tech_params`.

        Doesn't return anything, but overwrites `self.technosphere_matrix`.
        """
        # Backup the original matrix for comparison
        #original_matrix = self.technosphere_matrix.copy()

      
        # Create a new temporary technosphere matrix
        new_matrix = MatrixBuilder.build_matrix(
            self.tech_params, self.activity_dict, self.product_dict,
            "row", "col",
            new_data=TBMBuilder.fix_supply_use(self.tech_params, vector.copy())
        )

        # Iterate over the relevant row and column tech_indices and update only those in the original matrix
        if tech_indices is not None:
            self.technosphere_matrix[tech_indices[:, 0], tech_indices[:, 1]] = new_matrix[tech_indices[:, 0], tech_indices[:, 1]]
            print(self.technosphere_matrix[19611, 19631])
       
            
        

    def get_value(self, row, col):
        for i in range(self.technosphere_matrix.indptr[row], self.technosphere_matrix.indptr[row + 1]):
            if self.technosphere_matrix.indices[i] == col:
                value = self.technosphere_matrix.data[i]
                return value
        return 0.0  # Value is 0 if not found

    
    def get_activity_by_index(self, index):
        """Get the activity corresponding to the given index."""
        for key, value in self.activity_dict.items():
            if value == index:
                return key
        return None
    
    def get_activity_index(self, act_key):
        """Get the activity corresponding to the given index."""
        for key, value in self.activity_dict.items():
            if key[1] == act_key:
                return value
        return None
    
    def get_tech_indices(self, act_key):
        """Get the activity corresponding to the given index."""
        for key, label in self.activity_dict.items():
            if key[1] == act_key:
                return key, label
        return None
    

    def get_product_by_index(self, index):
        """Get the activity corresponding to the given index."""
        for key, value in self.product_dict.items():
            if value == index:
                return key
        return None
    
    def get_activity_tech_indices_from_db(self, database_name):
        """Return a dictionary with the tech_indices of all products that come from the specified database."""
        tech_indices = {}
        label = 'background'
        
        # Find activity keys that come from the specified database
        for activity_key, activity_value in self.activity_dict.items():
            if any(activity_key[0] == db_name for db_name in database_name):
                activity_index = self.activity_dict.get(activity_key)
                if activity_index is not None:
                    # Store the activity index in the list corresponding to the label
                    if label not in tech_indices:
                        tech_indices[label] = []
                    tech_indices[label].append(activity_index)
        
        # Sort the tech_indices within the list
        tech_indices[label].sort()
        
        return tech_indices
    

    def get_activity_tech_indices(self, act_name):
        """Return a dictionary with the tech_indices of all products that come from the specified database."""
        tech_indices = {}
        label = 'background'
        
        # Find activity keys that come from the specified database
        for activity_key, activity_value in self.activity_dict.items():
            if activity_key[1] == act_name:
                activity_index = self.activity_dict.get(activity_key)
                if activity_index is not None:
                    # Store the activity index in the list corresponding to the label
                    if label not in tech_indices:
                        tech_indices[label] = []
                    tech_indices[label].append(activity_index)
        
        # Sort the tech_indices within the list
        tech_indices[label].sort()
        
        return tech_indices
    

    def get_db_tech_indices(self, database_name):
        """Return a dictionary with the tech_indices of all products that come from the specified database."""
        tech_indices = {}
        label = 'background'
        
        # Find activity keys that come from the specified database
        for activity_key, activity_value in self.activity_dict.items():
            if any(activity_key[0] == db_name for db_name in database_name):
                activity_index = self.activity_dict.get(activity_key)
                if activity_index is not None:
                    # Find non-zero row tech_indices in the corresponding column of the technosphere matrix
                    product_tech_indices = self.technosphere_matrix[:, activity_index].nonzero()[0]
                    if label not in tech_indices:
                        tech_indices[label] = []
                    tech_indices[label].extend(product_tech_indices)
    
        # Return a flattened list of tech_indices
        tech_indices[label] = list(set(tech_indices[label]))
    
        return tech_indices
    

    def get_group_tech_indices(self, activity_labels):
        print(f"get_group_tech_indices called with: {activity_labels}")
        #print(self.demand)
        """Return {label: tech_indices} from input {activity_key: label}"""
        tech_indices = {}
        seen_keys = {}
        for key, label in activity_labels.items():
            if key in seen_keys:
                raise ValueError(f"Key {key} ({label}) already mapped to {seen_keys[key]}")
            seen_keys[key] = label
            if label not in tech_indices:
                tech_indices[label] = []
            if key in self.activity_dict:
                index = self.activity_dict[key]
                print(index)
                tech_indices[label].append(index)
                #print(tech_indices)
        return tech_indices
    

    def divide_elements(self, row1, col1, row2, col2):
        """Divide the element at (row, col1) by the element at (row, col2)."""
        if not hasattr(self, 'technosphere_matrix') or self.technosphere_matrix is None:
            self.load_technosphere_matrix()
        value1 = self.technosphere_matrix[row1, col1]
        value2 = self.technosphere_matrix[row2, col2]
        if value2 == 0:
            raise ValueError("Division by zero error: The element at (row2, col2) is zero.")
        result = value1 / value2
        return result


def clean_name(name):
    # Replace any non-alphanumeric character (including spaces) with an underscore
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return clean_name

def print_recursive_calculation_to_list(
    activity,
    lcia_method,
    amount=1,
    max_level=3,
    cutoff=1e-2,
    results=None,
    tab_character="  ",
    level=0,
    lca_obj=None,
    total_score=None,
    first=True,
):
    """Traverse a supply chain graph, and calculate the LCA scores of each component. 
    Collects the result in a list of dictionaries.

    Args:
        activity: ``Activity``. The starting point of the supply chain graph.
        lcia_method: tuple. LCIA method to use when traversing supply chain graph.
        amount: int. Amount of ``activity`` to assess.
        max_level: int. Maximum depth to traverse.
        cutoff: float. Fraction of total score to use as cutoff when deciding whether to traverse deeper.
        results: list. Collects output in a list of dictionaries if provided.
        tab_character: str. Character to use to indicate indentation.

    Internal args (used during recursion, do not touch):
        level: int.
        lca_obj: ``LCA``.
        total_score: float.
        first: bool.

    Returns:
        list. Collected results as list.
    """
    
    if results is None:
        results = []

    if lca_obj is None:
        lca_obj = bc.LCA({activity: amount}, lcia_method)
        lca_obj.lci()
        lca_obj.lcia()
        total_score = lca_obj.score
    elif total_score is None:
        raise ValueError
    else:
        lca_obj.redo_lcia({activity: amount})
        if abs(lca_obj.score) <= abs(total_score * cutoff):
            return results

    # Create a dictionary with the relevant information
    result_entry = {
        "amount": float(amount),
        "activity": clean_name(str(activity)),
        "indices": activity[1],
        "level": level
    }
    results.append(result_entry)

    if level < max_level:
        for exc in activity.technosphere():                                        
            print_recursive_calculation_to_list(
                activity=exc.input,
                lcia_method=lcia_method,
                amount=amount * exc["amount"],
                max_level=max_level,
                cutoff=cutoff,
                results=results,
                tab_character=tab_character,
                lca_obj=lca_obj,
                total_score=total_score,
                level=level + 1,
            )
    #print(level)
    return results


def sample_results(
        lca, 
        demands, 
        final_activities, 
        database_name, 
        bio_indices,
        tech_indices,
        include_only_specific_bio_uncertainty=True,
        include_only_specific_exc_uncertainty=True, 
        **kwargs):
    #exclude_tech_background=True
    """Draw a sample from `lca` and do contribution analysis.

    `lca` must be an instance of `MyMonteCarlo`, already prepared for LCIA
    calculations. Each time this function is called, the technosphere matrices
    are updated with a new Monte Carlo sample, and the LCIA calculations are
    repeated for each of the demands in `demands`. 

    """
    # Update matrices from random number generator
    _log.debug("New sample...")
    lca.new_sample(database_name,
                   bio_indices,
                   tech_indices,
                   include_only_specific_bio_uncertainty=include_only_specific_bio_uncertainty,  
                   include_only_specific_exc_uncertainty= include_only_specific_exc_uncertainty, )
    _log.debug("done")
    #exclude_tech_background=exclude_tech_background,
    
   
    # Do the calculation for each demand vector
    results = []
    for demand in demands:
        # This is not ideal, computationally, since we are using an
        # iterative solver for the MC samples, then factorizing anyway
        # for the contribution analysis...

        if lca is None:
            _log.debug("  [initialising]")
            lca = bc.LCA({activity: amount}, lcia_method)
            lca.lci(factorize=True)
            lca.lcia()
            total_score = lca.score
        else:
            #lca.decompose_technosphere()

           # _log.debug("Contributions to %s", demand)
            lca.redo_lci(demand)
            lca.redo_lcia(demand)
            
            score= lca.score

            #grouper = ScoreGrouper(lca)
            #contributions = grouper(final_activities)
            results.append(score)

    return results


def collect_results(
    lca,
    demands,
    final_activities,
    num_samples,
    method_label,
    demand_labels,
    component_order,
    database_name,
    bio_indices,
    tech_indices,
    include_only_specific_bio_uncertainty=True,
    include_only_specific_exc_uncertainty=True,



    **kwargs
):
    """Repeatedly call `sample_comparative_contribution` and collect results in
    a DataFrame."""
    samples = []
    for iteration in range(num_samples):
        results = sample_results(
            lca, 
            demands, 
            final_activities, 
            database_name, 
            bio_indices,
            tech_indices,
            include_only_specific_bio_uncertainty,# exclude_tech_background=exclude_tech_background, 
            include_only_specific_exc_uncertainty,
            **kwargs
            )
        #for label, result in zip(demands, results):
        for result in results:
            #order = list(component_order) + [
            #    k for k in result if k not in component_order
            #]
            samples.extend(
                [(demand_labels[0], 
                  #k, 
                  method_label[1], 
                  iteration,  
                  result
                  #result.get(k, 0)) for k in order
            )]
            )
    return pd.DataFrame(
        samples, columns=["scenario",  "method", "iteration",  "score"]
    )


