from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from functools import partial
from typing import List, Optional
import argparse
import time_details as T
import time

from vrp.problem_structures import VRPProblemDefinition

def _format_minutes(value: int) -> str:
    hours, minutes = divmod(int(value), 60)
    return f"{hours:02d}:{minutes:02d}"


def _ensure_minutes_matrix(matrix: List[List[Optional[float]]]) -> List[List[int]]:
    """Convert any duration matrix to integer minutes."""

    if matrix is None:
        raise ValueError("time_matrix must be provided")

    seconds_like = any(
        value is not None and float(value) > 60.0
        for row in matrix
        for value in row
    )

    minutes_matrix: List[List[int]] = []
    for row in matrix:
        minutes_row: List[int] = []
        for value in row:
            if value is None:
                minutes_row.append(0)
                continue
            minutes = float(value) / 60.0 if seconds_like else float(value)
            if minutes > 0:
                minutes_row.append(max(1, int(round(minutes))))
            else:
                minutes_row.append(0)
        minutes_matrix.append(minutes_row)
    return minutes_matrix


def solve_vrp_problem_definition(
    problem: VRPProblemDefinition,
    *,
    days : int =3,
    start_hour: int = 6,
    end_hour: int = 16,
    service_fallback_minutes: int = 120,
    timelimit_seconds: int = 30,
    debug: bool = False,
    guided_local_search: bool = True,
):
    """Solve the multi-day VRP instance defined by ``problem``.

    The implementation mirrors the existing command line solver but takes the
    structured :class:`~vrp.problem_structures.VRPProblemDefinition` so other
    modules can easily feed data without going through the ``time_details``
    helper module.
    """
    vehicles = len(problem.worker_constraints)

    if days <= 0:
        raise ValueError("days must be a positive integer")
    if vehicles <= 0:
        raise ValueError("vehicles must be a positive integer")

    day_start = start_hour * 60
    day_end = end_hour * 60
    base_matrix = _ensure_minutes_matrix(problem.time_matrix)
    matrix_size = len(base_matrix)

    service_fallback = max(service_fallback_minutes, 0)
    service_times: List[int] = [0] * matrix_size
    for node in problem.nodes:
        if 0 <= node.node_id < matrix_size:
            duration = int(round(node.service_duration.total_seconds() / 60.0))
            service_times[node.node_id] = duration if duration > 0 else service_fallback

    for idx in range(1, matrix_size):
        if service_times[idx] <= 0:
            service_times[idx] = service_fallback

    t0 = time.perf_counter()
    overnight_time = (day_start - day_end)
    num_nodes = matrix_size
    num_overnights = days - 1

    night_nodes_by_vehicle: List[List[int]] = []
    morning_nodes_by_vehicle: List[List[int]] = []
    all_night_nodes: List[int] = []
    all_morning_nodes: List[int] = []

    current_node_index = num_nodes
    for v in range(vehicles):
        vehicle_nights = list(range(current_node_index, current_node_index + num_overnights))
        current_node_index += num_overnights
        vehicle_mornings = list(range(current_node_index, current_node_index + num_overnights))
        current_node_index += num_overnights

        night_nodes_by_vehicle.append(vehicle_nights)
        morning_nodes_by_vehicle.append(vehicle_mornings)
        all_night_nodes.extend(vehicle_nights)
        all_morning_nodes.extend(vehicle_mornings)

    total_nodes = num_nodes + len(all_night_nodes) + len(all_morning_nodes)

    starts = [0] * vehicles
    ends = [0] * vehicles

    manager = pywrapcp.RoutingIndexManager(total_nodes, vehicles, starts, ends)
    model_parameters = pywrapcp.DefaultRoutingModelParameters()
    model_parameters.max_callback_cache_size = 2 * total_nodes * total_nodes
    routing = pywrapcp.RoutingModel(manager, model_parameters)

    def _matrix_index(node: int) -> int:
        return node if node < num_nodes else 0

    def transit_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        infinite_time = (day_end - day_start + service_fallback) * 1000

        if all_morning_nodes and from_node in all_night_nodes and to_node not in all_morning_nodes:
            return infinite_time
        if from_node in all_morning_nodes and to_node in all_night_nodes:
            return infinite_time
        if from_node in all_morning_nodes and to_node in all_morning_nodes:
            return infinite_time
        if from_node in all_night_nodes and to_node in all_night_nodes:
            return infinite_time

        return int(base_matrix[_matrix_index(from_node)][_matrix_index(to_node)])

    transit_callback_index = routing.RegisterTransitCallback(transit_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        service_time = 0
        if from_node in all_night_nodes:
            service_time = overnight_time
        elif from_node in all_morning_nodes or from_node == 0:
            service_time = 0
        else:
            service_time = service_times[_matrix_index(from_node)]

        return int(base_matrix[_matrix_index(from_node)][_matrix_index(to_node)] + service_time)

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        slack_max=(day_end - day_start),
        capacity=day_end,
        fix_start_cumul_to_zero=False,
        name='Time')

    time_dimension = routing.GetDimensionOrDie('Time')
    for v in range(vehicles):
        sv_start = time_dimension.SlackVar(routing.Start(v))
        if sv_start is not None:
            sv_start.SetValue(0)

    for node in problem.nodes:
        index = manager.NodeToIndex(node.node_id)
        tw = node.time_window
        start = day_start
        end = day_end
        if tw.start is not None and tw.end is not None:
            start = max(day_start, int(tw.start))
            end = min(day_end, int(tw.end))
        if start > end:
            start, end = day_start, day_end
        time_dimension.CumulVar(index).SetRange(start, end)
    for node in all_morning_nodes:
        index = manager.NodeToIndex(node)
        time_dimension.CumulVar(index).SetRange(day_start, day_start)
    for node in all_night_nodes:
        index = manager.NodeToIndex(node)
        time_dimension.CumulVar(index).SetRange(day_end, day_end)

    routing.AddConstantDimension(1, total_nodes + 1, True, "Counting")
    count_dimension = routing.GetDimensionOrDie('Counting')

    for v in range(vehicles):
        for node in morning_nodes_by_vehicle[v] + night_nodes_by_vehicle[v]:
            routing.SetAllowedVehiclesForIndex([v], manager.NodeToIndex(node))

    def day_transit_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        return 1 if from_node in all_night_nodes else 0

    day_cb = routing.RegisterTransitCallback(day_transit_callback)
    routing.AddDimension(day_cb, 0, days, True, "Day")
    day_dim = routing.GetDimensionOrDie("Day")

    for v in range(vehicles):
        day_dim.CumulVar(routing.Start(v)).SetRange(0, 0)
        for d, m in enumerate(morning_nodes_by_vehicle[v]):
            day_dim.CumulVar(manager.NodeToIndex(m)).SetRange(d, d)
        for d, n in enumerate(night_nodes_by_vehicle[v]):
            day_dim.CumulVar(manager.NodeToIndex(n)).SetRange(d, d)

    for node in problem.nodes:
        if node.time_window.day is not None:
            day_dim.CumulVar(manager.NodeToIndex(node.node_id)).SetRange(node.time_window.day, node.time_window.day)

    def workload_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node in all_night_nodes:
            service_time = 0
        elif from_node in all_morning_nodes or from_node == 0:
            service_time = 0
        else:
            service_time = service_times[_matrix_index(from_node)]
        travel_time = base_matrix[_matrix_index(from_node)][_matrix_index(to_node)]
        return int(service_time + travel_time)

    workload_cb_index = routing.RegisterTransitCallback(workload_callback)
    max_capacity = day_end
    for constraints in problem.worker_constraints:
        for bound in (constraints.max_daily_minutes, constraints.max_weekly_minutes):
            if bound:
                max_capacity = max(max_capacity, int(bound))
    routing.AddDimension(
        workload_cb_index,
        slack_max=0,
        capacity=max_capacity,
        fix_start_cumul_to_zero=True,
        name="Workload",
    )
    workload_dim = routing.GetDimensionOrDie("Workload")

    for vehicle_id, constraints in enumerate(problem.worker_constraints):
        upper_bound = None
        if constraints.max_daily_minutes is not None:
            upper_bound = constraints.max_daily_minutes
        elif constraints.max_weekly_minutes is not None:
            upper_bound = constraints.max_weekly_minutes
        if upper_bound is not None:
            workload_dim.CumulVar(routing.End(vehicle_id)).SetRange(0, int(upper_bound))

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    if guided_local_search:
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = timelimit_seconds
    search_parameters.log_search = debug

    t_setup = time.perf_counter()
    print(f"[TIMING] Setup time: {t_setup - t0:.3f} s")
    print("Solver started...")
    t1 = time.perf_counter()
    solution = routing.SolveWithParameters(search_parameters)
    t2 = time.perf_counter()
    print(f"[TIMING] Solve time: {t2 - t1:.3f} s")

    if not solution:
        print("no solution found")
        return None

    print("solution found.  Objective value is ", solution.ObjectiveValue())
    result = {
        'Dropped': [],
        'Scheduled': [],
    }
    print('Scheduled Routes:')
    print('[node, order, min time, max time]')
    for vehicle_id in range(vehicles):
        print(f"========= Vehicle {vehicle_id} =========")
        vehicle_schedule = []
        index = routing.Start(vehicle_id)
        if routing.IsEnd(index):
            print("Vehicle not used.")
            continue

        while not routing.IsEnd(index):
            cumultime = time_dimension.CumulVar(index)
            count = count_dimension.CumulVar(index)
            node = manager.IndexToNode(index)
            node_str = node
            if node in all_night_nodes:
                node_str = 'Overnight at {}, dummy for 1'.format(node)
            elif node in all_morning_nodes:
                node_str = 'Starting day at {}, dummy for 1'.format(node)

            mintime = solution.Min(cumultime)
            maxtime = solution.Max(cumultime)
            line_data = [node_str, solution.Value(count), _format_minutes(mintime), _format_minutes(maxtime)]
            print(line_data)
            vehicle_schedule.append(line_data)
            index = solution.Value(routing.NextVar(index))

        cumultime = time_dimension.CumulVar(index)
        count = count_dimension.CumulVar(index)
        mintime = solution.Min(cumultime)
        maxtime = solution.Max(cumultime)
        line_data = [manager.IndexToNode(index), solution.Value(count), _format_minutes(mintime), _format_minutes(maxtime)]
        print(line_data)
        vehicle_schedule.append(line_data)
        result['Scheduled'].append(vehicle_schedule)

    print('Dropped')
    print(result['Dropped'])
    print('Scheduled')
    return result

def main():
  
#################### !!!!! PARSER !!!!! ####################
    parser = argparse.ArgumentParser(description='Solve routing problem to a fixed set of destinations')
    parser.add_argument('--days', type=int, dest='days', default=2,
                        help='Number of days to schedule.  Default is 2 days')
    parser.add_argument('--start', type=int, dest='start', default=6,
                        help='The earliest any trip can start on any day, in hours. Default 6')
    parser.add_argument('--end', type=int, dest='end', default=18,
                        help='The earliest any trip can end on any day, in hours.  Default 18 (which is 6pm)')
    parser.add_argument('--waittime', type=int, dest='service', default=120,
                        help='The time required to wait at each visited node, in minutes.  Default is 30')
    parser.add_argument('-t, --timelimit', type=int, dest='timelimit', default=30,
                        help='Maximum run time for solver, in seconds.  Default is 10 seconds.')
    parser.add_argument('--debug', action='store_true', dest='debug', default=False,
                        help="Turn on solver logging.")
    parser.add_argument('--no_guided_local', action='store_true', dest='guided_local_off',
                        default=False,
                        help='Whether or not to use the guided local search metaheuristic')
    parser.add_argument('--skip_mornings', action='store_true', dest='skip_mornings',
                        default=False,
                        help='Whether or not to use dummy morning nodes.  Default is true')
    parser.add_argument('--vehicles', type=int, dest='vehicles', default=4,
                        help='Number of vehicles to schedule. Default is 2')

    args = parser.parse_args()
    day_start = args.start * 3600
    day_end = args.end * 3600

    if args.days <= 0:
        print("--days parameter must be 1 or more")
        assert args.days > 0
        
    
    num_vehicles = args.vehicles
    if num_vehicles <= 0:
        print("--vehicles parameter must be 1 or more")
        assert num_vehicles > 0   

#################### !!!!! END PARSER !!!!! ####################

#################### INITIALIZATION OF LISTS AND VARS ############################
    t0 = time.perf_counter()
    num_days = args.days - 1

    node_service_time = args.service * 60
    overnight_time =   (day_start - day_end) # -18*3600 #

    num_nodes = T.num_nodes()
    num_overnights = args.days - 1  # EXAMPLE: 4 overnights for 5 days

    # Each vehicle has its own set of overnight and morning dummy nodes
    # nested lists: vehicle -> list of dummy nodes
    night_nodes_by_vehicle = []
    morning_nodes_by_vehicle = []
    
    all_night_nodes = []
    all_morning_nodes = []
    
    
    # Create the dummy nodes for each vehicle
    current_node_index = num_nodes # start at real nodes count 
    for v in range(num_vehicles):
        vehicle_nights = list(range(current_node_index, current_node_index + num_overnights))
        current_node_index += num_overnights
        
        vehicle_mornings = list(range(current_node_index, current_node_index + num_overnights))
        current_node_index += num_overnights
        
        night_nodes_by_vehicle.append(vehicle_nights)
        morning_nodes_by_vehicle.append(vehicle_mornings)
        
        all_night_nodes.extend(vehicle_nights)
        all_morning_nodes.extend(vehicle_mornings)

    total_nodes = num_nodes + len(all_night_nodes) + len(all_morning_nodes)


    # multiple vehicles, each has a start and end at depot 
    starts = [0] * num_vehicles
    ends = [1] * num_vehicles
    
    # Create Routing Index Manager for multiple vehicles
    manager = pywrapcp.RoutingIndexManager(total_nodes, num_vehicles, starts, ends)
    print('made manager with total nodes {} = {} + {} + {}'.format(total_nodes,num_nodes,len(all_night_nodes),len(all_morning_nodes)))
    
    # Create Routing Model.
    # use precaching on OR-Tools side.  So Much Faster # from jmarca - i dont know what this does
    model_parameters = pywrapcp.DefaultRoutingModelParameters()
    model_parameters.max_callback_cache_size = 2 * total_nodes * total_nodes
    # creates the routing model object
    routing = pywrapcp.RoutingModel(manager, model_parameters)
    print('made routing model')
    
    # Create and register a transit callback. This callback returns the travel time between two nodes.
    # T is the time matrix module imported
    transit_callback_fn = partial(T.transit_callback,
                                  manager,
                                  day_end,
                                  all_night_nodes,
                                  all_morning_nodes)
    
    # this index is used to identify the callback
    transit_callback_index = routing.RegisterTransitCallback(transit_callback_fn)
    
    # Set the cost of travel for each arc using the callback.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    print('set the arc cost evaluator for all vehicles')

    # Create and register a time callback. This callback returns service time + travel time between two nodes.
    time_callback_fn = partial(T.time_callback,
                               manager,
                               node_service_time,
                               overnight_time,
                               all_night_nodes,
                               all_morning_nodes)

    # this index is used to identify the callback for the time 
    time_callback_index = routing.RegisterTransitCallback(time_callback_fn)
    
    # create time dimension. slack is the waiting time at each node and slack-max is the entire day length
    routing.AddDimension(
        time_callback_index,
        slack_max=(day_end - day_start),
        capacity=day_end,
        fix_start_cumul_to_zero=False,
        name='Time')
    
    # Get the time dimension object
    time_dimension = routing.GetDimensionOrDie('Time')

    # Allow waiting at customers; fix no-wait at depot STARTS only (End may have no SlackVar in some builds)
    for v in range(num_vehicles):
        sv_start = time_dimension.SlackVar(routing.Start(v))
        if sv_start is not None:
            sv_start.SetValue(0)  # Do NOT call SlackVar on routing.End(v); some OR-Tools builds return None there.
    
    print('created time dimension')
    
    # time windows are node specific !! Here implement job-specific time windows LATER
    # Add time window constraints for each regular node
    for node in range(2,num_nodes):
      index = manager.NodeToIndex(node)
      time_dimension.CumulVar(index).SetRange(day_start, day_end)

    # Add time window constraints for dummy nodes. These are generic for day logic
    # This also applies to the overnight nodes and morning nodes
    # node list starts at num_nodes up to total_nodes, where the dummy nodes are
    for node in all_morning_nodes:
      index = manager.NodeToIndex(node)
      time_dimension.CumulVar(index).SetRange(day_start, day_start)
    for node in all_night_nodes:
      index = manager.NodeToIndex(node)
      time_dimension.CumulVar(index).SetRange(day_end,day_end)

    print('done with time constraints')

    # Countin dimension g to count days
    routing.AddConstantDimension(1,  # increment by 1
                                 total_nodes+1, # the max count is visit every node
                                 True, # start count at zero
                                 "Counting")
    count_dimension = routing.GetDimensionOrDie('Counting')

    print('created count dim')
    
    # Set vehicle constraints to only visit their own dummy nodes        
    for v in range(num_vehicles):
      for node in morning_nodes_by_vehicle[v] + night_nodes_by_vehicle[v]:
        routing.SetAllowedVehiclesForIndex([v], manager.NodeToIndex(node))
    
    # Day transition dimension to enforce day logic
    def day_transit_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        return 1 if from_node in all_night_nodes else 0

    # day callback function and dimension
    day_cb = routing.RegisterTransitCallback(day_transit_callback)
    routing.AddDimension(day_cb, 0, args.days, True, "Day")
    day_dim = routing.GetDimensionOrDie("Day")

    # Enforce day logic by constraining the day dimension at dummy nodes
    for v in range(num_vehicles):
        day_dim.CumulVar(routing.Start(v)).SetRange(0, 0)
        for d, m in enumerate(morning_nodes_by_vehicle[v]):
            day_dim.CumulVar(manager.NodeToIndex(m)).SetRange(d, d)
        for d, n in enumerate(night_nodes_by_vehicle[v]):
            day_dim.CumulVar(manager.NodeToIndex(n)).SetRange(d, d)

    print('done setting up vehicle/dummy node constraints')


    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC) 30 seconds
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    print('set up the setup.  Total nodes is ', total_nodes, ' and real nodes is ', num_nodes)
    # Setting local search metaheuristics:
    if not args.guided_local_off:
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = args.timelimit
    search_parameters.log_search = args.debug

    # Solve the problem.
    t_setup = time.perf_counter()
    print(f"[TIMING] Setup time: {t_setup - t0:.3f} s")
    print("Solver started...")
    t1 = time.perf_counter()
    solution = routing.SolveWithParameters(search_parameters)
    t2 = time.perf_counter()
    print(f"[TIMING] Solve time: {t2 - t1:.3f} s")
    if not solution:
      print("no solution found")
    else:
      print("solution found.  Objective value is ",solution.ObjectiveValue())

      # Print the results
      result = {
        'Dropped': [],
        'Scheduled': [] # This will now be a list of lists (one list per vehicle)
      }

      # Return the dropped locations
      # for index in range(routing.Size()):
      #   if routing.IsStart(index) or routing.IsEnd(index):
      #     continue
      #   node = manager.IndexToNode(index)
      #   if node in all_night_nodes or node in all_morning_nodes:
      #     continue
      #   if solution.Value(routing.NextVar(index)) == index:
      #     result['Dropped'].append(node)
      
      # print('Dropped Nodes:')
      # print(result['Dropped'])
      # print('---')
      ## contiunue here !! # ##. #

      # Return the scheduled locations for ALL vehicles
      print('Scheduled Routes:')
      print('[node, order, min time, max time]')
      
      for vehicle_id in range(num_vehicles):
          print(f"========= Vehicle {vehicle_id} =========")
          vehicle_schedule = []
          index = routing.Start(vehicle_id)

          # Skip unused vehicles
          if routing.IsEnd(index):
              print("Vehicle not used.")
              continue
          
          while not routing.IsEnd(index):
            cumultime = time_dimension.CumulVar(index)
            count = count_dimension.CumulVar(index)
            node = manager.IndexToNode(index)
            
            node_str = node # Default
            if node in all_night_nodes:
              node_str = 'Overnight at {}, dummy for 1'.format(node)
            if node in all_morning_nodes:
              node_str = 'Starting day at {}, dummy for 1'.format(node)

            mintime = solution.Min(cumultime)
            maxtime = solution.Max(cumultime)

            line_data = [node_str, solution.Value(count),
                         _format_minutes(mintime),
                         _format_minutes(maxtime)]
            
            print(line_data)
            vehicle_schedule.append(line_data)
            index = solution.Value(routing.NextVar(index))
          # Add the end node for this vehicle
          cumultime = time_dimension.CumulVar(index)
          count = count_dimension.CumulVar(index)
          mintime = solution.Min(cumultime)
          maxtime = solution.Max(cumultime)

          line_data = [manager.IndexToNode(index),
                       solution.Value(count),
                       _format_minutes(mintime),
                       _format_minutes(maxtime)]
          print(line_data)
          vehicle_schedule.append(line_data)    

          # Add this vehicle's schedule to the main result
          result['Scheduled'].append(vehicle_schedule)

      print('Dropped')
      print(result['Dropped'])

      print('Scheduled')
      #print('[node, order, min time, max time]')
      #for line in result['Scheduled']:
      #  print(line)

      #print(result)

if __name__ == '__main__':
    main()
