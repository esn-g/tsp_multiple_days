from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from functools import partial
from datetime import datetime, time, timedelta
import argparse
import time_details as T

def timedelta_format(td):
    ts = int(td.total_seconds())
    tm = time(hour=(ts//3600),second=(ts%3600//60))
    return tm.strftime("%H:%S")

def main():

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

    num_days = args.days - 1

    node_service_time = args.service * 60
    overnight_time =   (day_start - day_end) # -18*3600 #

    # disjunction_penalty = 10000000

    Slack_Max = (day_end - day_start) - day_start # night node demand minus no-work day
    #  3600*24
    Capacity = day_end # most time that can be collected in one day

    num_nodes = T.num_nodes()
    num_overnights = args.days - 1  # 4 overnights for 5 days

    night_nodes_by_vehicle = []
    morning_nodes_by_vehicle = []
    
    all_night_nodes = []
    all_morning_nodes = []
    
    current_node_index = num_nodes # Start indexing from 42
    
    for v in range(num_vehicles):
        vehicle_nights = list(range(current_node_index, current_node_index + num_overnights))
        current_node_index += num_overnights
        
        vehicle_mornings = []
        if not args.skip_mornings:
            vehicle_mornings = list(range(current_node_index, current_node_index + num_overnights))
            current_node_index += num_overnights
        
        night_nodes_by_vehicle.append(vehicle_nights)
        morning_nodes_by_vehicle.append(vehicle_mornings)
        
        all_night_nodes.extend(vehicle_nights)
        all_morning_nodes.extend(vehicle_mornings)

    total_nodes = num_nodes + len(all_night_nodes) + len(all_morning_nodes)

    # Create the routing index manager.
    
    # multiple vehicles
    starts = [0] * num_vehicles
    ends = [1] * num_vehicles
    
    #manager = pywrapcp.RoutingIndexManager(total_nodes, 1, [0], [1])
    manager = pywrapcp.RoutingIndexManager(total_nodes, num_vehicles, starts, ends)
    

    print('made manager with total nodes {} = {} + {} + {}'.format(total_nodes,
                                                                   num_nodes,
                                                                   len(all_night_nodes),
                                                                   len(all_morning_nodes)))
    # Create Routing Model.
    # use precaching on OR-Tools side.  So Much Faster
    model_parameters = pywrapcp.DefaultRoutingModelParameters()
    model_parameters.max_callback_cache_size = 2 * total_nodes * total_nodes
    routing = pywrapcp.RoutingModel(manager, model_parameters)
    # routing = pywrapcp.RoutingModel(manager)

    transit_callback_fn = partial(T.transit_callback,
                                  manager,
                                  day_end,
                                  all_night_nodes,
                                  all_morning_nodes)

    transit_callback_index = routing.RegisterTransitCallback(transit_callback_fn)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    print('set the arc cost evaluator for all vehicles')

    time_callback_fn = partial(T.time_callback,
                               manager,
                               node_service_time,
                               overnight_time,
                               all_night_nodes,
                               all_morning_nodes)

    time_callback_index = routing.RegisterTransitCallback(time_callback_fn)


    # Define cost of each arc.

    # create time dimension
    routing.AddDimension(
        time_callback_index,
        Slack_Max,  # An upper bound for slack (the wait times at the locations).
        Capacity,  # An upper bound for the total time over each vehicle's route.
        False,  # Determine whether the cumulative variable is set to zero at the start of the vehicle's route.
        'Time')
    time_dimension = routing.GetDimensionOrDie('Time')

    print('created time dimension')

    # get rid of slack for all regular nodes, all morning nodes
    # but keep for depot, night nodes
    for node in range(2, num_nodes):
      index = manager.NodeToIndex(node)
      time_dimension.SlackVar(index).SetValue(0)
    for node in all_morning_nodes:
      index = manager.NodeToIndex(node)
      time_dimension.SlackVar(index).SetValue(0)


    # Allow all locations except the first two to be droppable.
    #for node in range(2, num_nodes):
    #  routing.AddDisjunction([manager.NodeToIndex(node)], disjunction_penalty)

    # Allow all overnight nodes to be dropped for free
    #for node in all_night_nodes:
    #  routing.AddDisjunction([manager.NodeToIndex(node)], 0)
    #for node in all_morning_nodes:
    #  routing.AddDisjunction([manager.NodeToIndex(node)], 0)

    # Add time window constraints for each regular node
    for node in range(2,num_nodes):
      index = manager.NodeToIndex(node)
      time_dimension.CumulVar(index).SetRange(day_start, day_end)

    # This also applies to the overnight nodes and morning nodes
    for node in range(num_nodes, total_nodes):
      index = manager.NodeToIndex(node)
      time_dimension.CumulVar(index).SetRange(day_start, day_end)


    # Add time window constraints for each vehicle start/end node.
    for veh in range(num_vehicles):
      index = routing.Start(veh)
      time_dimension.CumulVar(index).SetMin(day_start)
      index = routing.End(veh)
      time_dimension.CumulVar(index).SetMax(day_end)

    print('done with time constraints')

    # make sure the days happen in order. first end day 1, end day 2, etc, then node 1
    # create counting dimension
    routing.AddConstantDimension(1,  # increment by 1
                                 total_nodes+1, # the max count is visit every node
                                 True, # start count at zero
                                 "Counting")
    count_dimension = routing.GetDimensionOrDie('Counting')

    print('created count dim')

    # use count dim to enforce ordering of overnight, morning nodes
    solver = routing.solver()
    
    # NEW CONSTRAINT: Ensure vehicles only visit their *own* dummy nodes
    for v in range(num_vehicles):
        # Get all dummy nodes that *don't* belong to this vehicle
        disallowed_dummies = []
        for v_other in range(num_vehicles):
            if v == v_other:
                continue
            disallowed_dummies.extend(night_nodes_by_vehicle[v_other])
            disallowed_dummies.extend(morning_nodes_by_vehicle[v_other])

        for dummy_node in disallowed_dummies:
            dummy_index = manager.NodeToIndex(dummy_node)
            # Add a constraint that vehicle 'v' cannot be the
            # vehicle assigned to visit this dummy_index
            solver.Add(routing.VehicleVar(dummy_index) != v)

    print('done setting up vehicle/dummy node constraints')
    
    # use count dim to enforce ordering of overnight, morning nodes
    # --- THIS IS ALL WRAPPED IN A 'for v' LOOP NOW ---
    for v in range(num_vehicles):
        vehicle_night_nodes = night_nodes_by_vehicle[v]
        vehicle_morning_nodes = morning_nodes_by_vehicle[v]

        for i in range(len(vehicle_night_nodes)):
          inode = vehicle_night_nodes[i]
          iidx = manager.NodeToIndex(inode)
          iactive = routing.ActiveVar(iidx)

          for j in range(i+1, len(vehicle_night_nodes)):
            # make i come before j using count dimension
            jnode = vehicle_night_nodes[j]
            jidx = manager.NodeToIndex(jnode)
            jactive = routing.ActiveVar(jidx)
            solver.Add(iactive >= jactive)
            solver.Add(count_dimension.CumulVar(iidx) * iactive * jactive <=
                       count_dimension.CumulVar(jidx) * iactive * jactive)

          # if night node is active, AND night_node is not the last night,
          # must transition to corresponding morning node
          if i < len(vehicle_morning_nodes):
            i_morning_idx = manager.NodeToIndex(vehicle_morning_nodes[i])
            i_morning_active = routing.ActiveVar(i_morning_idx)
            solver.Add(iactive == i_morning_active)
            solver.Add(count_dimension.CumulVar(iidx) + 1 ==
                       count_dimension.CumulVar(i_morning_idx))

        for i in range(len(vehicle_morning_nodes)):
          inode = vehicle_morning_nodes[i]
          iidx = manager.NodeToIndex(inode)
          iactive = routing.ActiveVar(iidx)

          for j in range(i+1, len(vehicle_morning_nodes)):
            # make i come before j using count dimension
            jnode = vehicle_morning_nodes[j]
            jidx = manager.NodeToIndex(jnode)
            jactive = routing.ActiveVar(jidx)

            solver.Add(iactive >= jactive)
            solver.Add(count_dimension.CumulVar(iidx) * iactive * jactive <=
                       count_dimension.CumulVar(jidx) * iactive * jactive)


    print('done setting up ordering constraints between days')


    # Instantiate route start and end times to produce feasible times.
    # routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(0)))
    # routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(0)))


    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    #search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    print('set up the setup.  Total nodes is ', total_nodes, ' and real nodes is ', num_nodes)
    # Setting local search metaheuristics:
    if not args.guided_local_off:
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = args.timelimit
    search_parameters.log_search = args.debug

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
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
      for index in range(routing.Size()):
        if routing.IsStart(index) or routing.IsEnd(index):
          continue
        node = manager.IndexToNode(index)
        if node in all_night_nodes or node in all_morning_nodes:
          continue
        if solution.Value(routing.NextVar(index)) == index:
          result['Dropped'].append(node)
      
      print('Dropped Nodes:')
      print(result['Dropped'])
      print('---')
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

            mintime = timedelta(seconds=solution.Min(cumultime))
            maxtime = timedelta(seconds=solution.Max(cumultime))
            
            line_data = [node_str, solution.Value(count),
                         timedelta_format(mintime),
                         timedelta_format(maxtime)]
            
            print(line_data)
            vehicle_schedule.append(line_data)
            index = solution.Value(routing.NextVar(index))
          # Add the end node for this vehicle
          cumultime = time_dimension.CumulVar(index)
          count = count_dimension.CumulVar(index)
          mintime = timedelta(seconds=solution.Min(cumultime))
          maxtime = timedelta(seconds=solution.Max(cumultime))
          
          line_data = [manager.IndexToNode(index),
                       solution.Value(count),
                       timedelta_format(mintime),
                       timedelta_format(maxtime)]
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
