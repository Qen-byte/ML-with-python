import heapq

# A* Search Algorithm
def a_star_search(start, goal, graph, h):
    # Priority queue to store (cost, current_node)
    open_list = []
    heapq.heappush(open_list, (0, start))
    
    # Dictionary to store the cost from start to a given node
    g_cost = {start: 0}
    
    # Dictionary to store the path (came from)
    came_from = {start: None}
    
    while open_list:
        # Get the node with the lowest cost (priority)
        current_cost, current_node = heapq.heappop(open_list)
        
        # If we reached the goal, reconstruct the path
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            return path[::-1]  # Reverse the path to get start -> goal
        
        # Explore neighbors
        for neighbor, cost in graph[current_node]:
            tentative_g_cost = g_cost[current_node] + cost
            
            # If the new cost to the neighbor is lower than the previous cost
            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + h(neighbor, goal)  # g_cost + heuristic
                heapq.heappush(open_list, (f_cost, neighbor))
                came_from[neighbor] = current_node
    
    return None  # Return None if no path is found

# Example usage:

# Define a heuristic function (h)
def heuristic(node, goal):
    # In this case, we'll use a simple placeholder heuristic
    return abs(goal - node)

# Define a graph as an adjacency list
graph = {
    0: [(1, 1), (2, 4)],
    1: [(2, 2), (3, 5)],
    2: [(3, 1)],
    3: []
}

# Perform A* Search
start_node = 0
goal_node = 3
path = a_star_search(start_node, goal_node, graph, heuristic)

print("Path from start to goal:", path)
